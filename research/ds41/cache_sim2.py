"""cache_sim2.py TRACE_DIR [CAPACITY]: RAM-cache strategies the first replay (cache_sim.py) did not
try, on the same per-rank access traces, at CAPACITY units a rank (1364 at 96 GiB):

  lru            the engine today: one LRU a rank, every access touches, every miss is admitted
  lru-shared     one LRU over the four ranks' units (4 x CAPACITY), accesses merged in step order
  lip-prefill    misses of multi-token passes enter at the LRU end (evicted first unless reused);
                 decode misses and every hit go to the front
  tinylfu        a miss is admitted only if its use count beats the would-be victim's (counts
                 halved every 8 x CAPACITY accesses); hits touch as in LRU
  +7%            lru with 7% more capacity (the pinned host pool's staging, if it is idle)

Prints misses in decode (one-token) passes and in all passes, summed over the ranks.
"""
import collections
import os
import struct
import sys


def load(path):
    out = []
    data = open(path, "rb").read()
    i = 0
    while i + 12 <= len(data):
        p, = struct.unpack_from("<I", data, i)
        layer, tok, n, miss = struct.unpack_from("<4H", data, i + 4)
        units = struct.unpack_from("<%di" % n, data, i + 12)
        out.append((p, layer, tok, units))
        i += 12 + 4 * n
    return out


class LRU:
    def __init__(self, cap):
        self.cap, self.d = cap, collections.OrderedDict()

    def access(self, u, one):
        if u in self.d:
            self.d.move_to_end(u)
            return True
        self.d[u] = 1
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return False


class LIPPrefill(LRU):
    def access(self, u, one):
        if u in self.d:
            self.d.move_to_end(u)
            return True
        self.d[u] = 1
        if not one:
            self.d.move_to_end(u, last=False)   # the LRU end: first out unless reused
        if len(self.d) > self.cap:
            # never evict what was just inserted at the LRU end in favour of itself
            victim = next(iter(self.d))
            if victim == u and len(self.d) > 1:
                it = iter(self.d); next(it); victim = next(it)
            del self.d[victim]
        return False


class TinyLFU(LRU):
    def __init__(self, cap):
        super().__init__(cap)
        self.freq = collections.Counter()
        self.n = 0

    def access(self, u, one):
        self.freq[u] += 1
        self.n += 1
        if self.n >= 8 * self.cap:
            self.n = 0
            for k in list(self.freq):
                self.freq[k] //= 2
                if not self.freq[k]:
                    del self.freq[k]
        if u in self.d:
            self.d.move_to_end(u)
            return True
        if len(self.d) < self.cap:
            self.d[u] = 1
            return False
        victim = next(iter(self.d))
        if self.freq[u] > self.freq[victim]:
            del self.d[victim]
            self.d[u] = 1
        return False


def main():
    d = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 1364
    ranks = [load(os.path.join(d, "rank%d.bin" % r)) for r in range(4)]
    merged = sorted(((p, layer, r, tok, units) for r in range(4) for (p, layer, tok, units) in ranks[r]),
                    key=lambda x: (x[0], x[1], x[2]))

    def run(make, shared=False, scale=1.0):
        c = int(cap * scale)
        caches = [make(4 * c)] if shared else [make(c) for _ in range(4)]
        dm = am = da = aa = 0
        for p, layer, r, tok, units in merged:
            cache = caches[0] if shared else caches[r]
            one = tok == 1
            for u in units:
                hit = cache.access((r, u), one)
                aa += 1; am += not hit
                if one:
                    da += 1; dm += not hit
        return dm, da, am, aa

    rows = [("lru", lambda: run(LRU)), ("lru-shared", lambda: run(LRU, shared=True)),
            ("lip-prefill", lambda: run(LIPPrefill)), ("lip-prefill-shared", lambda: run(LIPPrefill, shared=True)),
            ("tinylfu", lambda: run(TinyLFU)), ("tinylfu-shared", lambda: run(TinyLFU, shared=True)),
            ("lru +7%", lambda: run(LRU, scale=1.07)), ("lru-shared +7%", lambda: run(LRU, shared=True, scale=1.07))]
    base = None
    print("capacity %d units a rank; misses summed over 4 ranks" % cap)
    for name, f in rows:
        dm, da, am, aa = f()
        if base is None:
            base = dm
        print("%-20s decode misses %6d of %7d (%.2f%%, %+.1f%% vs lru)   all passes %6d of %7d (%.2f%%)" % (
            name, dm, da, 100.0 * dm / da, 100.0 * (dm - base) / base, am, aa, 100.0 * am / aa))
        sys.stdout.flush()


main()
