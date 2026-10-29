"""Replays a RADIANCE_EXPERIMENTAL_DS41_ACCESS_TRACE capture against RAM-cache policies.

  cache_sim.py TRACE_DIR [CAPACITY]

Each rank<r>.bin holds one record per EP-routed layer pass: u32 pass, u16 layer, u16 tokens,
u16 units, u16 drive-read weights, then the selected file-tier units (i32 plan unit ids). The
engine's mapped cache keeps gate_up and down in two same-capacity classes that a unit always uses
together, so a unit-level cache of CAPACITY units (1137 a rank at 80 GiB) models it.

Prints, per rank and policy, the misses (units read from the drive) in all passes and in one-token
passes, against what the engine actually read. LRU reproduces the engine's policy, the check that
the replay is right; the others bound what a different policy would save:

  lru          the engine's policy: every access touches, every miss is admitted
  lru-noprefill  passes of more than one token neither touch nor admit
  slru         segmented LRU: a hit promotes into a protected 80% segment
  2q           Johnson & Shasha 2Q: A1in FIFO (25%), A1out ghost (50%), Am LRU
  lfu-aged     LFU with counts halved every CAPACITY accesses, LRU among equals
  opt          Belady's optimum (needs the future: the bound for any online policy)
"""
import collections
import heapq
import os
import struct
import sys


def load(path):
    recs = []
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while i + 12 <= len(data):
        p, = struct.unpack_from("<I", data, i)
        layer, tok, n, miss = struct.unpack_from("<4H", data, i + 4)
        units = struct.unpack_from("<%di" % n, data, i + 12)
        recs.append((p, layer, tok, units, miss))
        i += 12 + 4 * n
    return recs


def stream(recs):
    """(unit, one_token_pass, record index) for every access, in order."""
    for k, (p, layer, tok, units, miss) in enumerate(recs):
        for u in units:
            yield u, tok == 1, k


class LRU:
    def __init__(self, cap, admit_multi=True):
        self.cap, self.d, self.admit_multi = cap, collections.OrderedDict(), admit_multi

    def access(self, u, one):
        if u in self.d:
            if one or self.admit_multi:
                self.d.move_to_end(u)
            return True
        if one or self.admit_multi:
            self.d[u] = 1
            if len(self.d) > self.cap:
                self.d.popitem(last=False)
        return False


class SLRU:
    def __init__(self, cap, protected=0.8):
        self.pcap = int(cap * protected)
        self.bcap = cap - self.pcap
        self.prob, self.prot = collections.OrderedDict(), collections.OrderedDict()

    def access(self, u, one):
        if u in self.prot:
            self.prot.move_to_end(u)
            return True
        if u in self.prob:
            del self.prob[u]
            self.prot[u] = 1
            if len(self.prot) > self.pcap:
                v, _ = self.prot.popitem(last=False)
                self.prob[v] = 1
                if len(self.prob) > self.bcap:
                    self.prob.popitem(last=False)
            return True
        self.prob[u] = 1
        if len(self.prob) > self.bcap:
            self.prob.popitem(last=False)
        return False


class TwoQ:
    def __init__(self, cap):
        self.kin, self.kout = max(1, cap // 4), max(1, cap // 2)
        self.cap = cap
        self.a1in, self.a1out, self.am = collections.OrderedDict(), collections.OrderedDict(), collections.OrderedDict()

    def _evict(self):
        if len(self.a1in) + len(self.am) <= self.cap:
            return
        if len(self.a1in) > self.kin or not self.am:
            v, _ = self.a1in.popitem(last=False)
            self.a1out[v] = 1
            if len(self.a1out) > self.kout:
                self.a1out.popitem(last=False)
        else:
            self.am.popitem(last=False)

    def access(self, u, one):
        if u in self.am:
            self.am.move_to_end(u)
            return True
        if u in self.a1in:
            return True
        if u in self.a1out:
            del self.a1out[u]
            self.am[u] = 1
        else:
            self.a1in[u] = 1
        self._evict()
        return False


class LFUAged:
    def __init__(self, cap):
        self.cap, self.cnt, self.last, self.t, self.res = cap, collections.Counter(), {}, 0, set()

    def access(self, u, one):
        self.t += 1
        if self.t % self.cap == 0:
            for k in list(self.cnt):
                self.cnt[k] //= 2
        self.cnt[u] += 1
        self.last[u] = self.t
        if u in self.res:
            return True
        self.res.add(u)
        if len(self.res) > self.cap:
            v = min((x for x in self.res if x != u), key=lambda x: (self.cnt[x], self.last[x]))
            self.res.discard(v)
        return False


def opt(accesses, cap):
    """Belady: evict the resident unit whose next use is furthest away."""
    nxt = [0] * len(accesses)
    seen = {}
    for i in range(len(accesses) - 1, -1, -1):
        u = accesses[i][0]
        nxt[i] = seen.get(u, 1 << 60)
        seen[u] = i
    res, heap, hits = {}, [], []
    for i, (u, one, k) in enumerate(accesses):
        hit = u in res
        hits.append(hit)
        res[u] = nxt[i]
        heapq.heappush(heap, (-nxt[i], u))
        while len(res) > cap:
            negn, v = heapq.heappop(heap)
            if res.get(v) == -negn:
                del res[v]
    return hits


def main():
    d = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 1137
    ranks = sorted(f for f in os.listdir(d) if f.startswith("rank") and f.endswith(".bin"))
    print("capacity %d units a rank" % cap)
    for name in ranks:
        recs = load(os.path.join(d, name))
        acc = list(stream(recs))
        actual = sum(r[4] for r in recs) / 2
        actual_one = sum(r[4] for r in recs if r[2] == 1) / 2
        n_one = sum(1 for a in acc if a[1])
        print("%s: %d layer passes, %d accesses (%d in one-token passes); the engine read %d units "
              "from the drive (%d in one-token passes)" % (name, len(recs), len(acc), n_one, actual, actual_one))
        policies = [("lru", LRU(cap)), ("lru-noprefill", LRU(cap, admit_multi=False)),
                    ("slru", SLRU(cap)), ("2q", TwoQ(cap)), ("lfu-aged", LFUAged(cap))]
        for pname, pol in policies:
            miss = miss_one = 0
            for u, one, k in acc:
                if not pol.access(u, one):
                    miss += 1
                    miss_one += one
            print("  %-14s %6d misses (%5.1f%%), %6d in one-token passes (%5.1f%%)" % (
                pname, miss, 100.0 * miss / max(1, len(acc)), miss_one, 100.0 * miss_one / max(1, n_one)))
        hits = opt(acc, cap)
        miss = sum(1 for h in hits if not h)
        miss_one = sum(1 for h, a in zip(hits, acc) if not h and a[1])
        print("  %-14s %6d misses (%5.1f%%), %6d in one-token passes (%5.1f%%)" % (
            "opt", miss, 100.0 * miss / max(1, len(acc)), miss_one, 100.0 * miss_one / max(1, n_one)))


if __name__ == "__main__":
    main()
