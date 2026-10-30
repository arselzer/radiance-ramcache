import struct, os, sys, collections
d = sys.argv[1]
def load(path):
    data = open(path, 'rb').read(); i = 0; out = collections.OrderedDict()
    while i + 12 <= len(data):
        p, = struct.unpack_from('<I', data, i); layer, tok, n, miss = struct.unpack_from('<4H', data, i + 4)
        units = struct.unpack_from('<%di' % n, data, i + 12); i += 12 + 4 * n
        out.setdefault(p, {'tok': tok, 'layers': {}})['layers'][layer] = frozenset(units)
    return out
ranks = [load(os.path.join(d, 'rank%d.bin' % r)) for r in range(4)]
runs, cur = [], []
for p in ranks[0]:
    if any(p not in rk for rk in ranks): continue
    if ranks[0][p]['tok'] != 1:
        if cur: runs.append(cur); cur = []
        continue
    cur.append(p)
if cur: runs.append(cur)
layers = sorted({l for p in ranks[0] for l in ranks[0][p]['layers']})
def job_cost(rows):   # HostExpertPool::job_cost in expert units, jobs of at most 4 rows
    c = 0.0
    while rows > 0:
        r = min(rows, 4); c += 1 + 0.5 * (r - 1); rows -= r
    return c
res = {}
for k in (1, 2, 3, 4, 5, 6):
    distinct = cost = pairs = 0; n = 0
    for run in runs:
        for s in range(0, len(run) - k + 1, k):
            win = run[s:s + k]
            for l in layers:
                for rk in ranks:
                    cnt = collections.Counter()
                    for p in win: cnt.update(rk[p]['layers'].get(l, ()))
                    distinct += len(cnt); pairs += sum(cnt.values()); cost += sum(job_cost(v) for v in cnt.values())
                n += 1
    res[k] = (distinct / n, cost / n, pairs / n)
b = res[1]
print("per layer, summed over the 4 ranks; a k-token verify pass against k one-token steps")
for k, (dd, cc, pp) in res.items():
    print("k=%d  distinct %.2f (%.2f of k steps)  pool cost %.2f (%.2f of k steps)  expert-rows %.2f" % (
        k, dd, dd / (k * b[0]), cc, cc / (k * b[1]), pp))
