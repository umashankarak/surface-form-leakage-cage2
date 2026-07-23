import json, statistics, math
from scipy import stats
rows=[json.loads(l) for l in open("runs/gate2.jsonl")]
cells={}
for r in rows:
    cells.setdefault((r["model"],r["level"]),[]).append(r["total_reward"])
models=sorted({m for m,_ in cells})
print(f"{'model':16s} {'L0':>8s} {'L1':>8s} {'drop':>8s} {'95%CI':>9s} {'p':>7s}  sig?")
for m in models:
    l0=cells.get((m,"L0"),[]); l1=cells.get((m,"L1"),[])
    if len(l0)<2 or len(l1)<2: continue
    m0,m1=statistics.mean(l0),statistics.mean(l1)
    s0,s1=statistics.stdev(l0),statistics.stdev(l1)
    drop=m1-m0; se=math.sqrt(s0**2/len(l0)+s1**2/len(l1))
    if se==0: continue
    df=(s0**2/len(l0)+s1**2/len(l1))**2/((s0**2/len(l0))**2/(len(l0)-1)+(s1**2/len(l1))**2/(len(l1)-1))
    ci=stats.t.ppf(0.975,df)*se; p=2*(1-stats.t.cdf(abs(drop)/se,df))
    sig="YES" if abs(drop)>ci else "no"
    print(f"{m:16s} {m0:8.1f} {m1:8.1f} {drop:+8.1f} {'+/-'+format(ci,'.1f'):>9s} {p:7.3f}  {sig}")
