import json, numpy as np
from collections import Counter
R=json.load(open('out/breaks.json'))
def q(v): v=np.array(v,float); return f"median {np.median(v):.2f}, p25 {np.percentile(v,25):.2f}, p75 {np.percentile(v,75):.2f}"
for reason in ["beyond link reach","partner is chance","within rules (lost in scoring or matching)"]:
    s=[r for r in R if r['reason']==reason]
    print(reason,len(s))
    for k in ['gap_periods','blank_run_periods','removed_frac','residual_frac','lat_px','turn_deg']:
        print('  ',k,q([r[k] for r in s]))
    c=Counter()
    for r in s:
        for k,v in r['removed_by'].items(): c[k]+=v
    print('   removed by',dict(c))
s=[r for r in R if r['reason']=='beyond link reach']
print('beyond reach, blank run <1 period:',sum(r['blank_run_periods']<1 for r in s), ' 1-2:',sum(1<=r['blank_run_periods']<2 for r in s),' >=2:',sum(r['blank_run_periods']>=2 for r in s))
s=[r for r in R if r['reason']=='no continuation']
cat=Counter()
for r in s:
    if r['ahead_removed']>0.5: k='runs into removed ground: '+ {1:'text',2:'boundary',3:'river',4:'legend',5:'grid'}[int(max(r['ahead_by'],key=r['ahead_by'].get))]
    elif r['ahead_residual']>0.3: k='unlinked residual ink ahead'
    elif r['ahead_blank_run']>=3: k='blank paper ahead (>=3 periods)'
    else: k='mixed'
    cat[k]+=1
print('no continuation',len(s),dict(cat))
