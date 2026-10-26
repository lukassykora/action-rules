"""E1: full 24-regime calibration of the intended Final manuscript."""
from pathlib import Path
import json,time,warnings
import numpy as np
import pandas as pd
from action_rules.inference.counts import infer_counts
O=Path(__file__).resolve().parents[1];R=2000;B=999;M=4000
methods=['wald','wilson','percentile','bca','bayes','jeffreys','wald_rd','bayes_rd','cp']
regimes=[];summary=[];path=O/'results/e1_records.csv.gz'
if path.exists():path.unlink()
for design in ['fixed','random']:
 for support,N,q,qd in [('rich',2000,.10,.15),('moderate',500,.08,.12),('sparse',500,.03,.05)]:
  for effect,pu,pd_ in [('null',.5,.5),('modest',.6,.6),('strong',.8,.85),('boundary',.97,.98)]:
   j=len(regimes);seed=261100+j;rng=np.random.default_rng(seed)
   if design=='random':counts=rng.multinomial(N,[q*pu,q*(1-pu),qd*pd_,qd*(1-pd_),1-q-qd],size=R)
   else:
    nu,nd=round(N*q),round(N*qd);xu=rng.binomial(nu,pu,R);xd=rng.binomial(nd,pd_,R)
    counts=np.column_stack([xu,nu-xu,xd,nd-xd,np.full(R,N-nu-nd)])
   delta=pu+pd_-1;truth={'G':-30+100*delta,'U':q*delta,'H':q*(-30+100*delta)}
   spec=dict(index=j,design=design,support=support,effect=effect,N=N,q=q,qd=qd,pu=pu,pd=pd_,seed=seed)
   regimes.append(spec)
   for k,method in enumerate(methods):
    infer_seed=361100+j*100+(2 if method in ['percentile','bca'] else k)
    with warnings.catch_warnings():
     warnings.simplefilter('ignore');r=infer_counts(counts,a=-30,b=100,method=method,design=design,n_resamples=B,n_mc=M,random_state=infer_seed)
    blocks=[]
    for target,v in r.items():
     valid=v['supported'];covered=valid&(v['lower']<=truth[target])&(v['upper']>=truth[target]);nvalid=int(valid.sum())
     covered_valid=covered.sum()/nvalid if nvalid else np.nan;all_cov=covered.mean()
     summary.append(dict(spec,method=method,target=target,replicates=R,supported=nvalid,unsupported=R-nvalid,coverage=all_cov,coverage_supported=covered_valid,mcse=np.sqrt(all_cov*(1-all_cov)/R),median_width=np.nanmedian(v['upper']-v['lower'])))
     blocks.append(pd.DataFrame(dict(regime=j,design=design,support=support,effect=effect,method=method,target=target,replicate=np.arange(R),truth=truth[target],point=v['point'],lower=v['lower'],upper=v['upper'],supported=valid,covered=covered)))
    pd.concat(blocks).to_csv(path,index=False,mode='a',header=not path.exists(),compression='gzip')
   pd.DataFrame(summary).to_csv(O/'results/e1_summary.csv',index=False)
   print('E1',j+1,'/24:',design,support,effect,flush=True)
(O/'results/e1_settings.json').write_text(json.dumps(dict(R=R,B=B,M=M,regimes=regimes,methods=methods,coefficient_a=-30,coefficient_b=100,coverage_denominators='all replicates and supported replicates both recorded'),indent=2)+'\n')
