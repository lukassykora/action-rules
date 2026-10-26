"""E2: real-miner plasmode selection study, with checkpointed per-replicate results."""
from pathlib import Path
import argparse,time,warnings
import numpy as np
import pandas as pd
from action_rules.inference.counts import infer_counts
from study_common import O,LOADERS,Counter,mine,coefficients,plugin
parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=list(LOADERS));parser.add_argument('--replicates',type=int)
args=parser.parse_args();settings={'Attrition':1000,'Telco':400,'Bank':300,'Credit':300}
for ds in ([args.dataset] if args.dataset else ['Attrition','Telco','Bank','Credit']):
 c=LOADERS[ds]();population=Counter(c.df);total=args.replicates or settings[ds];rows=[];started=time.perf_counter()
 checkpoint=O/f'results/e2_{ds}_replicates.csv'
 done=0
 if checkpoint.exists():
  old=pd.read_csv(checkpoint);rows=old.to_dict('records');done=int(old.replicate.max())+1
 for rep in range(done,total):
  seed=771026+list(LOADERS).index(ds)*100000+rep;rng=np.random.default_rng(seed)
  sample=c.df.iloc[rng.integers(0,len(c.df),len(c.df))].reset_index(drop=True)
  permutation=rng.permutation(len(sample));split=len(sample)//2
  discovery=sample.iloc[permutation[:split]].reset_index(drop=True);estimation=sample.iloc[permutation[split:]].reset_index(drop=True)
  for flow,frame,scale in [('full',sample,1.),('split',discovery,len(discovery)/len(sample))]:
   model,rules=mine(c,frame,scale)
   if not rules:
    rows.append(dict(dataset=ds,replicate=rep,seed=seed,flow=flow,view='no_rules',method='none',rules=0));continue
   a,b=coefficients(c,model,rules);truth,_=plugin(population.cells(rules),a,b)
   training=Counter(frame).cells(rules);train_g,_=plugin(training,a,b);top=int(np.argmax(train_g));truth_loss=truth<=0
   for view,counts in ([('same',training)] if flow=='full' else [('same',training),('honest',Counter(estimation).cells(rules))]):
    point,_=plugin(counts,a,b)
    for m,method in enumerate(['point','wald','percentile','jeffreys','wilson','cp']):
     if method=='point':
      supported=np.isfinite(point);accepted=supported&(point>0);rejected=supported&(point<=0);covered=np.full(len(rules),False)
     else:
      with warnings.catch_warnings():
       warnings.simplefilter('ignore');r=infer_counts(counts,a,b,method=method,design='random',family_size=len(rules) if method=='cp' else 1,n_resamples=999,n_mc=4000,random_state=seed+10000*m)['G']
      supported=r['supported'];accepted=supported&(r['lower']>0);rejected=supported&(r['upper']<0)
      covered=supported&(r['lower']<=truth)&(r['upper']>=truth)
     count_loss=int(truth_loss.sum());count_good=len(rules)-count_loss
     rows.append(dict(dataset=ds,replicate=rep,seed=seed,flow=flow,view=view,method=method,rules=len(rules),loss_rules=count_loss,
       coverage=covered.mean() if method!='point' else np.nan,coverage_supported=covered.sum()/supported.sum() if supported.sum() and method!='point' else np.nan,
       accept=accepted.mean(),reject=rejected.mean(),abstain=(~accepted&~rejected).mean(),unsupported=(~supported).mean(),
       false_accept=((accepted&truth_loss).sum()/count_loss if count_loss else np.nan),false_reject=((rejected&~truth_loss).sum()/count_good if count_good else np.nan),
       any_false_accept=bool(np.any(accepted&truth_loss)),top_coverage=(bool(covered[top]) if method!='point' else np.nan),top_bias=point[top]-truth[top],
       accepted_true_gain=float(truth[accepted].mean()) if accepted.any() else np.nan))
  if (rep+1)%5==0 or rep==total-1:
   pd.DataFrame(rows).to_csv(checkpoint,index=False)
  if (rep+1)%25==0 or rep==total-1:print(ds,rep+1,'/',total,'elapsed',round(time.perf_counter()-started),'s',flush=True)
 frame=pd.DataFrame(rows);metrics=['coverage','coverage_supported','accept','reject','abstain','unsupported','false_accept','false_reject','any_false_accept','top_coverage','top_bias','accepted_true_gain','rules']
 records=[]
 for keys,g in frame[frame.rules>0].groupby(['dataset','flow','view','method']):
  for metric in metrics:
   x=pd.to_numeric(g[metric],errors='coerce').dropna().to_numpy();records.append(dict(zip(['dataset','flow','view','method'],keys),metric=metric,mean=x.mean() if len(x) else np.nan,mcse=x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan,replicates=len(x)))
 pd.DataFrame(records).to_csv(O/f'results/e2_{ds}_summary.csv',index=False)
