"""Separate nonempty-family coverage from all-replicate false-acceptance events."""
import numpy as np
import pandas as pd
from study_common import O,LOADERS
metrics=['coverage','coverage_supported','accept','reject','abstain','unsupported','false_accept','false_reject','any_false_accept','top_coverage','top_bias','accepted_true_gain','rules']
empties=[]
for ds in LOADERS:
 frame=pd.read_csv(O/f'results/e2_{ds}_replicates.csv');total=frame.replicate.nunique();records=[]
 for flow in ['full','split']:
  empty=frame[(frame.flow==flow)&(frame.rules==0)].replicate.nunique();empties.append(dict(dataset=ds,flow=flow,replicates=total,empty_families=empty,nonempty_families=total-empty))
 for keys,g in frame[frame.rules>0].groupby(['dataset','flow','view','method']):
  for metric in metrics:
   x=pd.to_numeric(g[metric],errors='coerce').dropna().to_numpy(dtype=float)
   if metric in ['any_false_accept','rules']:x=np.concatenate([x,np.zeros(total-len(x))])
   records.append(dict(zip(['dataset','flow','view','method'],keys),metric=metric,mean=x.mean() if len(x) else np.nan,mcse=x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan,replicates=len(x)))
 pd.DataFrame(records).to_csv(O/f'results/e2_{ds}_summary.csv',index=False)
pd.DataFrame(empties).to_csv(O/'results/e2_family_denominators.csv',index=False)
