"""E5: serial warmed timings with matched bootstrap replicate counts."""
import time,json,sys,platform,subprocess
import numpy as np
import pandas as pd
import scipy
from action_rules.inference.counts import infer_counts
from study_common import O,LOADERS,Counter,mine,coefficients
rows=[]
for ds,loader in LOADERS.items():
 c=loader();model,rules=mine(c,c.df);a,b=coefficients(c,model,rules)
 methods=['counting','wald','wilson','percentile','bca','jeffreys','cp','row_api','nw_api']
 for method in methods:
  def run():
   if method=='row_api':return model.confidence_intervals(c.df,method='bootstrap',n_bootstrap=999,random_state=42)
   if method=='nw_api':return model.confidence_intervals(c.df,method='newcombe_wilson')
   count=Counter(c.df).cells(rules)
   if method=='counting':return count
   return infer_counts(count,a,b,method=method,design='random',family_size=len(rules) if method=='cp' else 1,n_resamples=999,n_mc=10000,random_state=42)
  run()
  for repeat in range(7):
   start=time.perf_counter();run();rows.append(dict(dataset=ds,method=method,repeat=repeat,seconds=time.perf_counter()-start))
 pd.DataFrame(rows).to_csv(O/'results/e5_runtime.csv',index=False)
 print('E5 completed:',ds,flush=True)
cpu = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() if sys.platform=='darwin' else platform.processor()
(O/'results/environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,platform=platform.platform(),cpu=cpu,bootstrap_replicates=999,posterior_draws_runtime=10000,repetitions=7,execution='serial after E1-E4'),indent=2)+'\n')
