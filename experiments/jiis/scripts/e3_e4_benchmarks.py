"""E3 and E4: utility decisions, sensitivity, and 10x5-fold ranking."""
import json,itertools
import numpy as np
import pandas as pd
from action_rules.inference.counts import infer_counts
from action_rules.evaluation.cv import stratified_kfold_indices
from study_common import O,LOADERS,Counter,mine,coefficients,plugin,signature
allrows=[];sens=[];cvrows=[];stability=[];families=[];details=[]
methods=['wald','wald_rd','wilson','percentile','bca','bayes','jeffreys','cp']
for ds,loader in LOADERS.items():
 c=loader();model,rules=mine(c,c.df);cells=Counter(c.df).cells(rules);a,b=coefficients(c,model,rules)
 for i,r in enumerate(rules):details.append(dict(dataset=ds,rule=i+1,a=a[i],b=b[i],cells=json.dumps(cells[i].tolist()),mask_u=json.dumps(r.mask_undesired),mask_d=json.dumps(r.mask_desired)))
 for m in ['point']+methods:
  if m=='point':g,u=plugin(cells,a,b);results={k:dict(point=v,lower=v,upper=v,supported=np.isfinite(v)) for k,v in [('G',g),('U',u)]}
  else:results=infer_counts(cells,a,b,method=m,design='random',family_size=len(rules) if m=='cp' else 1,n_resamples=999,n_mc=4000,random_state=42)
  for target in ['G','U']:
   v=results[target];above=v['supported']&(v['lower']>0);below=v['supported']&(v['upper']<0);unsupported=~v['supported']
   for i in range(len(rules)):allrows.append(dict(dataset=ds,method=m,target=target,rule=i+1,point=v['point'][i],lower=v['lower'][i],upper=v['upper'][i],category='unsupported' if unsupported[i] else 'accept' if above[i] else 'reject' if below[i] else 'uncertain'))
 if ds=='Attrition':
  for kind,factor in [('baseline',1.),('cost',.5),('cost',1.5),('support',22/18),('support',14/18),('actionable',1.)]:
   # Alter a copied config rather than the canonical one.
   from copy import deepcopy
   cc=deepcopy(c)
   if kind=='cost':
    cc.intrinsic_utility_table={k:(v*factor if v<0 else v) for k,v in c.intrinsic_utility_table.items()};cc.transition_utility_table={k:(v*factor if v<0 else v) for k,v in c.transition_utility_table.items()}
   mm,rr=mine(cc,cc.df,scale=factor if kind=='support' else 1.,flexible=['OverTime','MonthlyIncome'] if kind=='actionable' else None)
   count=Counter(cc.df).cells(rr);aa,bb=coefficients(cc,mm,rr)
   for method in ['point','wilson','cp']:
    if method=='point':g,_=plugin(count,aa,bb);v=dict(point=g,lower=g,upper=g,supported=np.isfinite(g))
    else:v=infer_counts(count,aa,bb,method=method,family_size=len(rr) if method=='cp' else 1)['G']
    above=v['supported']&(v['lower']>0);below=v['supported']&(v['upper']<0)
    sens.append(dict(dataset=ds,type=kind,setting=factor,method=method,rules=len(rr),accept=above.sum(),reject=below.sum(),unsupported=(~v['supported']).sum(),uncertain=(v['supported']&~above&~below).sum()))
  for B in [500,2000,10000]:
   for seed in [42,43,44]:
    v=infer_counts(cells,a,b,method='percentile',n_resamples=B,random_state=seed)['G'];i=3
    sens.append(dict(dataset=ds,type='bootstrap',setting=f'{B}:{seed}',method='percentile',rules=len(rules),accept=int((v['lower']>0).sum()),reject=int((v['upper']<0).sum()),unsupported=int((~v['supported']).sum()),uncertain=int((v['supported']&(v['lower']<=0)&(v['upper']>=0)).sum()),R4_lower=v['lower'][i]))
 for repeat in range(10):
  sigs=[]
  for fold,(train,test) in enumerate(stratified_kfold_indices(c.df[c.target],5,42+repeat)):
   tr,te=c.df.iloc[train],c.df.iloc[test];mm,rr=mine(c,tr,len(tr)/len(c.df));sigs.append(set(signature(r) for r in rr))
   if not rr:families.append(dict(dataset=ds,repeat=repeat,fold=fold,rules=0,unsupported_test=0));continue
   aa,bb=coefficients(c,mm,rr);traincells,testcells=Counter(tr).cells(rr),Counter(te).cells(rr);inference=infer_counts(traincells,aa,bb,method='wilson',design='fixed');gtest,utest=plugin(testcells,aa,bb);testnu=testcells[:,:2].sum(axis=1)
   families.append(dict(dataset=ds,repeat=repeat,fold=fold,rules=len(rr),unsupported_test=int((~np.isfinite(gtest)).sum())))
   scores={'point_U':inference['U']['point'],'lower_U':inference['U']['lower'],'point_G':inference['G']['point'],'lower_G':inference['G']['lower'],'filter_G':np.where(inference['G']['lower']>0,inference['G']['lower'],-np.inf)}
   for strategy,score in scores.items():
    eligible=np.flatnonzero(np.isfinite(score));order=eligible[np.argsort(-score[eligible],kind='stable')];selected=order[:int(np.ceil(.2*len(order)))];valid=selected[np.isfinite(gtest[selected])]
    cvrows.append(dict(dataset=ds,repeat=repeat,fold=fold,strategy=strategy,rules=len(rr),eligible=len(eligible),selected=len(selected),unsupported_selected=len(selected)-len(valid),
       mean_gain=float(np.mean(gtest[valid])) if len(valid) else np.nan,
       aggregate_gain=float(np.sum(testnu[valid]*gtest[valid])) if len(valid) else 0. if not len(selected) else np.nan))
  for i,j in itertools.combinations(range(5),2):
   union=sigs[i]|sigs[j];stability.append(dict(dataset=ds,repeat=repeat,fold1=i,fold2=j,jaccard=len(sigs[i]&sigs[j])/len(union) if union else np.nan))
  print('E4',ds,repeat+1,'/10',flush=True)
 if ds in ['Bank','Credit']:
  flexible=['contact','campaign'] if ds=='Bank' else ['LIMIT_BAL','PAY_AMT1'];_,rr=mine(c,c.df,flexible=flexible)
  sens.append(dict(dataset=ds,type='actionable',setting=','.join(flexible),method='none',rules=len(rr)))
 pd.DataFrame(allrows).to_csv(O/'results/e3_intervals.csv',index=False);pd.DataFrame(sens).to_csv(O/'results/e3_sensitivity.csv',index=False);pd.DataFrame(cvrows).to_csv(O/'results/e4_fold_results.csv',index=False);pd.DataFrame(stability).to_csv(O/'results/e4_stability.csv',index=False);pd.DataFrame(families).to_csv(O/'results/e4_families.csv',index=False);pd.DataFrame(details).to_csv(O/'results/e3_rule_details.csv',index=False)
print('E3/E4 completed.',flush=True)
