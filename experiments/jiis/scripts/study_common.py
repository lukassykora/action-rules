"""Canonical data access and counted-rule helpers for E2--E5."""
from pathlib import Path
import json
import numpy as np
from action_rules import ActionRules
from action_rules.inference.base import compute_realistic_gain,extract_rule_masks
import _datasets as datasets
O=Path(__file__).resolve().parents[1]
if (O/'data').exists():
    datasets._ci_data_dir=lambda:O/'data'
    datasets._legacy_data_dir=lambda:O/'data'
LOADERS={'Telco':datasets.load_telco,'Bank':datasets.load_bank_marketing,'Attrition':datasets.load_employee_attrition,'Credit':datasets.load_credit_card_default}


def mine(config,frame,scale=1.,flexible=None):
    """Mine with thresholds scaled once, using the upstream rounding convention."""
    model=ActionRules(config.min_stable_attributes,config.min_flexible_attributes,max(1,round(config.min_undesired_support*scale)),config.min_undesired_confidence,max(1,round(config.min_desired_support*scale)),config.min_desired_confidence,intrinsic_utility_table=config.intrinsic_utility_table,transition_utility_table=config.transition_utility_table)
    model.fit(frame,config.stable_attributes,config.flexible_attributes if flexible is None else flexible,config.target,config.target_undesired_state,config.target_desired_state)
    return model,extract_rule_masks(model.output)


class Counter:
    """Reuse string arrays and antecedent masks within one immutable dataset."""
    def __init__(self,frame):
        self.frame=frame;self.columns={k:frame[k].astype(str).to_numpy() for k in frame};self.masks={}
    def mask(self,conditions):
        key=tuple(sorted((k,str(v)) for k,v in conditions.items()))
        if key not in self.masks:
            out=np.ones(len(self.frame),dtype=bool)
            for k,v in key:out &= self.columns[k]==v
            self.masks[key]=out
        return self.masks[key]
    def cells(self,rules):
        rows=[]
        for r in rules:
            u,d=self.mask(r.mask_undesired),self.mask(r.mask_desired)
            if np.any(u&d):raise ValueError('Inference requires genuine disjoint rules')
            y=self.columns[r.target_attribute];nu,nd=int(u.sum()),int(d.sum())
            xu,xd=int((u&(y==str(r.target_undesired))).sum()),int((d&(y==str(r.target_desired))).sum())
            rows.append([xu,nu-xu,xd,nd-xd,len(y)-nu-nd])
        return np.asarray(rows,dtype=int).reshape(-1,5)


def coefficients(config,model,rules):
    """Resolve the package's two utility tables into a and b."""
    args=config.intrinsic_utility_table,config.transition_utility_table,model.output.column_values
    a=np.array([compute_realistic_gain(r,0.,1.,*args) for r in rules])
    b=np.array([compute_realistic_gain(r,1.,1.,*args) for r in rules])-a
    return a,b


def plugin(c,a,b):
    """Return G and U, preserving missing antecedents as unavailable."""
    nu,nd=c[:,:2].sum(axis=1),c[:,2:4].sum(axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        d=c[:,0]/nu+c[:,2]/nd-1
        return a+b*d,nu/c.sum(axis=1)*d


def signature(rule):
    """Identify a rule across independent mining runs."""
    return tuple(sorted(rule.mask_undesired.items())),tuple(sorted(rule.mask_desired.items()))
