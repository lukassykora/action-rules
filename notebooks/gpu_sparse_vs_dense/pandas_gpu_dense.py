import pandas as pd
from action_rules import ActionRules

stable_attributes = ["gender", "SeniorCitizen", "Partner"]
flexible_attributes = ["PhoneService",
                       "InternetService",
                       "OnlineSecurity",
                       "DeviceProtection",
                       "TechSupport",
                       "StreamingTV"]
target = 'Churn'
min_stable_attributes = 2
min_flexible_attributes = 1  # min 1
min_undesired_support = 80
min_undesired_confidence = 0.6
min_desired_support = 80
min_desired_confidence = 0.6
undesired_state = 'Yes'
desired_state = 'No'

pd.set_option('display.max_columns', None)
data_frame = pd.read_csv("./../data/telco.csv", sep=";")
data_frame = pd.concat([data_frame] * 20)

# Action Rules Mining
action_rules = ActionRules(
    min_stable_attributes=min_stable_attributes,
    min_flexible_attributes=min_flexible_attributes,
    min_undesired_support=min_undesired_support,
    min_undesired_confidence=min_undesired_confidence,
    min_desired_support=min_desired_support,
    min_desired_confidence=min_desired_confidence,
    verbose=False
)

action_rules.fit(
    data=data_frame,
    stable_attributes=stable_attributes,
    flexible_attributes=flexible_attributes,
    target=target,
    target_undesired_state=undesired_state,
    target_desired_state=desired_state,
    use_gpu=True,
    use_sparse_matrix=False,
)
print('Number of action rules: ' + str(len(action_rules.get_rules().action_rules)))
