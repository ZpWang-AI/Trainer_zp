# follow <<Connective Prediction for Implicit Discourse Relation Recognition via Knowledge Distillation>>
# change the label list of pdtb3, level2
# in order to suit the ans word map
LEVEL1_LABEL_LIST = [
    "Comparison",
    "Contingency",
    "Expansion",
    "Temporal"
]
LEVEL2_LABEL_LIST = {
    "pdtb2": [
        "Comparison.Concession",
        "Comparison.Contrast",
        
        "Contingency.Cause",
        "Contingency.Pragmatic cause",
        
        "Expansion.Alternative",
        "Expansion.Conjunction",
        "Expansion.Instantiation",
        "Expansion.List",
        "Expansion.Restatement",
        
        "Temporal.Asynchronous",
        "Temporal.Synchrony"
    ],
    "pdtb3": [
        "Comparison.Concession",
        "Comparison.Contrast",
        "Comparison.Similarity"  # New added
        
        "Contingency.Cause",
        # "Contingency.Cause+Belief",
        "Contingency.Condition",
        "Contingency.Purpose",
        
        "Expansion.Conjunction",
        "Expansion.Equivalence",
        "Expansion.Instantiation",
        "Expansion.Level-of-detail",
        "Expansion.Manner",
        "Expansion.Substitution",
        
        "Temporal.Asynchronous",
        "Temporal.Synchronous"
    ],
    "conll": [
        "Comparison.Concession",
        "Comparison.Contrast",
        
        "Contingency.Cause.Reason",
        "Contingency.Cause.Result",
        "Contingency.Condition",
        
        "Expansion.Alternative",
        "Expansion.Alternative.Chosen alternative",
        "Expansion.Conjunction",
        "Expansion.Exception",
        "Expansion.Instantiation",
        "Expansion.Restatement",
        
        "Temporal.Asynchronous.Precedence",
        "Temporal.Asynchronous.Succession",
        "Temporal.Synchrony"
    ]
}


if __name__ == '__main__':
    for v in LEVEL2_LABEL_LIST.values():
        if v != sorted(v):
            print(v)
            break
    else:
        print('all sorted')