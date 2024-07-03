from attack.transformation.sentence.swap_generation import MLMSentence

# transformation methods
trans_methods = {
    "MLMSentence_trans_1_1": {
        "func": MLMSentence,
        "args": {
            "trans_min": 1,
            "trans_max": 1,
        }
    },
    "MLMSentence_trans_2_2": {
        "func": MLMSentence,
        "args": {
            "trans_min": 2,
            "trans_max": 2,
        }
    },
    "MLMSentence_trans_3_3": {
        "func": MLMSentence,
        "args": {
            "trans_min": 3,
            "trans_max": 3,
        }
    },
    "MLMSentence_trans_2_5": {
        "func": MLMSentence,
        "args": {
            "trans_min": 2,
            "trans_max": 5,
        }
    },
    "MLMSentence_large_trans_1_1": {
        "func": MLMSentence,
        "args": {
            "trans_min": 1,
            "trans_max": 1,
            "model_name_or_path": "facebook/bart-large",
        }
    },
    "MLMSentence_large_trans_2_2": {
        "func": MLMSentence,
        "args": {
            "trans_min": 2,
            "trans_max": 2,
            "model_name_or_path": "facebook/bart-large",
        }
    },
    "MLMSentence_large_trans_3_3": {
        "func": MLMSentence,
        "args": {
            "trans_min": 3,
            "trans_max": 3,
            "model_name_or_path": "facebook/bart-large",
        }
    },
    "MLMSentence_large_trans_2_5": {
        "func": MLMSentence,
        "args": {
            "trans_min": 2,
            "trans_max": 5,
            "model_name_or_path": "facebook/bart-large",
        }
    },
}