from attack.transformation.word.swap_mlm import MLMSuggestion

# transformation methods
trans_methods = {
    "MLMSuggestion_bert_pct10": {
        "func": MLMSuggestion,
        "args": {
            "masked_model": "bert-base-uncased",
            "trans_min": 10,
            "trans_max": None,
            "max_sent_size": 512,
            "trans_p": 0.1
        },
    },
    "MLMSuggestion_bert_pct20": {
        "func": MLMSuggestion,
        "args": {
            "masked_model": "bert-base-uncased",
            "trans_min": 10,
            "trans_max": None,
            "max_sent_size": 512,
            "trans_p": 0.2
        },
    },
    "MLMSuggestion_bert_pct30": {
        "func": MLMSuggestion,
        "args": {
            "masked_model": "bert-base-uncased",
            "trans_min": 10,
            "trans_max": None,
            "max_sent_size": 512,
            "trans_p": 0.3
        },
    },
    "MLMSuggestion_bert_pct40": {
        "func": MLMSuggestion,
        "args": {
            "masked_model": "bert-base-uncased",
            "trans_min": 10,
            "trans_max": None,
            "max_sent_size": 512,
            "trans_p": 0.4
        },
    },
    "MLMSuggestion_bert_pct50": {
        "func": MLMSuggestion,
        "args": {
            "masked_model": "bert-base-uncased",
            "trans_min": 10,
            "trans_max": None,
            "max_sent_size": 512,
            "trans_p": 0.5
        },
    },
    "MLMSuggestion_bert_pct60": {
        "func": MLMSuggestion,
        "args": {
            "masked_model": "bert-base-uncased",
            "trans_min": 10,
            "trans_max": None,
            "max_sent_size": 512,
            "trans_p": 0.6
        },
    },
    "MLMSuggestion_bert_pct70": {
        "func": MLMSuggestion,
        "args": {
            "masked_model": "bert-base-uncased",
            "trans_min": 10,
            "trans_max": None,
            "max_sent_size": 512,
            "trans_p": 0.7
        },
    },
}