from attack.transformation.sentence.back_trans import BackTransSentence

# transformation methods
trans_methods = {
    "BackTransSentence_s1_w1": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 1,
            "trans_sent_window": 1,
        }
    },
    "BackTransSentence_s3_w1": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 3,
            "trans_sent_window": 1,
        }
    },
    "BackTransSentence_s1_w3": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 1,
            "trans_sent_window": 3,
        }
    },
    "BackTransSentence_s3_w3": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 3,
            "trans_sent_window": 3,
        }
    },
    "BackTransSentence_s3_w5": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 3,
            "trans_sent_window": 5,
        }
    },
    "BackTransSentence_Helsinki_s1_w1": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 1,
            "trans_sent_window": 1,
            "from_model_name": "Helsinki-NLP/opus-mt-en-fr",
            "to_model_name": "Helsinki-NLP/opus-mt-fr-en",
        }
    },
    "BackTransSentence_Helsinki_s3_w1": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 3,
            "trans_sent_window": 1,
            "from_model_name": "Helsinki-NLP/opus-mt-en-fr",
            "to_model_name": "Helsinki-NLP/opus-mt-fr-en",
        }
    },
    "BackTransSentence_Helsinki_s1_w3": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 1,
            "trans_sent_window": 3,
            "from_model_name": "Helsinki-NLP/opus-mt-en-fr",
            "to_model_name": "Helsinki-NLP/opus-mt-fr-en",
        }
    },
    "BackTransSentence_Helsinki_s3_w3": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 3,
            "trans_sent_window": 3,
            "from_model_name": "Helsinki-NLP/opus-mt-en-fr",
            "to_model_name": "Helsinki-NLP/opus-mt-fr-en",
        }
    },
    "BackTransSentence_Helsinki_s3_w5": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 3,
            "trans_sent_window": 5,
            "from_model_name": "Helsinki-NLP/opus-mt-en-fr",
            "to_model_name": "Helsinki-NLP/opus-mt-fr-en",
        }
    },
}