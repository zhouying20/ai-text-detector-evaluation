from attack.transformation.paragraph.back_trans import BackTrans


# transformation methods
trans_methods = {
    "BackTrans_r1": {
        "func": BackTrans,
        "args": {
            "trans_iter": 1
        }
    },
    # "BackTrans_r3": {
    #     "func": BackTrans,
    #     "args": {
    #         "trans_iter": 3
    #     }
    # },
    # "BackTrans_r5": {
    #     "func": BackTrans,
    #     "args": {
    #         "trans_iter": 5
    #     }
    # },
    "BackTrans_Helsinki_r1": {
        "func": BackTrans,
        "args": {
            "trans_iter": 1,
            "from_model_name": "Helsinki-NLP/opus-mt-en-fr",
            "to_model_name": "Helsinki-NLP/opus-mt-fr-en",
        }
    },
    "BackTrans_Helsinki_r3": {
        "func": BackTrans,
        "args": {
            "trans_iter": 3,
            "from_model_name": "Helsinki-NLP/opus-mt-en-fr",
            "to_model_name": "Helsinki-NLP/opus-mt-fr-en",
        }
    },
    "BackTrans_Helsinki_r5": {
        "func": BackTrans,
        "args": {
            "trans_iter": 5,
            "from_model_name": "Helsinki-NLP/opus-mt-en-fr",
            "to_model_name": "Helsinki-NLP/opus-mt-fr-en",
        }
    },
}
