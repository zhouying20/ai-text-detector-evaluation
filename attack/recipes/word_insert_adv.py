from attack.transformation.word.insert_adv import InsertAdv

# transformation methods
trans_methods = {
    "InsertAdv_pct10": {
        "func": InsertAdv,
        "args": {
            "trans_p": 0.1
        },
    },
    "InsertAdv_pct20": {
        "func": InsertAdv,
        "args": {
            "trans_p": 0.2
        },
    },
    "InsertAdv_pct30": {
        "func": InsertAdv,
        "args": {
            "trans_p": 0.3
        },
    },
    "InsertAdv_pct40": {
        "func": InsertAdv,
        "args": {
            "trans_p": 0.4
        },
    },
    "InsertAdv_pct50": {
        "func": InsertAdv,
        "args": {
            "trans_p": 0.5
        },
    },
}