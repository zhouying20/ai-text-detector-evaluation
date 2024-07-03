from attack.transformation.char.word_case import WordCase

# transformation methods
trans_methods = {
    "WordCase_pct10": {
        "func": WordCase,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.1
        },
    },
    "WordCase_pct20": {
        "func": WordCase,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.2
        },
    },
    "WordCase_pct30": {
        "func": WordCase,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.3
        },
    },
    "WordCase_pct40": {
        "func": WordCase,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.4
        },
    },
    "WordCase_pct50": {
        "func": WordCase,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.5
        },
    },
}