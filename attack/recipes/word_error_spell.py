from attack.transformation.word.error_spell import SpellingError

# transformation methods
trans_methods = {
    "SpellingError_pct10": {
        "func": SpellingError,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.1
        },
    },
    "SpellingError_pct20": {
        "func": SpellingError,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.2,
        },
    },
    "SpellingError_pct30": {
        "func": SpellingError,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.3,
        },
    },
    "SpellingError_pct40": {
        "func": SpellingError,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.4,
        },
    },
}