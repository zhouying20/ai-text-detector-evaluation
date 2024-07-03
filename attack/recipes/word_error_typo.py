from attack.transformation.word.error_typo import Typos

# transformation methods
trans_methods = {
    "Typos_pct10": {
        "func": Typos,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.1
        },
    },
    "Typos_pct20": {
        "func": Typos,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.2
        },
    },
    "Typos_pct30": {
        "func": Typos,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.3
        },
    },
    "Typos_pct40": {
        "func": Typos,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.4
        },
    },
    "Typos_replace_pct20": {
        "func": Typos,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.2,
            "mode": "replace"
        },
    },
    "Typos_swap_pct20": {
        "func": Typos,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.2,
            "mode": "swap"
        },
    },
    "Typos_insert_pct20": {
        "func": Typos,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.2,
            "mode": "insert"
        },
    },
    "Typos_delete_pct20": {
        "func": Typos,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.2,
            "mode": "delete"
        },
    },
}