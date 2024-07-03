from attack.transformation.char.punctuation_removal import PunctuationRemoval


# transformation methods
trans_methods = {
    "PunctuationRemoval_last": {
        "func": PunctuationRemoval,
        "args": {
            "only_last": True,
        }
    },
    "PunctuationRemoval_p10": {
        "func": PunctuationRemoval,
        "args": {
            "removal_min": 1,
            "removal_max": 10,
            "removal_p": 0.1,
        }
    },
    "PunctuationRemoval_p20": {
        "func": PunctuationRemoval,
        "args": {
            "removal_min": 1,
            "removal_max": 10,
            "removal_p": 0.2,
        }
    },
    "PunctuationRemoval_p30": {
        "func": PunctuationRemoval,
        "args": {
            "removal_min": 1,
            "removal_max": 10,
            "removal_p": 0.3,
        }
    },
}
