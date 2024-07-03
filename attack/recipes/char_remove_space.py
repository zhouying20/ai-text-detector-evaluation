from attack.transformation.char.space_removal import SpeaceRemoval

# transformation methods
trans_methods = {
    "SpeaceRemoval_1_1": {
        "func": SpeaceRemoval,
        "args": {
            "removal_min": 1,
            "removal_max": 1,
            "removal_prob": 1.0,
        },
    },
    "SpeaceRemoval_3_3": {
        "func": SpeaceRemoval,
        "args": {
            "removal_min": 3,
            "removal_max": 3,
            "removal_prob": 1.0,
        },
    },
    "SpeaceRemoval_5_5": {
        "func": SpeaceRemoval,
        "args": {
            "removal_min": 5,
            "removal_max": 5,
            "removal_prob": 1.0,
        },
    },
    "SpeaceRemoval_3_10": {
        "func": SpeaceRemoval,
        "args": {
            "removal_min": 3,
            "removal_max": 10,
            "removal_prob": 1.0,
        },
    },
}