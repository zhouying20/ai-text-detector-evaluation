from attack.transformation.sentence.repeat import RepeatSentence

# transformation methods
trans_methods = {
    "RepeatSentence_1_1": {
        "func": RepeatSentence,
        "args": {
            "min_repeat": 1,
            "max_repeat": 1,
        }
    },
    "RepeatSentence_2_2": {
        "func": RepeatSentence,
        "args": {
            "min_repeat": 2,
            "max_repeat": 2,
        }
    },
    "RepeatSentence_3_3": {
        "func": RepeatSentence,
        "args": {
            "min_repeat": 3,
            "max_repeat": 3,
        }
    },
    "RepeatSentence_1_3": {
        "func": RepeatSentence,
        "args": {
            "min_repeat": 1,
            "max_repeat": 3,
        }
    },
}