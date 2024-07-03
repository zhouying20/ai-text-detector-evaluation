from attack.transformation.char.punctuation_append import PunctuationAppend


# transformation methods
trans_methods = {
    "PunctuationAppend_bracket": {
        "func": PunctuationAppend,
        "args": {
            "add_bracket": True,
        }
    },
    "PunctuationAppend_no_bracket": {
        "func": PunctuationAppend,
        "args": {
            "add_bracket": False,
        }
    },
}
