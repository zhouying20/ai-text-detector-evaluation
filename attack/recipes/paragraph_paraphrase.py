from attack.transformation.paragraph.paraphrase import Paraphrase


# transformation methods
trans_methods = {
    # "Paraphrase_l40_o40": {
    #     "func": Paraphrase,
    #     "args": {
    #         "lex_diversity": 40,
    #         "order_diversity": 40,
    #     }
    # },
    "Paraphrase_l40_o40_no_prefix": {
        "func": Paraphrase,
        "args": {
            "lex_diversity": 40,
            "order_diversity": 40,
            "use_prefix": False,
        }
    },
    # "Paraphrase_l40_o100": {
    #     "func": Paraphrase,
    #     "args": {
    #         "lex_diversity": 40,
    #         "order_diversity": 100,
    #     }
    # },
    # "Paraphrase_l60_o100": {
    #     "func": Paraphrase,
    #     "args": {
    #         "lex_diversity": 60,
    #         "order_diversity": 100,
    #     }
    # },
    # "Paraphrase_l80_o100": {
    #     "func": Paraphrase,
    #     "args": {
    #         "lex_diversity": 80,
    #         "order_diversity": 100,
    #     }
    # },
    # "Paraphrase_r3": {
    #     "func": Paraphrase,
    #     "args": {
    #         "trans_iter": 3
    #     }
    # },
}
