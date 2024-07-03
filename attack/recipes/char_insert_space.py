from attack.transformation.char.insert_space import InsertSpace


# transformation methods
trans_methods = {
    # "InsertSpace_1_1_prob70": {
    #     "func": InsertSpace,
    #     "args": {
    #         "insert_min": 1,
    #         "insert_max": 1,
    #         "insert_prob": 0.7,
    #     }
    # },
    # "InsertSpace_2_2_prob70": {
    #     "func": InsertSpace,
    #     "args": {
    #         "insert_min": 2,
    #         "insert_max": 2,
    #         "insert_prob": 0.7,
    #     }
    # },
    # "InsertSpace_3_3_prob70": {
    #     "func": InsertSpace,
    #     "args": {
    #         "insert_min": 3,
    #         "insert_max": 3,
    #         "insert_prob": 0.7,
    #     }
    # },
    # "InsertSpace_3_3_prob100": {
    #     "func": InsertSpace,
    #     "args": {
    #         "insert_min": 3,
    #         "insert_max": 3,
    #         "insert_prob": 1.0,
    #     }
    # },
    # "InsertSpace_2_5_prob70": {
    #     "func": InsertSpace,
    #     "args": {
    #         "insert_min": 2,
    #         "insert_max": 5,
    #         "insert_prob": 0.7,
    #     }
    # },
    "InsertSpace_5_10_prob100": {
        "func": InsertSpace,
        "args": {
            "insert_min": 5,
            "insert_max": 10,
            "insert_prob": 1.0,
        }
    },
}
