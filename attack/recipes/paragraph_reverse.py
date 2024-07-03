from attack.transformation.paragraph.reverse_by_pivot import ReverseByPivot


# transformation methods
trans_methods = {
    "ReverseByPivot_t3": {
        "func": ReverseByPivot,
        "args": {
            "trans_iter": 3
        }
    },
    "ReverseByPivot_t1": {
        "func": ReverseByPivot,
        "args": {
            "trans_iter": 1
        }
    }
}
