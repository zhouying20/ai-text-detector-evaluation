import os
import json
import logging
import argparse

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from tqdm import tqdm
from utils.common import setup_logger
from utils.attack import load_textflint_dataset, do_transform, do_transform_batch

# override Textflint settings!!!
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = "~/.cache/huggingface/hub"

logger = logging.getLogger()
setup_logger(logger)


trans_fn = None


def init(trans_obj, trans_args, gpus, n_gpu=1):
    # init global variable for attacker
    global trans_fn

    gpu_list = list(map(str.strip, gpus.split(",")))
    num_gpu = len(gpu_list)

    # get unique id for each process with mp.current_process
    p_idx = int(mp.current_process()._identity[0]) # base started with 1
    gpu_i = ((p_idx - 1) * n_gpu) % num_gpu
    gpu_j = gpu_i + n_gpu
    use_gpus = ",".join(gpu_list[gpu_i: gpu_j])
    os.environ["CUDA_CUDA_VISIBLE_DEVICES"] = use_gpus
    device = f"cuda:{use_gpus}"

    trans_fn = trans_obj(**trans_args, device=device)

    logger.info(
        "*******************************\n"
        f"initializing process-{p_idx}...]\n"
        f"\t gpus: {os.environ.get('CUDA_CUDA_VISIBLE_DEVICES', None)}\n"
        f"\t transformation function: \n"
        f"{trans_fn.__repr__()}\n"
        "*******************************\n",
    )


def transform_helper_func(data):
    assert trans_fn is not None, "please init trans_fn firstly..."
    sample, data_point = data
    res_trans = do_transform(sample, data_point, trans_fn)
    return res_trans


class MultiProcessingHelper:
    def __init__(self):
        self.total = None

    def __call__(self, data_samples, output_file, func, workers=None, init_fn=None, init_args=None):
        self.total = len(data_samples)

        with mp.Pool(workers, initializer=init_fn, initargs=init_args) as pool, \
             tqdm(pool.imap(func, data_samples), total=self.total, dynamic_ncols=True) as pbar, \
             open(output_file, "w") as wf:
                for idx, trans_list in enumerate(pbar):
                    for one in trans_list:
                        if "sample_id" in one.keys():
                            assert one["sample_id"] == idx
                        else:
                            one["sample_id"] = idx
                        wf.write(json.dumps(one, ensure_ascii=False) + "\n")


def get_trans_methods(trans_method):
    if trans_method == "paragraph_paraphrase":
        from attack.recipes.paragraph_paraphrase import trans_methods
    elif trans_method == "paragraph_back_trans":
        from attack.recipes.paragraph_back_trans import trans_methods
    elif trans_method == "paragraph_reverse":
        from attack.recipes.paragraph_reverse import trans_methods
    elif trans_method == "sentence_back_trans":
        from attack.recipes.sentence_back_trans import trans_methods
    elif trans_method == "sentence_mlm":
        from attack.recipes.sentence_mlm import trans_methods
    elif trans_method == "sentence_append_irr":
        from attack.recipes.sentence_append_irr import trans_methods
    elif trans_method == "sentence_repeat":
        from attack.recipes.sentence_repeat import trans_methods
    elif trans_method == "word_error_spell":
        from attack.recipes.word_error_spell import trans_methods
    elif trans_method == "word_error_typo":
        from attack.recipes.word_error_typo import trans_methods
    elif trans_method == "word_insert_adv":
        from attack.recipes.word_insert_adv import trans_methods
    elif trans_method == "word_mlm":
        from attack.recipes.word_mlm import trans_methods
    elif trans_method == "char_remove_space":
        from attack.recipes.char_remove_space import trans_methods
    elif trans_method == "char_word_case":
        from attack.recipes.char_word_case import trans_methods
    elif trans_method == "char_insert_space":
        from attack.recipes.char_insert_space import trans_methods
    elif trans_method == "char_insert_punc":
        from attack.recipes.char_insert_punc import trans_methods
    elif trans_method == "char_remove_punc":
        from attack.recipes.char_remove_punc import trans_methods
    elif trans_method == "dev":
        from attack.recipes.dev_data import trans_methods
    else:
        raise NotImplementedError(f"not supported transformation -> {trans_method}")

    return trans_methods


def main():
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--trans_method", type=str, required=True)
    parser.add_argument("--test_file", default="data/CheckGPT/original/test.jsonl")
    parser.add_argument("--output_dir", default="output/checkgpt/transformations/")
    parser.add_argument("--prefix_key", type=str, default=None)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_gpu_per_process", type=int, default=1)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    args.num_gpus = len(args.gpus.split(","))
    assert args.num_gpu_per_process == 1
    assert args.num_workers == 1 or args.num_workers % args.num_gpus == 0

    dataset, test_samples = load_textflint_dataset(
        args.test_file,
        prefix_key=args.prefix_key, text_key=args.text_key,
    )
    trans_methods = get_trans_methods(args.trans_method)

    for trans_name, trans_json in trans_methods.items():
        output_file = os.path.join(args.output_dir, trans_name+".jsonl")
        if args.num_workers == 1:
            trans_cls = trans_json["func"](**trans_json["args"])
            do_transform_batch(dataset, test_samples, trans_cls, output_file)
        else:
            worker = MultiProcessingHelper()
            worker(
                list(zip(dataset, test_samples, strict=True)),
                output_file,
                transform_helper_func,
                workers=args.num_workers,
                init_fn=init,
                init_args=(trans_json["func"], trans_json["args"], args.gpus, args.num_gpu_per_process, )
            )


if __name__ == "__main__":
    main()
