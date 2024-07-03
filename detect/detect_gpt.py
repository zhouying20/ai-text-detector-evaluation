import os
import json
import torch
import logging
import argparse
import numpy as np
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from tqdm import tqdm
from pathlib import Path
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from utils.common import setup_logger
from utils.detect import label_mapping, load_data, calc_metrics
from detect.lib_detect_gpt.run import get_perturbation_results


logger = logging.getLogger()
setup_logger(logger)

device = None
base_model = None
base_tokenizer = None
mask_model = None
mask_tokenizer = None

def init(base_model_name, mask_model_name, gpus, n_gpu=1):
    # 在这里初始化 Attacker 等需要全局使用的变量，通过 current_process 可以获取当前进程的唯一ID，从而设置环境变量
    global device, base_model, base_tokenizer, mask_model, mask_tokenizer

    gpu_list = list(map(str.strip, gpus.split(",")))
    num_gpu = len(gpu_list)

    p_idx = int(mp.current_process()._identity[0]) # base started with 1
    gpu_i = ((p_idx - 1) * n_gpu) % num_gpu
    os.environ["CUDA_CUDA_VISIBLE_DEVICES"] = str(gpu_i)
    device = torch.device(f"cuda:{gpu_i}")

    base_model, base_tokenizer, mask_model, mask_tokenizer = load_detect_gpt_model(
        base_model_name, mask_model_name, device=device
    )

    logger.info(
        "*******************************\n"
        f"initializing process-{p_idx}...]\n"
        f"\t gpus: {os.environ.get('CUDA_CUDA_VISIBLE_DEVICES', None)}\n"
        f"\t cuda_device: {device}\n"
        f"\t base_model: {base_model_name}\n"
        f"\t mask_model: {mask_model_name}\n"
        "*******************************\n",
    )


def load_detect_gpt_model(base_model_name, mask_model_name, device="cuda"):
    if base_model_name is not None and base_model_name not in ["none", "None", "text-davinci-003"]:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
        model.eval()
    elif base_model_name == "text-davinci-003":
        model = "text-davinci-003"
        tokenizer = None
    else:
        raise NotImplementedError()

    logger.info(f"Loading mask model of type {mask_model_name}...")
    if mask_model_name is not None and mask_model_name not in ["none", "None"]:
        mask_tokenizer = AutoTokenizer.from_pretrained(mask_model_name)
        mask_model = AutoModelForSeq2SeqLM.from_pretrained(mask_model_name).to(device)
        mask_model.eval()
    else:
        raise NotImplementedError()

    return model, tokenizer, mask_model, mask_tokenizer


def detectgpt_detect(generation, mask_model, mask_tokenizer, base_model, base_tokenizer, device="cuda"):
    token_num = len(mask_tokenizer.tokenize(generation))
    if token_num > 510:
        gen_input = mask_tokenizer.decode(mask_tokenizer(generation)['input_ids'][:510])
    elif token_num < 50:
        return np.NaN, None
    else:
        gen_input = generation
    output = get_perturbation_results(gen_input, mask_model, mask_tokenizer, base_model, base_tokenizer, device)

    z_score = (output["original_ll"] - output["mean_perturbed_ll"]) / output["std_perturbed_ll"]
    return z_score, output


def detectgpt_batch(args): #discarded
    data = load_data(args.test_file)
    model, tokenizer, mask_model, mask_tokenizer = load_detect_gpt_model(args.base_model, args.mask_model)

    detect_fn = partial(
        detectgpt_detect,
        mask_model=mask_model, mask_tokenizer=mask_tokenizer,
        base_model=model, base_tokenizer=tokenizer,
    )

    resume_idx = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as rf:
            resume_idx = sum(1 for l in rf if l.strip() != "")

    preds = []
    goldens = []
    with open(args.output_file, "a+") as wf:
        for idx, one in tqdm(enumerate(data[resume_idx:]), initial=resume_idx, total=len(data), dynamic_ncols=True):
            text = one[args.text_key]
            label = one["label"]
            if args.prefix_key:
                text = one[args.prefix_key].strip() + " " + text

            pred_z, output = detect_fn(text)
            if pred_z > args.threshold:
                pred = "gpt"
            else:
                pred = "human"

            preds.append(label_mapping[pred])
            goldens.append(label_mapping[label])

            one["pred"] = pred
            one["score"] = pred_z
            one["threshold"] = args.threshold
            one["cache_output"] = output
            wf.write(json.dumps(one, ensure_ascii=False) + "\n")

    metric_report = calc_metrics(goldens, preds)
    logger.info("*************************")
    logger.info(f"[Model] DetectGPT {args.base_model} / {args.mask_model}")
    logger.info(f"[Data] {args.test_file}")
    logger.info(json.dumps(metric_report, indent=4, ensure_ascii=False))
    logger.info("*************************")


def detectgpt_single(sample, text_key, prefix_key, threshold):
    assert base_model is not None, "base_model not init... please check!!!"

    text = sample[text_key]
    if prefix_key:
        text = sample[prefix_key].strip() + " " + text

    pred_score, output = detectgpt_detect(
        text,
        mask_model=mask_model,
        mask_tokenizer=mask_tokenizer,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        device=device,
    )
    if pred_score > threshold:
        pred = "gpt"
    else:
        pred = "human"

    sample["pred"] = pred
    sample["score"] = pred_score
    sample["threshold"] = threshold
    sample["cache_output"] = output
    return sample


class MultiProcessingHelper:
    def __call__(self, data_samples, output_file, func, workers=None, init_fn=None, init_args=None):
        total = len(data_samples)

        with mp.Pool(workers, initializer=init_fn, initargs=init_args) as pool, \
             tqdm(pool.imap(func, data_samples), total=total, dynamic_ncols=True) as pbar, \
             open(output_file, "w") as wf:
                for idx, pred_sample in enumerate(pbar):
                    if "sample_id" in pred_sample:
                        assert pred_sample["sample_id"] == idx
                    wf.write(json.dumps(pred_sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--num_gpu_per_process", type=int, default=1)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")

    parser.add_argument("--base_model", default="openai-community/gpt2-large", type=str)
    parser.add_argument("--mask_model", default="t5-3b", type=str)

    parser.add_argument("--test_file", default="data/CheckGPT/perturbed/test-5k/BackTrans_Helsinki_r3.jsonl")
    parser.add_argument("--output_dir", default="output/checkgpt/detect/detect_gpt")

    parser.add_argument("--prefix_key", default=None, type=str)
    parser.add_argument("--text_key", default="text", type=str)
    parser.add_argument("--threshold", default=3, type=float)
    args = parser.parse_args()

    path_obj = Path(args.test_file)
    filename = path_obj.stem
    rec_dir = path_obj.parts[-2]
    args.output_file = os.path.join(args.output_dir, f"{rec_dir}-{filename}.jsonl")

    dataset = load_data(args.test_file)
    detect_func = partial(
        detectgpt_single,
        text_key=args.text_key,
        prefix_key=args.prefix_key,
        threshold=args.threshold,
    )
    worker = MultiProcessingHelper()
    worker(
        dataset,
        args.output_file,
        detect_func,
        workers=args.num_workers,
        init_fn=init,
        init_args=(args.base_model, args.mask_model, args.gpus, args.num_gpu_per_process, )
    )


if __name__ == "__main__":
    main()
