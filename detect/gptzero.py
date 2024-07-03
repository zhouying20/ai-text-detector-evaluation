import os
import time
import json
import nltk
import random
import logging
import argparse
import requests
nltk.download('punkt')

from tqdm import tqdm
from pathlib import Path
from utils.detect import load_data, calc_metrics, label_mapping
from utils.common import setup_logger


GPTZERO_API_KEYS = [
    ""
]

logger = logging.getLogger()
setup_logger(logger)


def gptzero_detect(generation):
    gptzero_key = random.choice(GPTZERO_API_KEYS)

    url = "https://api.gptzero.me/v2/predict/text"
    payload = {
        "document": generation,
        "version": "2024-01-09"
    }
    headers = {
        "x-api-key": gptzero_key,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    attempts = 0
    while response.status_code != 200 and attempts < 10:
        attempts += 1
        time.sleep(2)
        gptzero_key = random.choice(GPTZERO_API_KEYS)
        logger.warning(f"[Error] Request with error: {response.text}")
        logger.warning(f"[Error] Switching to GPT Zero API key: {gptzero_key}")
        headers["x-api-key"] = gptzero_key
        response = requests.post(url, json=payload, headers=headers)

    try:
        res_dict = response.json()
        doc_stat = res_dict["documents"][0]
        ai_score = doc_stat["class_probabilities"]["ai"]
        mixed_score = doc_stat["class_probabilities"]["mixed"]
    except Exception as e:
        logger.warning(f"[Error] Parsing {response.text}, got {e}")
        return None, None
    else:
        # time.sleep(0.2)
        return float(ai_score + mixed_score), res_dict


def do_gptzero(args):
    data = load_data(args.test_file)
    if args.size_limit:
        data = data[:args.size_limit]

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

            pred_score, pred_cache = gptzero_detect(text)
            if pred_score > args.threshold:
                pred = "gpt"
            else:
                pred = "human"

            preds.append(label_mapping[pred])
            goldens.append(label_mapping[label])

            one["pred"] = pred
            one["score"] = pred_score
            one["threshold"] = args.threshold
            one["cache_output"] = pred_cache
            wf.write(json.dumps(one, ensure_ascii=False) + "\n")

    metric_report = calc_metrics(goldens, preds)
    logger.info("*************************")
    logger.info(f"[Model] GPTZero")
    logger.info(f"[Data] {args.test_file}")
    logger.info(json.dumps(metric_report, indent=4, ensure_ascii=False))
    logger.info("*************************")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--num_shards", default=1, type=int)

    parser.add_argument("--test_file", default="data/CheckGPT/perturbed/test-5k/BackTrans_Helsinki_r3.jsonl")
    parser.add_argument("--output_dir", default="output/checkgpt/detect/gptzero")

    parser.add_argument("--prefix_key", default=None, type=str)
    parser.add_argument("--text_key", default="text", type=str)
    parser.add_argument("--threshold", default=0.88, type=float)
    parser.add_argument("--size_limit", default=500, type=int)
    args = parser.parse_args()

    path_obj = Path(args.test_file)
    filename = path_obj.stem
    rec_dir = path_obj.parts[-2]
    args.output_file = os.path.join(args.output_dir, f"{rec_dir}-{filename}.jsonl")

    do_gptzero(args)


if __name__ == "__main__":
    main()
