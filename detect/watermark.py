import os
import json
import torch
import hashlib
import logging
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from functools import partial
from transformers import AutoTokenizer, LogitsWarper
from utils.common import setup_logger
from utils.detect import load_data, calc_metrics, label_mapping


logger = logging.getLogger()
setup_logger(logger)


def hash_fn(x):
    # solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits
    x = np.int64(x)
    return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')


def get_z(num_green, total, fraction):
    return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)


def get_green_list(last_token_ids, fraction, vocab_size):
    all_masks = []
    for last_token_id in last_token_ids:
        random_seed = hash_fn(last_token_id)
        rng = np.random.default_rng(random_seed)
        mask = np.full(vocab_size, False)
        mask[:int(fraction * vocab_size)] = True
        rng.shuffle(mask)
        all_masks.append(mask)
    return np.array(all_masks)


def entropy(p):
    """Calculate the entropy of a distribution p using pytorch."""
    return -torch.sum(p * torch.log(p))


def spike_entropy(p, modulus=1):
    """Calculate the spike entropy of a distribution p using pytorch."""
    return torch.sum(p / (1.0 + modulus * p))


class WatermarkLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for watermarking distributions with green-listed tokens. Implementation of https://arxiv.org/abs/2301.10226.
    Args:
        fraction (`float`):
            The fraction of the distribution to be green-listed.
        strength (`float`):
            The strength of the green-listing. A higher value means that the green-listed tokens will have a higher logit score.
    """

    def __init__(self, fraction: float = 0.5, strength: float = 2.0, debug=False):
        self.fraction = fraction
        self.strength = strength
        self.debug = debug
        self.entropies = []
        self.spike_entropies = []
        self.mean_logit = []
        self.watermark_probability_mass = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        green_list_mask = get_green_list(input_ids[:, -1].tolist(), self.fraction, scores.shape[-1])
        green_list_mask = torch.tensor(green_list_mask, dtype=torch.float32, device=scores.device)
        watermark = self.strength * green_list_mask
        final_logits = scores + watermark
        if self.debug:
            distribution = scores.softmax(-1)
            self.entropies.append(entropy(distribution).item())
            self.spike_entropies.append(spike_entropy(distribution).item())
            self.mean_logit.append(torch.mean(scores).item())
            self.watermark_probability_mass.append(torch.sum(final_logits.softmax(-1) * green_list_mask).item())
            if len(self.entropies) % 1000 == 0:
                print(f"Vocab size: {scores.shape[-1]}")
                print(f"Entropy ({len(self.entropies)} tokens): {np.mean(self.entropies):.4f}")
                print(f"Spike Entropy: {np.mean(self.spike_entropies):.4f}")
                print(f"Mean logit: {np.mean(self.mean_logit):.4f}")
                print(f"Watermark probability mass: {np.mean(self.watermark_probability_mass):.4f}")
        return final_logits


def watermark_detect(sequence, watermark_fraction, vocab_size):
    green_tokens = 0
    total_tokens = len(sequence)
    for i in range(1, total_tokens):
        green_list = get_green_list([sequence[i - 1]], watermark_fraction, vocab_size)[0]
        if green_list[sequence[i]]:
            green_tokens += 1
    z_val = get_z(green_tokens, total_tokens - 1, watermark_fraction)
    return z_val


def get_watermark_vocab_size(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if "opt" in base_model:
        vocab_size = 50272
    else:
        vocab_size = tokenizer.vocab_size
    return vocab_size


def do_watermark_detect(args):
    data = load_data(args.test_file)
    vocab_size = get_watermark_vocab_size(args.base_model)

    detect_fn = partial(
        watermark_detect,
        watermark_fraction=args.watermark_fraction,
        vocab_size=vocab_size,
    )

    preds = []
    goldens = []
    with open(args.output_file, "w") as wf:
        for idx, one in tqdm(enumerate(data), total=len(data)):
            text = one[args.text_key]
            label = one["label"]
            if args.prefix_key:
                text = one[args.prefix_key].strip() + " " + text

            pred_score = detect_fn(text)
            if pred_score > args.threshold:
                pred = "human"
            else:
                pred = "gpt"

            preds.append(label_mapping[pred])
            goldens.append(label_mapping[label])

            one["pred"] = pred
            one["score"] = pred_score
            one["threshold"] = args.threshold
            wf.write(json.dumps(one, ensure_ascii=False) + "\n")

    metric_report = calc_metrics(goldens, preds)
    logger.info("*************************")
    logger.info(f"[Model] Watermark {args.base_model}")
    logger.info(f"[Data] {args.test_file}")
    logger.info(json.dumps(metric_report, indent=4, ensure_ascii=False))
    logger.info("*************************")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--num_shards", default=1, type=int)

    parser.add_argument("--base_model", default="gpt2-xl", type=str)
    parser.add_argument("--watermark_fraction", default=0.5, type=float)

    parser.add_argument("--test_file", default="data/CheckGPT/perturbed/test-5k/BackTrans_Helsinki_r3.jsonl")
    parser.add_argument("--output_dir", default="output/checkgpt/detect/watermark")

    parser.add_argument("--prefix_key", default=None, type=str)
    parser.add_argument("--text_key", default="text", type=str)

    parser.add_argument("--threshold", default=4.0, type=float)
    args = parser.parse_args()

    path_obj = Path(args.test_file)
    filename = path_obj.stem
    rec_dir = path_obj.parts[-2]
    args.output_file = os.path.join(args.output_dir, f"{rec_dir}-{filename}.jsonl")

    do_watermark_detect(args)


if __name__ == "__main__":
    main()
