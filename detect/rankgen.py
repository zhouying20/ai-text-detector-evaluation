import os
import json
import torch
import logging
import argparse
import nltk
nltk.download('punkt')

from tqdm import tqdm
from pathlib import Path
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, AutoModel
from utils.common import setup_logger
from utils.detect import label_mapping, load_data, calc_metrics


logger = logging.getLogger()
setup_logger(logger)

## Score 越高，Human 可能性越大

# adapted from https://github.com/martiansideofthemoon/rankgen/blob/master/rankgen/rankgen_encoder.py
class RankGenEncoder():
    def __init__(self, model_path, device="cuda", max_batch_size=32, model_size=None, cache_dir=None):
        assert model_path in ["kalpeshk2011/rankgen-t5-xl-all", "kalpeshk2011/rankgen-t5-xl-pg19", "kalpeshk2011/rankgen-t5-base-all", "kalpeshk2011/rankgen-t5-large-all"]
        self.max_batch_size = max_batch_size
        self.device = torch.device(device)
        if model_size is None:
            if "t5-large" in model_path or "t5_large" in model_path:
                self.model_size = "large"
            elif "t5-xl" in model_path or "t5_xl" in model_path:
                self.model_size = "xl"
            else:
                self.model_size = "base"
        else:
            self.model_size = model_size

        self.tokenizer = T5Tokenizer.from_pretrained(f"google/t5-v1_1-{self.model_size}", cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, inputs, vectors_type="prefix", verbose=False, return_input_ids=False):
        tokenizer = self.tokenizer
        max_batch_size = self.max_batch_size
        if isinstance(inputs, str):
            inputs = [inputs]
        if vectors_type == 'prefix':
            inputs = ['pre ' + input for input in inputs]
            max_length = 512
        else:
            inputs = ['suffi ' + input for input in inputs]
            max_length = 128

        all_embeddings = []
        all_input_ids = []
        for i in tqdm(range(0, len(inputs), max_batch_size), total=(len(inputs) // max_batch_size) + 1, disable=not verbose, desc=f"Encoding {vectors_type} inputs:"):
            tokenized_inputs = tokenizer(inputs[i:i + max_batch_size], return_tensors="pt", padding=True)
            for k, v in tokenized_inputs.items():
                tokenized_inputs[k] = v[:, :max_length]
            tokenized_inputs = tokenized_inputs.to(self.device)
            with torch.inference_mode():
                batch_embeddings = self.model(**tokenized_inputs)
            all_embeddings.append(batch_embeddings)
            if return_input_ids:
                all_input_ids.extend(tokenized_inputs.input_ids.cpu().tolist())
        return {
            "embeddings": torch.cat(all_embeddings, dim=0),
            "input_ids": all_input_ids
        }


def load_rankgen_model(base_model_name):
    rankgen_encoder = RankGenEncoder(base_model_name)
    return rankgen_encoder


def rankgen_detect(generation, prefix, encoder):
    prefix_vector = encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    suffix_vector = encoder.encode(generation, vectors_type="suffix")["embeddings"]
    score = (prefix_vector * suffix_vector).sum().item()

    return -1 * score


def do_rankgen(args):
    data = load_data(args.test_file)
    rankgen_encoder = load_rankgen_model(args.base_model)

    preds = []
    goldens = []
    with open(args.output_file, "w") as wf:
        for idx, one in tqdm(enumerate(data), total=len(data), dynamic_ncols=True):
            text = one[args.text_key]
            label = one["label"]
            if args.prefix_key and args.prefix_key in one.keys():
                prefix = one[args.prefix_key].strip()
            else:
                sents = [str(s).strip() for s in sent_tokenize(text)]
                prefix = sents[0]
                text = " ".join(sents[1:])

            pred_score = rankgen_detect(text, prefix, rankgen_encoder)
            if pred_score > args.threshold:
                pred = "gpt"
            else:
                pred = "human"

            preds.append(label_mapping[pred])
            goldens.append(label_mapping[label])

            one["pred"] = pred
            one["score"] = pred_score
            one["threshold"] = args.threshold
            wf.write(json.dumps(one, ensure_ascii=False) + "\n")

    metric_report = calc_metrics(goldens, preds)
    logger.info("*************************")
    logger.info(f"[Model] RankGen {args.base_model}")
    logger.info(f"[Data] {args.test_file}")
    logger.info(json.dumps(metric_report, indent=4, ensure_ascii=False))
    logger.info("*************************")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--num_shards", default=1, type=int)

    parser.add_argument("--base_model", default="kalpeshk2011/rankgen-t5-xl-all", type=str)
    parser.add_argument("--test_file", default="data/CheckGPT/perturbed/test-5k/BackTrans_Helsinki_r3.jsonl")
    parser.add_argument("--output_dir", default="output/checkgpt/detect/rankgen")

    parser.add_argument("--prefix_key", default=None, type=str)
    parser.add_argument("--text_key", default="text", type=str)
    parser.add_argument("--threshold", default=-7.0, type=float)
    args = parser.parse_args()

    path_obj = Path(args.test_file)
    filename = path_obj.stem
    rec_dir = path_obj.parts[-2]
    args.output_file = os.path.join(args.output_dir, f"{rec_dir}-{filename}.jsonl")

    do_rankgen(args)


if __name__ == "__main__":
    main()
