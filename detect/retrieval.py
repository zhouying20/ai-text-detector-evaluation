import os
import json
import logging
import argparse
import numpy as np

import nltk
nltk.download('punkt')

from tqdm import tqdm
from pathlib import Path
from functools import partial
from retriv import SearchEngine
from utils.common import setup_logger
from utils.detect import n_gram_f1_score, load_data, calc_metrics, label_mapping
from detect.lib_sim.models import load_model
from detect.lib_sim.embed_sentences import embed_all


logger = logging.getLogger()
setup_logger(logger)

## Score 越高，GPT 可能性越大
def load_retrieval_corpus(base_data_folder, retrieval_corpus, prefix_key=None, text_key="text", min_tokens=50):
    corpus_folder = "detect/data/open-generation-data"
    if retrieval_corpus == "sharegpt":
        corpora_files = [
            f"{corpus_folder}/ShareGPT_longer_than_100_shorter_than_300.jsonl",
        ]
    elif retrieval_corpus == "sharegpt-test":
        corpora_files = [
            f"{corpus_folder}/ShareGPT_longer_than_100_shorter_than_300.jsonl",
            f"{base_data_folder}/test.jsonl",
        ]
    elif retrieval_corpus == "pooled":
        corpora_files = [
            f"{base_data_folder}/corpus.jsonl",
            f"{base_data_folder}/test.jsonl",
        ]
    elif retrieval_corpus == "train":
        corpora_files = [
            f"{base_data_folder}/corpus.jsonl",
        ]
    elif retrieval_corpus == "test":
        corpora_files = [
            f"{base_data_folder}/test.jsonl",
        ]
    else:
        raise NotImplementedError(f"not supported corpus -> {retrieval_corpus}")

    for fname in corpora_files:
        assert os.path.exists(fname), f"{fname} not exists, please check!!!"

    gens_list = []
    for op_file in corpora_files:
        with open(op_file, "r") as rf:
            data = [json.loads(line.strip()) for line in rf]

        # iterate over data and tokenize each instance
        for idx, one in tqdm(enumerate(data), total=len(data), dynamic_ncols=True):
            if "label" in one.keys() and one["label"] == "human": # load gpt text only
                continue

            if op_file.startswith(corpus_folder):
                if isinstance(one["gen_completion"], str):
                    gen_text = one["gen_completion"]
                else:
                    gen_text = one["gen_completion"][0]
            elif op_file.startswith(base_data_folder):
                gen_text = one[text_key].strip()
                if prefix_key and prefix_key in one.keys():
                    gen_text = one[prefix_key].strip() + " " + gen_text
            else:
                raise NotImplementedError()

            gen_tokens = gen_text.split()
            if len(gen_tokens) <= min_tokens:
                continue
            gens_list.append(" ".join(gen_tokens))

    print("Number of gens: ", len(gens_list))
    return gens_list


def load_retrieval_model(sim_model_name, sp_model=None):
    sim_model = load_model(sim_model_name, sp_model=sp_model)
    sim_model.eval()
    embedder = partial(embed_all, model=sim_model)
    return embedder


def do_retrieval_detect(args):
    test_samples = load_data(args.test_file)
    corpus_list = load_retrieval_corpus(
        args.corpus_folder, 
        args.retrieval_corpus,
        prefix_key=args.prefix_key,
        text_key=args.text_key,
        min_tokens=args.min_tokens,
    )

    # index the cand_gens
    if args.technique == "sim":
        embedder = load_retrieval_model(args.sim_model, sp_model=args.sim_sp_model)
        gen_vecs = embedder(sentences=corpus_list, disable=True)
    elif args.technique == "bm25":
        collection = [
            {"text": x, "id": f"doc_{i}"}
            for i, x in enumerate(corpus_list)
        ]
        se = SearchEngine(f"index-{args.output_file.split('/')[1]}")
        se.index(collection)
    else:
        raise NotImplementedError(f"not supported -> {args.technique}")

    # iterate over cands and get similarity scores
    preds = []
    goldens = []
    with open(args.output_file, "w") as wf:
        for sample in tqdm(test_samples, dynamic_ncols=True):
            text = sample[args.text_key]
            label = sample["label"]
            if args.prefix_key and args.prefix_key in sample.keys():
                text = sample[args.prefix_key].strip() + " " + text

            if args.technique == "sim":
                cand_vecs = embedder(sentences=[text, ], disable=True)
                # get similarity scores
                sim_matrix = np.matmul(cand_vecs, gen_vecs.T)
                norm_matrix = np.linalg.norm(cand_vecs, axis=1, keepdims=True) * np.linalg.norm(gen_vecs, axis=1, keepdims=True).T
                sim_scores = sim_matrix / norm_matrix
                pred_score = np.max(sim_scores, axis=1)[0]
            elif args.technique == "bm25":
                try:
                    bm25_res = se.search(text)[0] # most similar document in bm25
                    pred_score = n_gram_f1_score(text, bm25_res['text'])[2]
                except Exception as e:
                    logger.warning(f"got error when retrieval from bm25 index -> {e}")
                    pred_score = None
            else:
                raise NotImplementedError()

            if pred_score is not None:
                pred_score = float(pred_score)

            if pred_score is not None and pred_score > args.threshold:
                pred = "gpt"
            else:
                pred = "human"

            preds.append(label_mapping[pred])
            goldens.append(label_mapping[label])

            sample["pred"] = pred
            sample["score"] = pred_score
            sample["threshold"] = args.threshold
            wf.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metric_report = calc_metrics(goldens, preds)
    logger.info("*************************")
    logger.info(f"[Retrieval] {args.technique}")
    logger.info(f"[Corpus] {args.retrieval_corpus}")
    logger.info(f"[Data] {args.test_file}")
    logger.info(json.dumps(metric_report, indent=4, ensure_ascii=False))
    logger.info("*************************")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--num_shards", default=1, type=int)

    parser.add_argument("--sim_model", default="detect/data/sim_models/model.para.lc.100.pt", type=str)
    parser.add_argument("--sim_sp_model", default="detect/data/sim_models/paranmt.model", type=str)
    parser.add_argument("--technique", default="bm25", type=str, choices=["bm25", "sim"])
    parser.add_argument("--corpus_folder", default="data/CheckGPT/original", type=str)
    parser.add_argument("--retrieval_corpus", default="sharegpt", type=str, choices=["sharegpt", "sharegpt-test", "pooled", "train", "test"])

    parser.add_argument("--test_file", default="data/CheckGPT/perturbed/test-5k/BackTrans_Helsinki_r3.jsonl")
    parser.add_argument("--output_dir", default="output/checkgpt/detect/retrieval")

    parser.add_argument("--prefix_key", default=None, type=str)
    parser.add_argument("--text_key", default="text", type=str)

    parser.add_argument("--threshold", default=0.75, type=float)
    parser.add_argument("--min_tokens", default=50, type=int)
    args = parser.parse_args()

    path_obj = Path(args.test_file)
    filename = path_obj.stem
    rec_dir = path_obj.parts[-2]
    args.output_file = os.path.join(args.output_dir, f"{args.technique}-{args.retrieval_corpus}-{rec_dir}-{filename}.jsonl")

    do_retrieval_detect(args)


if __name__ == "__main__":
    main()
