import os
import json
import argparse
from tqdm import tqdm
from functools import partial
from detect.lib_sim.models import load_model
from detect.lib_sim.embed_sentences import embed_all, similarity


def load_sim_model(model_path="dipper_paraphrases/sim/model.para.lc.100.pt"):
    # load paraphrase model
    paraphrase_model = load_model(model_path, sp_model="detect/data/sim_models/paranmt.model")
    paraphrase_model.eval()
    embedder = partial(embed_all, model=paraphrase_model, disable=True)
    return embedder


def score_similarity(human_text, gpt_text, embedder):
    human_vec, gpt_vec = embedder(sentences=[human_text, gpt_text])
    return similarity(human_vec, gpt_vec)


def main(args):
    embedder = load_sim_model(args.sim_model_path)

    origins = list()
    with open(args.origin_file, "r") as rf:
        for line in rf:
            origins.append(json.loads(line))

    with open(args.test_file, "r") as rf, open(args.output_file, "w") as wf:
        for idx, line in tqdm(enumerate(rf)):
            ori = origins[idx]
            cur = json.loads(line)
            assert ori["uid"] == cur["uid"]
            sim_score = score_similarity(ori[args.text_key], cur[args.text_key], embedder)
            cur["sim_score"] = float(sim_score)
            wf.write(json.dumps(cur, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--sim_model_path", type=str, default="detect/data/sim_models/model.para.lc.100.pt",)
    parser.add_argument("--origin_file", type=str, default=None,)
    parser.add_argument("--test_file", type=str, default=None,)
    parser.add_argument("--output_file", type=str, default=None,)
    parser.add_argument("--text_key", type=str, default="text",)
    args = parser.parse_args()

    main(args)
