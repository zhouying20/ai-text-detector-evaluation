import os
import re
import json
import pickle
import string
import functools
import numpy as np
import collections as cll

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

from detect.lib_sim.models import load_model
from detect.lib_sim.embed_sentences import embed_all, similarity


label_mapping = {
    "human": 0,
    "gpt": 1,
}
label_names = ["human", "gpt"]


def load_data(test_file, local_rank=0, num_shards=1):
    with open(test_file, "r") as rf:
        data = [json.loads(line.strip()) for line in rf]

    if num_shards > 1:
        partitions = form_partitions(data, num_shards)
        data = partitions[local_rank]

    return data


def form_partitions(dataset, num_shards):
    p_indices = np.round(np.linspace(0, len(dataset), num_shards + 1))
    p_indices = [int(x) for x in p_indices]
    partitions = [dataset[p_indices[i]:p_indices[i + 1]] for i in range(len(p_indices) - 1)]
    assert len(partitions) == num_shards
    return partitions


def calc_metrics(labels, preds, scores=None):
    labels = np.array(labels)
    preds = np.array(preds)
    if scores is not None:
        auc_score = np.array(scores)
    else:
        auc_score = preds

    # auc = roc_auc_score(labels, auc_score)
    clf_report = classification_report(labels, preds, target_names=label_names, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel() / len(labels)
    binary_f1 = f1_score(labels, preds)

    return {
        "f1": binary_f1,
        "acc": clf_report["accuracy"],
        "gpt_acc": clf_report["gpt"]["recall"],
        "human_acc": clf_report["human"]["recall"],
        "avg_f1": clf_report["weighted avg"]["f1-score"],
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }
    # return {
    #     "auc_roc": auc,
    #     "f1": binary_f1,
    #     "acc": clf_report["accuracy"],
    #     "avg_prec": clf_report["weighted avg"]["precision"],
    #     "avg_recall": clf_report["weighted avg"]["recall"],
    #     "avg_f1": clf_report["weighted avg"]["f1-score"],
    #     "gpt_prec": clf_report["gpt"]["precision"],
    #     "gpt_recall": clf_report["gpt"]["recall"],
    #     "gpt_f1": clf_report["gpt"]["f1-score"],
    #     "human_prec": clf_report["human"]["precision"],
    #     "human_recall": clf_report["human"]["recall"],
    #     "human_f1": clf_report["human"]["f1-score"],
    #     "true_positive": tp,
    #     "false_positive": fp,
    #     "true_negative": tn,
    #     "false_negative": fn,
    # }


def calc_attack_metrics(labels, preds, origin_preds):
    if origin_preds is None:
        return {
            "attack_success": 0.0,
            "attack_gpt_success": 0.0,
            "attack_human_success": 0.0,
        }

    gpt_cnt = 0
    gpt_flip = 0
    human_cnt = 0
    human_flip = 0
    for golden, pred, origin in zip(labels, preds, origin_preds, strict=True):
        if golden == label_mapping["gpt"]:
            gpt_cnt += 1
            if origin == golden and pred != golden:
                gpt_flip += 1
        elif golden == label_mapping["human"]:
            human_cnt += 1
            if origin == golden and pred != golden:
                human_flip += 1
        else:
            raise NotImplementedError(f"not supported label -> {golden}")

    return {
        "attack_success": (gpt_flip + human_flip) / len(labels),
        "attack_gpt_success": gpt_flip / gpt_cnt,
        "attack_human_success": human_flip / human_cnt,
    }


# def save_plot():
#     stats = get_roc(acc_gold, acc_gen)
#     stats2 = get_roc(acc_gold, acc_pp0)
#     logger.info_tpr_target(stats[0], stats[1], "generation", args.target_fpr)
#     logger.info_tpr_target(stats2[0], stats2[1], "paraphrase", args.target_fpr)

#     with open("roc_plots/detectgpt.pkl", 'wb') as f:
#         pickle.dump((stats, stats2), f)


def do_sim_stuff(gen_tokens, gold_tokens, pp0_tokens, sim_cache, sim_fn, args, sim_gold, sim_pp0):
    if sim_fn is None:
        sim_gold.append(False)
        sim_pp0.append(False)
        return
    gen_vec, _ = sim_fn(gen_tokens, sim_cache)
    gold_vec, _ = sim_fn(gold_tokens, sim_cache)
    pp0_vec, _ = sim_fn(pp0_tokens, sim_cache)
    sim_gold.append(similarity(gen_vec, gold_vec) > args.sim_threshold)
    sim_pp0.append(similarity(gen_vec, pp0_vec) > args.sim_threshold)


def load_sim_stuff(args):
    sim_gold = []
    sim_pp0 = []

    if os.path.exists(args.sim_cache):
        with open(args.sim_cache, "rb") as f:
            sim_cache = pickle.load(f)
        # save a copy of cache as a backup
        with open(args.sim_cache + ".bak", "wb") as f:
            pickle.dump(sim_cache, f)
    else:
        sim_cache = {}

    # load paraphrase model
    if os.path.exists("dipper_paraphrases/sim/model.para.lc.100.pt"):
        paraphrase_model = load_model("dipper_paraphrases/sim/model.para.lc.100.pt")
        paraphrase_model.eval()
        embedder = functools.partial(embed_all, model=paraphrase_model, disable=True)
        sim_fn = functools.partial(get_sim_vectors, embedder=embedder)
    else:
        sim_fn = None
    return sim_gold, sim_pp0, sim_cache, sim_fn


def get_sim_vectors(sequence, cache, embedder):
    cache_updated = False
    if sequence in cache:
        gen_vec = cache[sequence]
    else:
        gen_vec = embedder(sentences=[sequence])[0]
        cache[sequence] = gen_vec
        cache_updated = True
    return gen_vec, cache_updated


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def n_gram_f1_score(prediction, ground_truth, gram=1, stopwords=None):
    """Calculate word level F1 score."""
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    prediction_tokens = [
        " ".join(prediction_tokens[i:i + gram])
        for i in range(0, len(prediction_tokens) - gram + 1)
    ]
    ground_truth_tokens = [
        " ".join(ground_truth_tokens[i:i + gram])
        for i in range(0, len(ground_truth_tokens) - gram + 1)
    ]

    if stopwords:
        prediction_tokens = [x for x in prediction_tokens if x not in stopwords]
        ground_truth_tokens = [x for x in ground_truth_tokens if x not in stopwords]

    if not prediction_tokens or not ground_truth_tokens:
        return 1.0, 1.0, 1.0, True
    common = cll.Counter(prediction_tokens) & cll.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0, False
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, False
