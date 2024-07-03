import os
import json
import textstat
import numpy as np
import pandas as pd
from glob import glob


class TextStatEvaluator(object):
    def __init__(self, text_key="text") -> None:
        self.text_key = text_key

    def calc_metrics(self, samples):
        res_each_scores = {
            "flesch_reading_ease": list(),
            "flesch_kincaid_grade": list(),
            "gunning_fog": list(),
            "smog_index": list(),
            "automated_readability_index": list(),
            "coleman_liau_index": list(),
            "linsear_write_formula": list(),
            "dale_chall_readability_score": list(),
            "text_standard": list(),
            "spache_readability": list(),
            "difficult_words": list(),
        }
        for sample in samples:
            res_each_scores["flesch_reading_ease"].append(textstat.flesch_reading_ease(sample))
            res_each_scores["flesch_kincaid_grade"].append(textstat.flesch_kincaid_grade(sample))
            res_each_scores["gunning_fog"].append(textstat.gunning_fog(sample))
            res_each_scores["smog_index"].append(textstat.smog_index(sample))
            res_each_scores["automated_readability_index"].append(textstat.automated_readability_index(sample))
            res_each_scores["coleman_liau_index"].append(textstat.coleman_liau_index(sample))
            res_each_scores["linsear_write_formula"].append(textstat.linsear_write_formula(sample))
            res_each_scores["dale_chall_readability_score"].append(textstat.dale_chall_readability_score(sample))
            res_each_scores["text_standard"].append(textstat.text_standard(sample, float_output=True))
            res_each_scores["spache_readability"].append(textstat.spache_readability(sample))
            res_each_scores["difficult_words"].append(textstat.difficult_words(sample))

        return res_each_scores

    def load_text_list(self, text_file):
        res = list()
        with open(text_file, "r") as rf:
            for line in rf:
                lj = json.loads(line)
                if lj["label"] == "gpt":
                    res.append(lj[self.text_key])
        return res

    def do_eval(self, perturbed_file):
        perturbed_samples = self.load_text_list(perturbed_file)
        perturbed_scores = self.calc_metrics(perturbed_samples)

        metric_report = dict()
        for metric in perturbed_scores.keys():
            ps = np.array(perturbed_scores[metric])
            metric_report[metric] = np.mean(ps)

        print("*************************")
        print(f"[{perturbed_file}]")
        print(json.dumps(metric_report, indent=4, ensure_ascii=False))
        print("*************************")
        print("\n")

        return metric_report


def main():
    test_dir = "data/CheckGPT/perturbed/test-5k"
    output_file = "output/checkgpt/similarity/readability.csv"
    test_files = sorted(glob(os.path.join(test_dir, "*.jsonl")))

    res_evals = list()
    evaluator = TextStatEvaluator("text")
    for test in test_files:
        eval_record = {
            "test_file": test,
        }
        eval_report = evaluator.do_eval(test)
        res_evals.append(dict(list(eval_record.items()) + list(eval_report.items())))
    res_df = pd.DataFrame.from_records(res_evals)
    res_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
