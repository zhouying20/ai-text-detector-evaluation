import os
import json
import logging
import argparse
import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path
from sklearn.metrics import roc_curve
from utils.common import setup_logger
from utils.detect import label_mapping, calc_metrics, calc_attack_metrics


logger = logging.getLogger()
setup_logger(logger)


def align_score(origin_score):
    if origin_score is None or np.isnan(origin_score): # set it to a bigger value -> GPT
        return np.inf
    return float(origin_score)


class DetectorMetricCalculator:
    def __init__(
        self,
        origin_file, test_files,
        threshold=None,
        optimal_threshold=None,
        seperate_decision=False,
    ) -> None:
        self.origin_file = origin_file
        self.test_files = test_files
        self.optimal_threshold = optimal_threshold
        self.seperate_decision = seperate_decision

        if not seperate_decision:
            assert origin_file is not None
            origin_samples, origin_thres = self.load_pred_samples(origin_file)

            if threshold is not None:
                self.threshold = threshold
            elif optimal_threshold is not None:
                self.threshold = self.find_optimal_threshold(origin_samples, method=optimal_threshold)
            else:
                self.threshold = origin_thres

    def find_optimal_threshold(self, data_samples, method=""):
        """
            Reference: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
        """
        y_true = list()
        y_scores = list()
        for sample in data_samples:
            if np.isinf(sample["score"]): continue # not consider outliers
            y_true.append(label_mapping[sample["golden"]])
            y_scores.append(sample["score"])

        if method == "youden":
            optimal_threshold = self._optimal_threshold_youden(y_true, y_scores)
        elif method == "youden-dipper":
            optimal_threshold = self._optimal_threshold_youden_dipper(y_true, y_scores)
        elif method == "gmean":
            optimal_threshold = self._optimal_threshold_gmean(y_true, y_scores)
        elif method == "dipper":
            optimal_threshold = self._optimal_threshold_dipper(y_true, y_scores)
        else:
            raise NotImplementedError(f"not supported method for optimal threshold -> {method}")

        return optimal_threshold

    def _optimal_threshold_youden(self, y_true, y_scores):
        # Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]

    def _optimal_threshold_youden_dipper(self, y_true, y_scores, target_fpr=[0.01, 0.2]):
        # Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        condition = (fpr >= target_fpr[0]) & (fpr <= target_fpr[1])
        fpr = fpr[condition]
        tpr = tpr[condition]
        thresholds = thresholds[condition]
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]

    def _optimal_threshold_gmean(self, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        optimal_idx = np.argmax(gmeans)
        return thresholds[optimal_idx]

    def _optimal_threshold_dipper(self, y_true, y_scores, target_fpr=0.01):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_idx = None
        for idx in range(len(fpr)):
            if fpr[idx] >= target_fpr:
                if idx == 0:
                    optimal_idx = [idx,]
                else:
                    optimal_idx = [idx-1, idx]
                break

        if optimal_idx is None:
            res_thres = thresholds[-1]
        else:
            res_thres = np.mean([thresholds[i] for i in optimal_idx])
        return res_thres

    def load_pred_samples(self, data_file):
        file_thres = None
        pred_samples = list()
        with open(data_file, "r") as rf:
            for line in rf:
                lj = json.loads(line)
                if file_thres is None:
                    file_thres = lj["threshold"]
                else:
                    assert lj["threshold"] == file_thres, "plase ensure all line with same threshold in origin file"
                pred_samples.append({
                    "golden": lj["label"],
                    # "pred": lj["pred"],
                    "score": align_score(lj["score"]),
                })
        return pred_samples, file_thres

    def file_metric(self, data_file, origin_preds=None):
        file_obj = Path(data_file)
        method = file_obj.parts[-2]

        pred_samples, file_thres = self.load_pred_samples(data_file)

        if self.seperate_decision:
            threshold = self.find_optimal_threshold(pred_samples, method=self.optimal_threshold)
        else:
            threshold = self.threshold

        goldens = list()
        preds = list()
        scores = list()
        for sample in pred_samples:
            goldens.append(label_mapping[sample["golden"]])
            scores.append(sample["score"])
            if sample["score"] >= threshold:
                preds.append(label_mapping["gpt"])
            else:
                preds.append(label_mapping["human"])

        class_report = calc_metrics(goldens, preds, scores)
        attack_report = calc_attack_metrics(goldens, preds, origin_preds)
        metric_report = dict(list(class_report.items()) + list(attack_report.items()))
        logger.info("*************************")
        logger.info(f"[Method] {method}")
        logger.info(f"[Data] {data_file}")
        logger.info(f"[Data Threshold] {file_thres}")
        logger.info(f"[Used Threshold] {threshold}")
        logger.info(json.dumps(metric_report, indent=4, ensure_ascii=False))
        logger.info("*************************")
        logger.info("\n")

        res_report = {
            "method": method,
            "data": data_file,
        }
        res_report = dict(list(res_report.items()) + list(metric_report.items()))
        res_report["threshold"] = threshold
        return res_report, preds

    def batch_metric(self, output_file):
        records = list()
        if self.origin_file is not None:
            origin_report, origin_preds = self.file_metric(self.origin_file, origin_preds=None)
            records.append(origin_report)
        else:
            origin_preds = None

        for tf in self.test_files:
            test_report, _ = self.file_metric(tf, origin_preds=origin_preds)
            records.append(test_report)
        rec_df = pd.DataFrame.from_records(records)
        rec_df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--origin_file", type=str, default=None,)
    parser.add_argument("--test_dir", type=str, required=True,)
    parser.add_argument("--output_file", type=str, required=True,)
    parser.add_argument("--optimal_threshold", type=str, default=None, choices=["youden", "youden-dipper", "gmean", "dipper"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seperate_decision", action="store_true")
    args = parser.parse_args()

    if args.threshold is not None and args.optimal_threshold:
        logger.warning("--threshold and --optimal_threshold should not passed in the same time...")
        exit()

    test_files = sorted(glob(os.path.join(args.test_dir, "*.jsonl")))
    if args.origin_file is None:
        assert "Ace" in test_files[0]
        args.origin_file = test_files[0]
        test_files = test_files[1:]
    else:
        if "Ace" in test_files[0]:
            assert test_files[0] == os.path.abspath(args.origin_file)
            test_files = test_files[1:]

    calc = DetectorMetricCalculator(
        args.origin_file, test_files,
        threshold=args.threshold, 
        optimal_threshold=args.optimal_threshold,
        seperate_decision=args.seperate_decision
    )
    calc.batch_metric(args.output_file)


if __name__ == "__main__":
    main()
