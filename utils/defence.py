import os
import json
import random
import logging

from glob import glob
from pathlib import Path
from datasets import Dataset


logger = logging.getLogger()


def load_train_dataset(train_dir, excludes=None, shuffle=True):
    def filter_excludes(datafile):
        filename = Path(datafile).name
        return all(ex not in filename for ex in excludes)

    trans_files = sorted(glob(os.path.join(train_dir, "*.jsonl")))
    total_num = len(trans_files)
    if excludes is not None:
        trans_files = list(filter(filter_excludes, trans_files))
        print(f"total {total_num} files; remain {len(trans_files)} files.")
        # print(trans_files)

    train_samples = list()
    for file in trans_files:
        with open(file, "r") as rf:
            for line in rf:
                train_samples.append(json.loads(line))

    if shuffle:
        random.shuffle(train_samples)

    logger.info(f"[Data] load {len(train_samples)} samples from {len(trans_files)} trans files in {train_dir}")

    return Dataset.from_list(train_samples)