import json
import string
import logging

from typing import List
from tqdm import tqdm
from textflint.adapter import auto_dataset
from textflint.input.dataset import Dataset


logger = logging.getLogger()


def load_textflint_dataset(test_file, prefix_key=None, text_key="text"):
    # prepare test data
    test_data_set = list()
    test_samples = list()
    with open(test_file, "r") as rf:
        for idx, line in enumerate(rf):
            r_j = json.loads(line.strip())
            x_text = ""
            if prefix_key is not None and prefix_key in r_j.keys():
                prefix = r_j[prefix_key].strip()
                if prefix[-1] not in string.punctuation:
                    x_text += prefix + ". "
                else:
                    x_text += prefix + " "
            x_text += r_j[text_key]
            if idx == 0: print(x_text)
            test_data_set.append({
                "x": x_text,
            })
            test_samples.append(r_j)
    dataset = auto_dataset(test_data_set, task="UT")

    return dataset, test_samples


def do_transform(
    sample, data_sample,
    trans_fn,
    gpt_only=False,
    text_key="text",
    max_trans=1,
):
    res_samples = list()
    if gpt_only and data_sample["label"] == "human":
        return [data_sample, ]
    else:
        trans_rst = trans_fn.transform(
            sample, n=max_trans, field="x",
        )
        if len(trans_rst) < max_trans:
            print(f"required {max_trans} transformations <- {len(trans_rst)}")
            trans_rst = [sample, ]
        for one_res in trans_rst:
            res_dict = one_res.dump()
            to_write = {
                text_key: res_dict["x"],
                "sample_id": res_dict["sample_id"],
            }
            for k, v in data_sample.items():
                if k != text_key: to_write[k] = v
            res_samples.append(to_write)
    return res_samples


def do_transform_batch(
    dataset: Dataset, data_samples: List[dict],
    trans_fn, output_file: str,
    gpt_only=False,
    text_key="text",
    max_trans=1
):
    assert len(dataset) == len(data_samples)

    # initialize current index of dataset
    dataset.init_iter()
    logger.info(f"******Start {trans_fn}!******")

    with open(output_file, "w") as wf:
        for idx, sample in tqdm(enumerate(dataset), dynamic_ncols=True, total=len(dataset)):
            data_point = data_samples[idx]
            res_samples = do_transform(
                sample, data_point, 
                trans_fn=trans_fn,
                gpt_only=gpt_only,
                text_key=text_key,
                max_trans=max_trans,
            )
            for to_write in res_samples:
                if "sample_id" in to_write.keys():
                    assert to_write["sample_id"] == idx
                else:
                    to_write["sample_id"] = idx
                    logger.debug(f"origin sample returned -> {to_write}")
                wf.write(json.dumps(to_write, ensure_ascii=False) + "\n")

    logger.info(f"******Finish {trans_fn}!******")
