import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
from utils.detect import calc_metrics


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def set_distributed_args(args):
    if args.local_rank == -1:  # single-node multi-gpu -> DP (or cpu)
        args.n_gpu = torch.cuda.device_count()
        args.local_world_size = 1
        args.device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        args.batch_size = args.batch_size_per_device * args.n_gpu
        args.batch_size = max(args.batch_size, args.batch_size_per_device) # in case of cpu-mode
    else:  # distributed mode -> DDP
        raise NotImplementedError("DDP is not implemented yet...")
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
        args.local_world_size = dist.get_world_size()
        args.device = str(torch.device("cuda", args.local_rank))


def setup_for_distributed_model(
    model: nn.Module,
    device: object,
    optimizer: torch.optim.Optimizer = None,
    n_gpu: int = 1,
    local_rank: int = -1,
    fp16: bool = False,
    fp16_opt_level: str = "O1",
) -> (nn.Module, torch.optim.Optimizer):
    model.to(device)

    if fp16:
        raise NotImplementedError()
        try:
            import apex
            from apex import amp

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if local_rank != -1:
        raise NotImplementedError()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device if device else local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, optimizer


class RobertaClassificationEvaluator(object):
    def __init__(self, config) -> None:
        self.config = config

        self.local_rank = config.local_rank if config.local_rank != -1 else 0
        self.world_size = config.local_world_size
        self.batch_size = config.batch_size
        self.device = config.device

        self.padding = "max_length"
        self.max_seq_length = 512

        if "openai" in config.model_path or "RADAR" in config.model_path:
            self.label_mapping = {
                "gpt": 0,
                "human": 1,
            }
            self.label_names = ["gpt", "human"]
        else:
            self.label_mapping = {
                "human": 0,
                "gpt": 1,
            }
            self.label_names = ["human", "gpt"]

        model_config = RobertaConfig.from_pretrained(config.model_path, num_labels=2)
        tokenizer = RobertaTokenizer.from_pretrained(config.model_path, use_fast=False)
        model = RobertaForSequenceClassification.from_pretrained(config.model_path, config=model_config)
        model, _ = setup_for_distributed_model(
            model, self.device,
            n_gpu=config.n_gpu,
            local_rank=config.local_rank,
        )
        self.model = model
        self.tokenizer = tokenizer

    def prepare_data(
        self,
        data_file,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    ) -> DataLoader:

        def tokenize_func(examples):
            # Tokenize the texts
            text_key = self.config.text_key
            prefix_key = self.config.prefix_key
            label_key = self.config.label_key

            token_args = (
                (examples[text_key],) if prefix_key is None else (examples[text_key], examples[prefix_key])
            )
            result = self.tokenizer(
                *token_args,
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=True
            )

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_key in examples:
                result["label"] = [self.label_mapping[l] for l in examples[label_key]]
            return result

        def label_split_collate_fn(batch):
            input_ids = torch.stack([d['input_ids'] for d in batch], dim=0).to(self.device)
            attn_masks = torch.stack([d['attention_mask'] for d in batch], dim=0).to(self.device)

            labels = [d['label'] for d in batch]

            return {
                'input_ids': input_ids,
                'attention_mask': attn_masks
            }, labels

        _data = {"data": data_file}
        dataset = load_dataset("json", data_files=_data)["data"]
        dataset = dataset.map(
            tokenize_func,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                shuffle=False,
                rank=self.local_rank,
                num_replicas=self.world_size,
                drop_last=False
            )
        else:
            sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
            sampler=sampler,
            collate_fn=label_split_collate_fn,
        )

        return dataloader

    def do_eval(self, test_file, output_file):
        model = self.model
        softmax = nn.Softmax(dim=1)

        with open(test_file, "r") as rf:
            test_samples = [json.loads(line) for line in rf]
        dataloader = self.prepare_data(test_file)

        cur_idx = 0
        all_labels = list()
        all_preds = list()
        with open(output_file, "w") as wf:
            for batch_inputs, batch_labels in tqdm(dataloader, dynamic_ncols=True):
                all_labels += batch_labels

                with torch.no_grad():
                    logits = model(**batch_inputs).logits
                    scores = softmax(logits).detach().cpu()
                    logits = logits.cpu().numpy()

                preds = np.argmax(logits, axis=1).tolist()
                all_preds += preds
                for idx, (pred, pred_score) in enumerate(zip(preds, scores)):
                    gpt_score = float(pred_score[self.label_names.index("gpt")])
                    sample = test_samples[cur_idx]
                    sample["pred"] = self.label_names[pred]
                    sample["score"] = gpt_score # gpt score
                    sample["threshold"] = 0.5
                    wf.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    cur_idx += 1

        metric_report = calc_metrics(all_labels, all_preds)
        print("*************************")
        print(f"[Model] {self.config.model_path}")
        print(f"[Data] {test_file}")
        print(json.dumps(metric_report, indent=4, ensure_ascii=False))
        print("*************************")
        print("\n")

        return metric_report

    def do_predict_one(self, sample):
        softmax = nn.Softmax(dim=1)

        encodings = self.tokenizer(
            sample,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encodings).logits
            scores = softmax(logits).detach().cpu()
        return float(scores[0][1]) # gpt score


def get_gpt_scores(args):
    records = list()
    with open(args.test_file, "r") as rf:
        for line in rf:
            one = json.loads(line)
            records.append(one)

    model = RobertaClassificationEvaluator(args)
    with open(args.output_file, "w") as wf:
        for rec in tqdm(records, dynamic_ncols=True):
            test_sample = rec[args.text_key]
            gpt_score = model.do_predict_one(test_sample)
            rec["gpt_score"] = gpt_score
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1,)
    parser.add_argument("--mode", type=str, default="test", choices=["test", "score"],)
    parser.add_argument("--model_path", type=str, required=True,)

    parser.add_argument("--test_file", type=str, required=True,)
    parser.add_argument("--output_file", type=str, default=None,)
    parser.add_argument("--output_dir", type=str, default="output/checkgpt/detect/classifier",)

    # model args
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--batch_size_per_device", type=int, default=256,)

    # data args
    parser.add_argument("--text_key", type=str, default="text",)
    parser.add_argument("--prefix_key", type=str, default=None, help="Key for input text, second part (after [SEP]).")
    parser.add_argument("--label_key", type=str, default="label", help="Key for jsonl labels.")

    args = parser.parse_args()
    set_distributed_args(args)
    set_seed(args)

    if args.output_file is None:
        assert args.output_dir is not None
        path_obj = Path(args.test_file)
        filename = path_obj.stem
        rec_dir = path_obj.parts[-2]
        args.output_file = os.path.join(args.output_dir, f"{rec_dir}-{filename}.jsonl")

    if args.mode == "score":
        get_gpt_scores(args)
    elif args.mode == "test":
        assert args.test_file is not None
        evaluator = RobertaClassificationEvaluator(args)
        evaluator.do_eval(args.test_file, args.output_file)
    else:
        raise NotImplementedError(f"not supported {args.mode}")


if __name__ == "__main__":
    main()
