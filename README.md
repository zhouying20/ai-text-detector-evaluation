# Evaluation of the Robustness of AI-Text Detectors

This repository contains code and resources of our paper:

[Navigating the Shadows: Unveiling Effective Disturbances for Modern AI Content Detectors](https://arxiv.org/abs/2406.08922). In ACL 2024, Main Conference

## Datasets

Perturbed AI-text data available at [ai-text-perturbed-data](https://drive.google.com/file/d/1pVzdod5s-i_ylVoKgCbs1PKPhSJOgF-P/view?usp=sharing)

## Attack
```bash
python attack/flint/do_trans.py \
    --test_file data/CheckGPT/original/dev.jsonl \
    --output_dir output/checkgpt/transformations/dev/ \
    --text_key text \
    --gpus 0,1,2,3,4,5,6,7 \
    --num_workers 8 \
    --trans_method dev
```

## Detect
```bash
python detect/classifier.py \
    --test_file data/CheckGPT/perturbed/test-5k/BackTrans_Helsinki_r3.jsonl \
    --output_dir output/checkgpt/detect \
    --model_path output/checkgpt/model/roberta-base \
    --batch_size_per_device 64
```

## Defence
```bash
python defence/train_al.py \
    --model_name_or_path output/checkgpt/model/roberta-base \
    --train_dir data/CheckGPT/perturbed/dev-85k \
    --output_dir output/checkgpt/model/roberta-budget-al \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 256 \
    --max_seq_length 512 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --num_train_epochs 1 \
    --warmup_ratio 0.05 \
    --evaluation_strategy no \
    --save_strategy steps \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 10 \
    --shuffle_train true \
    --do_train true
```

## Citation
If you find our paper/resources useful, please cite:
```
@inproceedings{Zhou2024_ACL,
 author = {Ying Zhou and
           Ben He and
           Le Sun},
 title = {Navigating the Shadows: Unveiling Effective Disturbances for Modern AI Content Detectors},
 booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics.},
 year = {2024},
}
```
