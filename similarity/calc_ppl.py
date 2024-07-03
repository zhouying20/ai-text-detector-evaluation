import json
import torch
import argparse

from tqdm import tqdm
from similarity.models.llm_model import LlamaPPLModel


def main(args):
    # use_deepspeed = args.local_rank != -1
    llama_ppl = LlamaPPLModel(seq_max_len=1024, batch_size=2, use_deepspeed=False)
    # if use_deepspeed:
    #     import deepspeed
    #     llama_ppl.model = deepspeed.init_inference(
    #         llama_ppl.model,
    #         mp_size=4,
    #         dtype=torch.float16,
    #         checkpoint=None,
    #         replace_with_kernel_inject=True,
    #     )

    with open(args.test_file, "r") as rf, open(args.output_file, "w") as wf:
        for line in tqdm(rf.readlines(), dynamic_ncols=True):
            rec = json.loads(line)
            text_sample = rec[args.text_key].strip()
            if len(text_sample.split(" ")) > 1: # assert sentence contains more than 2 words
                rec["ppl"] = llama_ppl.eval_ppl(text_sample)[0]
            else:
                rec["ppl"] = None
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument(
        "--local-rank", "--local_rank", type=int, default=-1,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument(
        "--test_file", type=str, default=None,
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Where to save the final results.",
    )
    parser.add_argument(
        "--text_key", type=str, default="text",
        help="Key for input text.",
    )
    args = parser.parse_args()

    main(args)
