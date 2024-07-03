import os
import json
import time
from glob import glob
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
from ast import literal_eval


client = OpenAI(
    api_key="",
    api_version="2023-03-15-preview",
)


judge_prompt = """You are given an array of 13 sentences. Please rate these sentences and reply with an array of scores assigned to these sentences. Each score is on a scale from 1 to 10, the higher the score, the sentence is written more like a human. Your reply example: [2,2,2,2,2,2,2,2,2,2,2,2,2].

Sentences:
{sents}
"""


def get_gpt4_judge_scores(sents):
    input_sents = "\n".join(f"{idx}. {s}" for idx, s in enumerate(sents, start=1))
    prompt = judge_prompt.format(sents=input_sents)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        time.sleep(0.02)
        res_text = response.choices[0].message.content
        res_list = literal_eval(res_text)
    except Exception as e:
        print(e)
        return None, None
    else:
        return res_list, response.dict()


def main():
    base_dir = "data/CheckGPT/perturbed/test-5k"
    save_file = "output/checkgpt/similarity/gpt_judgement.jsonl"
    file_list = sorted(glob(os.path.join(base_dir, "*.jsonl")))
    assert "Ace" in file_list[0]
    origin_file = file_list[0]
    file_list = file_list[1:]

    gpt_idxs = list()
    gpt_texts = list()
    with open(origin_file, "r") as rf:
        for idx, line in enumerate(rf):
            lj = json.loads(line)
            if lj["label"] == "gpt":
                gpt_idxs.append(idx)
                gpt_texts.append(lj["text"])
    
    list_of_file_texts = list()
    for file in file_list:
        file_texts = list()
        with open(file, "r") as rf:
            for idx, line in enumerate(rf):
                if idx in gpt_idxs:
                    lj = json.loads(line)
                    assert lj["label"] == "gpt"
                    file_texts.append(lj["text"])
        list_of_file_texts.append(file_texts)

    resume_idx = 0
    if os.path.exists(save_file):
        with open(save_file, "r") as rf:
            resume_idx = sum(1 for _ in rf)

    with open(save_file, "a+") as wf:
        for idx, text in tqdm(enumerate(gpt_texts), total=len(gpt_texts)):
            if idx < resume_idx: continue

            input_sents = list()
            input_sents.append(text)
            for file_texts in list_of_file_texts:
                input_sents.append(file_texts[idx])

            gpt_score, gpt_out = get_gpt4_judge_scores(input_sents)
            one = {
                "idx": gpt_idxs[idx],
                "transformations": file_list,
                "gpt_scores": gpt_score,
                "gpt_output": gpt_out,
            }
            wf.write(json.dumps(one, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
