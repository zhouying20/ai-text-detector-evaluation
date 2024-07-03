r"""
Back translation class
==========================================================
"""

__all__ = ['MLMSentence']

import torch
import random
import string
import difflib
import logging

from nltk.tokenize import sent_tokenize
from textflint.common import device as default_device
from textflint.generation.transformation import Transformation


logger = logging.getLogger()


class MLMSentence(Transformation):
    r"""
    Back Translation with hugging-face translation models.
    A sentence can only be transformed into one sentence at most.

    """
    def __init__(
        self,
        trans_min=1,
        trans_max=1,
        max_gen_len=64,
        max_retry_times=20,
        model_name_or_path="facebook/bart-base",
        seed=42,
        device=None,
        **kwargs
    ):
        r"""
        :param str from_model_name: model to translate original language to
            target language
        :param str to_model_name: model to translate target language to
            original language
        :param device: indicate utilize cpu or which gpu device to
            run neural network
        """
        super().__init__()
        self.trans_min = trans_min
        self.trans_max = trans_max
        self.max_retry_times = max_retry_times
        self.max_gen_len = max_gen_len

        self.rng = random.Random(seed)
        self.device = self.get_device(device) if device else default_device
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.tokenizer = None

        self.prefix_list = ['We know that', 'We can say that', 'It is about that', 'It is known that', 'The fact is that', 'The answer is that'] # TODO, more prefixes?

    def __repr__(self):
        return 'MLMSentence'

    @staticmethod
    def get_device(device):
        r"""
        Get gpu or cpu device.

        :param str device: device string
                           "cpu" means use cpu device.
                           "cuda:0" means use gpu device which index is 0.
        :return: device in torch.
        """
        if "cuda" not in device:
            return torch.device("cpu")
        else:
            return torch.device(device)

    def get_model(self):
        # accelerate load efficiency
        from transformers import BartForConditionalGeneration, BartTokenizer

        """ Load models of translation. """
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name_or_path)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)

    def is_valid_mask_generation(self, span, left_part, right_part, max_match_radio=0.7):
        _span = span.lower()
        _left = left_part.lower()
        _right = right_part.lower()

        if len(_span) < 1 or _span in string.punctuation:
            return False
        if _left in _span: ## assert not copy left
            return False
        # if len(_span.split()) > self.max_gen_len: ## the max word num
        #     return False

        match_radio_left = difflib.SequenceMatcher(None, _span, _left).find_longest_match(0, len(_span), 0, len(_left)).size / len(_span)
        match_radio_right = difflib.SequenceMatcher(None, _span, _right).find_longest_match(0, len(_span), 0, len(_right)).size / len(_span)

        if match_radio_left > max_match_radio or match_radio_right > max_match_radio:
            return False
        return True

    def _bart_mask_generate_v1(self, contexts, prefixes, left_parts, right_parts, n=1):
        fillings = {i: set() for i in range(len(contexts))}
        gen_index = [i for i, ft in fillings.items() if len(ft) < n]

        for _ in range(self.max_retry_times):
            if len(gen_index) <= 0:
                break

            gen_contexts = [contexts[j] for j in gen_index]
            with torch.no_grad():
                batch = self.tokenizer(gen_contexts, padding=True, truncation=True, return_tensors="pt")
                batch = batch.to(self.device)
                gen_ids = self.model.generate(
                    **batch,
                    max_length=1024,
                    do_sample=True,
                    num_beams=5,
                    num_return_sequences=3,
                )
                gen_texts = self.tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )

            assert len(gen_texts) == 3 * len(gen_index)
            for k, gen_idx in enumerate(gen_index):
                gen_text = gen_texts[k*3: k*3+3]
                prefix = prefixes[gen_idx]
                left_part = left_parts[gen_idx]
                right_part = right_parts[gen_idx]

                for gen_body in gen_text:
                    gen_body_sents = [str(s).strip() for s in sent_tokenize(gen_body)]
                    for sent in gen_body_sents:
                        if not sent.startswith(prefix) or sent in right_part:
                            continue
                        # span = sent.removeprefix(prefix).strip()
                        span = sent.strip()
                        if not self.is_valid_mask_generation(span, left_part, right_part):
                            continue
                        if span[0].islower():
                            span = span[0].upper() + span[1:]
                        if len(fillings[gen_idx]) < n:
                            fillings[gen_idx].add(span)

            gen_index = [i for i, ft in fillings.items() if len(ft) < n]

        # for idx in fillings.keys():
        #     if len(fillings[idx]) <= 0:
        #         print(f'Note: there is no valid mask-filling generated for {contexts[idx]}.')
        #     elif len(fillings[idx]) < n:
        #         print(f'Note: there is only {len(fillings[idx])} / {n} fillings for {contexts[idx]}.')

        return {i: list(fts) for i, fts in fillings.items()}


    def _bart_mask_generate(self, text, prefix, left_part, right_part, n=1):
        res_texts = set()
        for _ in range(self.max_retry_times):
            if len(res_texts) >= n:
                break

            with torch.no_grad():
                batch = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                batch = batch.to(self.device)
                gen_ids = self.model.generate(
                    **batch,
                    max_length=1024,
                    do_sample=True,
                    num_beams=5,
                    num_return_sequences=3,
                )
                gen_texts = self.tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )

            assert len(gen_texts) == 3
            for gen_body in gen_texts:
                res_texts.add(gen_body.strip())
                # gen_body_sents = [str(s).strip() for s in sent_tokenize(gen_body)]
                # for sent in gen_body_sents:
                #     if not sent.startswith(prefix) or sent in right_part:
                #         continue
                #     span = sent.strip()
                #     if self.is_valid_mask_generation(span, left_part, right_part):
                #         res_texts.add(gen_body)
                #         break

        return list(res_texts)

    def _get_masked_bart_text(self, first_part, mask_sent, second_part):
        # prefix = self.rng.choice(self.prefix_list)
        prefix = " ".join(mask_sent.split()[:3])
        content = "{} {} <mask> {}".format(first_part, prefix, second_part)
        return content, prefix

    def _transform_v1(self, sample, n=1, field='x', **kwargs):
        assert n == 1
        if self.model is None:
            self.get_model()

        sents = sample.get_sentences(field)
        trans_max = min(self.trans_max, len(sents))
        trans_cnt = self.rng.randint(self.trans_min, trans_max)

        trans_idxs = self.rng.sample(list(range(len(sents))), trans_cnt)
        trans_ctxs = list()
        trans_prefixes = list()
        trans_lefts = list()
        trans_rights = list()
        for idx in trans_idxs:
            left = " ".join(sents[:idx])
            right = " ".join(sents[idx+1:])
            context, prefix = self._get_masked_bart_text(left, sents[idx], right)
            trans_ctxs.append(context)
            trans_prefixes.append(prefix)
            trans_lefts.append(left)
            trans_rights.append(right)

        gened_texts = self._bart_mask_generate(trans_ctxs, trans_prefixes, trans_lefts, trans_rights, n=n)
        gened_res = list()
        for i in range(len(sents)):
            if i in trans_idxs and len(gened_texts[trans_idxs.index(i)]) > 0:
                gened_res.append(gened_texts[trans_idxs.index(i)][0])
            else:
                gened_res.append(sents[i])

        return [sample.replace_field(field, " ".join(gened_res)), ]

    def _transform(self, sample, n=1, field='x', **kwargs):
        assert n == 1
        if self.model is None:
            self.get_model()

        sents = sample.get_sentences(field)
        trans_max = min(self.trans_max, len(sents))
        trans_min = min(self.trans_min, trans_max)
        trans_cnt = self.rng.randint(trans_min, trans_max)

        trans_sample = sample.clone(sample)
        for _ in range(trans_cnt):
            trans_sents = trans_sample.get_sentences(field)
            idx = self.rng.randint(0, len(trans_sents)-1)

            left = " ".join(trans_sents[:idx])
            right = " ".join(trans_sents[idx+1:])
            context, prefix = self._get_masked_bart_text(left, trans_sents[idx], right)

            gened_texts = self._bart_mask_generate(context, prefix, left, right, n=n)
            if len(gened_texts) < n:
                logger.debug(f"generated {len(gened_texts)} <<< {n}, return origin text...")

            trans_sample = trans_sample.replace_field(field, gened_texts[0])

        return [trans_sample, ]


if __name__ == "__main__":
    from textflint.input.component.sample import UTSample
    sample = UTSample({
        "x": "I am thinking of having a monthly subscription that will give a user x amount of tokens a month and will refill to the same amount every month if they run out of tokens then they can be overcharged for what they use. Being that getting an API on this hub is more difficult due to having the API providers having produce higher quality APIs I would like to pay the APIs that are listed on the hub using the tokens that users do not use giving all the API a percentage of the unused tokens value that gets divided between everyone. I feel as this is a good way to fix the problems that Rapid API has I would love to hear the communities ideas on this and hear what you would be happy to pay for something like this and what things you don’t like or think would be a problem.",
        "y": "human",
    })
    mlm = MLMSentence(2, 5, model_name_or_path="facebook/bart-large",)
    res_sample = mlm._transform(sample)[0]
    print(res_sample.get_text("x"))
