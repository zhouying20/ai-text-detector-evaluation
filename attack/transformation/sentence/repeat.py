r"""
Back translation class
==========================================================
"""

__all__ = ['RepeatSentence']

import random

from textflint.generation.transformation import Transformation


class RepeatSentence(Transformation):
    r"""
    Back Translation with hugging-face translation models.
    A sentence can only be transformed into one sentence at most.

    """
    def __init__(
        self,
        min_repeat=2,
        max_repeat=3,
        min_sent_len=50,
        max_sent_len=300,
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
        self.rng = random.Random(seed)
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat
        self.min_sent_len = min_sent_len
        self.max_sent_len = max_sent_len

    def __repr__(self):
        return 'RepeatSentence'

    def _transform(self, sample, n=1, field='x', **kwargs):
        sents = sample.get_sentences(field)
        repeat_cnt = self.rng.randint(self.min_repeat, self.max_repeat)
        repeat_cnt = min(len(sents), repeat_cnt)

        try_time = 0
        valid_idxs = set()
        while len(valid_idxs) < repeat_cnt and try_time < 10:
            try_time += 1
            rnd_idxs = self.rng.sample(range(len(sents)), repeat_cnt)
            for _i in rnd_idxs:
                if self.min_sent_len <= len(sents[_i]) <= self.max_sent_len:
                    valid_idxs.add(_i)

        if len(valid_idxs) == 0: # no valid sentence to be repeated
            return [sample.clone(sample), ]
        elif len(valid_idxs) > repeat_cnt: # too much, need to sample -> repeat_cnt
            valid_idxs = set(self.rng.sample(list(valid_idxs), repeat_cnt))

        repeated_sents = list()
        for i in range(len(sents)):
            repeated_sents.append(sents[i])
            if i in valid_idxs:
                repeated_sents.append(sents[i])
        return [sample.replace_field(field, " ".join(repeated_sents)), ]
