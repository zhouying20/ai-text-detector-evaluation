r"""
Paraphrasing class
==========================================================
"""

__all__ = ['ReverseByPivot']

import random

from textflint.generation.transformation import Transformation


class ReverseByPivot(Transformation):
    r"""
    Back Translation with hugging-face translation models.
    A sentence can only be transformed into one sentence at most.

    """
    def __init__(
        self,
        trans_iter=1,
        seed=42,
        **kwargs
    ):
        r"""
        :param int seed: model to translate original language to
            target language

        """
        super().__init__()
        self.trans_iter = trans_iter
        self.rng = random.Random(seed)

    def __repr__(self):
        return 'ReverseByPivot'

    def _transform(self, sample, n=1, field='x', **kwargs):
        trans_sample = sample.clone(sample)

        for _ in range(self.trans_iter):
            sents = trans_sample.get_sentences(field)
            if len(sents) < 2:
                return [trans_sample, ]
            elif len(sents) == 2:
                trans_sample = trans_sample.replace_field(field, f"{sents[1]} {sents[0]}") # reverse
            else:
                pivot_idx = self.rng.randint(0, len(sents)-1)
                reversed_sents = sents[pivot_idx+1:] + [sents[pivot_idx],] + sents[:pivot_idx]
                trans_sample = trans_sample.replace_field(field, " ".join(reversed_sents))

        return [trans_sample, ]
