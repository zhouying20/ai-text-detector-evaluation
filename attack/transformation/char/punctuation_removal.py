r"""
Add punctuation at the beginning and end of a sentence
==========================================================
"""

__all__ = ['PunctuationRemoval']

import string
import random

from textflint.generation.transformation import Transformation


class PunctuationRemoval(Transformation):
    r"""
    Transforms input by add punctuation at the end of sentence.

    """
    def __init__(
        self,
        removal_min=1,
        removal_max=10,
        removal_p=0.5,
        seed=42,
        only_last=False,
        **kwargs
    ):
        r"""
        :param bool add_bracket: whether add punctuation like bracket at the
            beginning and end of sentence.

        """
        self.removal_min = removal_min
        self.removal_max = removal_max
        self.removal_p = removal_p
        self.only_last = only_last
        self.rng = random.Random(seed)
        super().__init__()

    def __repr__(self):
        return 'PunctuationRemoval'

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according transform_field.

        :param ~Sample sample: input data, normally one data component.
        :param str field: indicate which field to transform.
        :param int n: number of generated samples
        :return list trans_samples: transformed sample list.
        """
        tokens = sample.get_words(field)
        pun_indices = self._get_punctuation_indicies(tokens)

        if not pun_indices:
            return [sample.clone(sample), ]

        if not self.only_last:
            removal_cnt = self.get_removal_cnt(len(pun_indices))
            pun_indices = self.rng.sample(pun_indices, removal_cnt)

        return [sample.delete_field_at_indices(field, pun_indices), ]

    def get_removal_cnt(self, size):
        r"""
        Get the num of words/chars transformation.

        :param int size: the size of target sentence
        :return int: number of words to apply transformation.
        """

        cnt = int(self.removal_p * size)

        if cnt < self.removal_min:
            return self.removal_min
        if self.removal_max is not None and cnt > self.removal_max:
            return self.removal_max

        return cnt

    def _get_punctuation_indicies(self, tokens):
        r"""
        Get indices of punctuations at the end of tokens

        :param list tokens: word list
        :return list indices: indices list

        """
        indices = []
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] in string.punctuation:
                indices.append(i)
            elif self.only_last:
                break

        indices.reverse()
        return indices


if __name__ == "__main__":
    from textflint.input.component.sample import UTSample
    sample = UTSample({
        "x": "A paragraph is a collection of words strung together to make a longer unit than a sentence. Several sentences often make a paragraph. There are normally three to eight sentences in a paragraph. Paragraphs can start with a five-space indentation or by skipping a line and then starting over. This makes it simpler to tell when one paragraph ends and the next starts simply it has 3-9 lines. A topic phrase appears in most ordered types of writing, such as essays. This paragraph's topic sentence informs the reader about the topic of the paragraph. In most essays, numerous paragraphs make statements to support a thesis statement, which is the essay's fundamental point. Paragraphs may signal when the writer changes topics. Each paragraph may have a number of sentences, depending on the topic.",
        "y": "human",
    })
    trans = PunctuationRemoval(removal_min=3, removal_max=3, only_last=False)
    res_sample = trans._transform(sample)[0]
    print(res_sample.get_text("x"))
