r"""
Swapping words by Mask Language Model
==========================================================
"""

__all__ = ['InsertSpace']

import string
import random

from textflint.generation.transformation import Transformation


class InsertSpace(Transformation):
    r"""
    Transforms an input by replacing its tokens with words of mask language
    predicted.
    To accelerate transformation for long text, input single sentence to
    language model rather than whole text.

    """
    def __init__(
        self,
        insert_min=1,
        insert_max=4,
        insert_prob=0.5,
        seed=42,
        **kwargs
    ):
        r"""
        :param int trans_min: Minimum number of character will be augmented.
        :param int trans_max: Maximum number of character will be augmented.
            If None is passed, number of augmentation is calculated via aup_char_p.
            If calculated result from aug_p is smaller than aug_max, will use
            calculated result from aup_char_p. Otherwise, using aug_max.
        :param float trans_p: Percentage of character (per token) will be
            augmented.
        :param list stop_words: List of words which will be skipped from augment
            operation.
        """
        self.insert_min = insert_min
        self.insert_max = insert_max
        self.insert_prob = insert_prob

        self.rng = random.Random(seed)

    def __repr__(self):
        return 'InsertSpace'

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according field.

        :param Sample sample: input data, normally one data component.
        :param str field: indicate which field to apply transformation
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """
        if self.rng.random() > self.insert_prob:
            return [sample.clone(sample), ]

        insert_cnt = self.rng.randint(self.insert_min, self.insert_max)
        text = sample.get_text(field)
        if insert_cnt > len(text):
            return [sample.clone(sample), ]

        insert_indices = self.rng.sample(list(range(len(text))), insert_cnt)
        trans_res = ""
        for i, cr in enumerate(text):
            trans_res += cr
            if i in insert_indices:
                trans_res += " "

        return [sample.replace_field(field, trans_res), ]


if __name__ == "__main__":
    from textflint.input.component.sample import UTSample
    sample = UTSample({
        "x": "A paragraph is a collection of words strung together to make a longer unit than a sentence. Several sentences often make a paragraph. There are normally three to eight sentences in a paragraph. Paragraphs can start with a five-space indentation or by skipping a line and then starting over. This makes it simpler to tell when one paragraph ends and the next starts simply it has 3-9 lines. A topic phrase appears in most ordered types of writing, such as essays. This paragraph's topic sentence informs the reader about the topic of the paragraph. In most essays, numerous paragraphs make statements to support a thesis statement, which is the essay's fundamental point. Paragraphs may signal when the writer changes topics. Each paragraph may have a number of sentences, depending on the topic.",
        "y": "human",
    })
    trans = InsertSpace(insert_min=2, insert_max=8, insert_prob=1.0)
    res_sample = trans._transform(sample)[0]
    print(res_sample.get_text("x"))
