r"""
Add adverb word before verb word with given pos tags
==========================================================
"""

__all__ = ["InsertAdv"]

import random

from textflint.common.settings import ADVERB_PATH
from textflint.common.utils.load import plain_lines_loader
from textflint.common.utils.list_op import trade_off_sub_words
from textflint.common.utils.install import download_if_needed
from textflint.generation.transformation import Transformation


class InsertAdv(Transformation):
    r"""
    Transforms an input by add adverb word before verb.

    """
    def __init__(
        self,
        trans_p=0.5,
        seed=42,
        **kwargs
    ):
        super().__init__()
        self.rng = random.Random(seed)
        self.trans_p = trans_p
        self.adverb_list = plain_lines_loader(download_if_needed(ADVERB_PATH))

    def __repr__(self):
        return 'InsertAdv'

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according transform_field.

        :param ~Sample sample: input data, normally one data component.
        :param str field: indicate which field to transform
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """
        pos_tags = sample.get_pos(field)
        _insert_indices = self._get_verb_location(pos_tags)

        if not _insert_indices:
            return [sample.clone(sample), ]

        sample_cnt = int(len(_insert_indices) * self.trans_p)
        sample_cnt = max(1, sample_cnt)
        _insert_indices = self.rng.sample(_insert_indices, sample_cnt)

        insert_words = []
        insert_indices = []

        for index in _insert_indices:
            _insert_words = self._get_random_adverbs(n=n)

            if _insert_words:
                insert_indices.append(index)
                insert_words.append(_insert_words)

        if not insert_words:
            return [sample.clone(sample), ]

        insert_words, insert_indices = trade_off_sub_words(
            insert_words, insert_indices, n=n)
        trans_samples = []

        # get substitute candidates combinations
        for i in range(len(insert_words)):
            single_insert_words = insert_words[i]
            trans_samples.append(
                sample.insert_field_before_indices(
                    field, insert_indices, single_insert_words))

        return trans_samples

    @staticmethod
    def _get_verb_location(pos_tags):
        verb_location = []

        for i, pos in enumerate(pos_tags):
            if pos in ['VB', 'VBP', 'VBZ', 'VBG', 'VBD', 'VBN']:
                verb_location.append(i)

        return verb_location

    def _get_random_adverbs(self, n):
        sample_num = min(n, len(self.adverb_list))

        return random.sample(self.adverb_list, sample_num)


if __name__ == "__main__":
    from textflint.input.component.sample import UTSample
    sample = UTSample({
        "x": "A paragraph is a collection of words strung together to make a longer unit than a sentence. Several sentences often make a paragraph. There are normally three to eight sentences in a paragraph. Paragraphs can start with a five-space indentation or by skipping a line and then starting over. This makes it simpler to tell when one paragraph ends and the next starts simply it has 3-9 lines. A topic phrase appears in most ordered types of writing, such as essays. This paragraph's topic sentence informs the reader about the topic of the paragraph. In most essays, numerous paragraphs make statements to support a thesis statement, which is the essay's fundamental point. Paragraphs may signal when the writer changes topics. Each paragraph may have a number of sentences, depending on the topic.",
        "y": "human",
    })
    trans = InsertAdv(trans_p=0.2)
    res_sample = trans._transform(sample)[0]
    print(res_sample.get_text("x"))
