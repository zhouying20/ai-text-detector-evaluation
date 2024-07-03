r"""
Typos Transformation for add/remove punctuation.
==========================================================

"""

__all__ = ['Typos']

from textflint.generation.transformation import WordSubstitute
from textflint.common.utils.word_op import *


class Typos(WordSubstitute):
    r"""
    Transformation that simulate typos error to transform sentence.

    https://arxiv.org/pdf/1711.02173.pdf

    """
    def __init__(
        self,
        trans_min=1,
        trans_max=10,
        trans_p=0.3,
        stop_words=None,
        mode="random",
        skip_first_char=True,
        skip_last_char=True,
        **kwargs
    ):
        r"""
        :param int trans_min: Minimum number of character will be augmented.
        :param int trans_max: Maximum number of character will be augmented.
            If None is passed, number of augmentation is calculated via
            aup_char_p.If calculated result from aug_p is smaller than aug_max,
            will use calculated result from aup_char_p. Otherwise, using
            aug_max.
        :param float trans_p: Percentage of character (per token) will be
            augmented.
        :param list stop_words: List of words which will be skipped from augment
            operation.
        :param str mode: just support ['random', 'replace', 'swap', 'insert',
            'delete'].
        :param bool skip_first_char: whether skip the first char of target word.
        :param bool skip_last_char: whether skip the last char of target word.

        """
        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words)
        self._mode = mode
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char

    def __repr__(self):
        return 'Typos' + '_' + self._mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode_value):
        assert mode_value in ['random', 'replace', 'swap', 'insert', 'delete']
        self._mode = mode_value

    def skip_aug(self, tokens, mask, **kwargs):
        return self.pre_skip_aug(tokens, mask)

    def _get_candidates(self, word, n=5, **kwargs):
        r"""
        Returns a list of words with typo errors.

        :param str word: token word to transform.
        :param int n: number of transformed tokens to generate.
        :param kwargs:
        :return list replaced_tokens: replaced tokens list

        """
        candidates = set()

        for i in range(n):
            typo_method = self._get_typo_method()
            # default operate at most one character in a word
            result = typo_method(
                word, 1, self.skip_first_char, self.skip_last_char)
            if result:
                candidates.add(result)

        if len(candidates) > 0:
            return list(candidates)
        else:
            return []

    def _get_typo_method(self):
        if self._mode == 'replace':
            return replace
        elif self._mode == 'swap':
            return swap
        elif self._mode == 'insert':
            return insert
        elif self._mode == 'delete':
            return delete
        else:
            return random.choice([replace, swap, insert, delete])


if __name__ == "__main__":
    from textflint.input.component.sample import UTSample
    sample = UTSample({
        "x": "A paragraph is a collection of words strung together to make a longer unit than a sentence. Several sentences often make a paragraph. There are normally three to eight sentences in a paragraph. Paragraphs can start with a five-space indentation or by skipping a line and then starting over. This makes it simpler to tell when one paragraph ends and the next starts simply it has 3-9 lines. A topic phrase appears in most ordered types of writing, such as essays. This paragraph's topic sentence informs the reader about the topic of the paragraph. In most essays, numerous paragraphs make statements to support a thesis statement, which is the essay's fundamental point. Paragraphs may signal when the writer changes topics. Each paragraph may have a number of sentences, depending on the topic.",
        "y": "human",
    })
    trans = Typos(trans_min=1, trans_max=64, trans_p=0.1, mode="random",)
    res_sample = trans._transform(sample)[0]
    print(res_sample.get_text("x"))
