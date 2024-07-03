r"""
Typos Transformation for add/remove punctuation.
==========================================================

"""

__all__ = ['WordCase']

from textflint.generation.transformation import WordSubstitute


class WordCase(WordSubstitute):
    r"""
    Transformation that simulate typos error to transform sentence.
    """
    def __init__(
        self,
        trans_min=1,
        trans_max=10,
        trans_p=0.1,
        stop_words=None,
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
        """
        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words)

    def __repr__(self):
        return 'WordCase'

    def skip_aug(self, tokens, mask, **kwargs):
        return self.pre_skip_aug(tokens, mask)

    def _get_candidates(self, word, n=1, **kwargs):
        r"""
        Returns a list of words with typo errors.

        :param str word: token word to transform.
        :param int n: number of transformed tokens to generate.
        :param kwargs:
        :return list replaced_tokens: replaced tokens list

        """
        assert n == 1

        candidates = set()

        for i in range(n):
            result = self._get_reversed_word_case(word)
            if result:
                candidates.add(result)

        if len(candidates) > 0:
            return list(candidates)
        else:
            return []

    def _get_reversed_word_case(self, token):
        if len(token) <= 1:
            return token

        if token[0].islower():
            return token[0].upper() + token[1:]
        elif token[0].isupper():
            return token[0].lower() + token[1:]


if __name__ == "__main__":
    from textflint.input.component.sample import UTSample
    sample = UTSample({
        "x": "A paragraph is a collection of words strung together to make a longer unit than a sentence. Several sentences often make a paragraph. There are normally three to eight sentences in a paragraph. Paragraphs can start with a five-space indentation or by skipping a line and then starting over. This makes it simpler to tell when one paragraph ends and the next starts simply it has 3-9 lines. A topic phrase appears in most ordered types of writing, such as essays. This paragraph's topic sentence informs the reader about the topic of the paragraph. In most essays, numerous paragraphs make statements to support a thesis statement, which is the essay's fundamental point. Paragraphs may signal when the writer changes topics. Each paragraph may have a number of sentences, depending on the topic.",
        "y": "human",
    })
    trans = WordCase(trans_p=0.1)
    res_sample = trans._transform(sample)[0]
    print(res_sample.get_text("x"))
