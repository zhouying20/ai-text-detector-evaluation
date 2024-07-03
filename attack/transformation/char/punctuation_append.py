r"""
Add punctuation at the beginning and end of a sentence
==========================================================
"""

__all__ = ['PunctuationAppend']

import string

from textflint.generation.transformation import Transformation


class PunctuationAppend(Transformation):
    r"""
    Transforms input by add punctuation at the end of sentence.

    """
    def __init__(
        self,
        add_bracket=True,
        **kwargs
    ):
        r"""
        :param bool add_bracket: whether add punctuation like bracket at the
            beginning and end of sentence.

        """
        super().__init__()
        self.add_bracket = add_bracket

    def __repr__(self):
        return 'PunctuationAppend'

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according transform_field.

        :param ~Sample sample: input data, normally one data component.
        :param str field: indicate which field to transform.
        :param int n: number of generated samples
        :return list trans_samples: transformed sample list.

        """
        trans_samples = []
        tokens = sample.get_words(field)
        # remove origin punctuation at the end of sentence
        pun_indices = self._get_punctuation_scope(tokens)
        if pun_indices:
            sample = sample.delete_field_at_indices(field, pun_indices)
            tokens = sample.get_words(field)

        for beginning_pun, end_pun in self._generate_punctuation(n):
            trans_sample = sample.insert_field_after_index(
                field, len(tokens) - 1, end_pun)
            trans_sample = trans_sample.insert_field_before_index(
                field, 0, beginning_pun)
            trans_samples.append(trans_sample)

        return trans_samples

    @staticmethod
    def _get_punctuation_scope(tokens):
        r"""
        Get indices of punctuations at the end of tokens

        :param list tokens: word list
        :return list indices: indices list

        """
        indices = []
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] in string.punctuation:
                indices.append(i)
            else:
                break

        indices.reverse()

        return indices

    def _generate_punctuation(self, n):
        r"""
        Generate punctuation.

        :param int n:
        :return list: insert punctuations

        """
        bracket_puns = [
            ['"', '"'],
            ['{', '}'],
            ['[', ']'],
            ['(', ')'],
            ['`', '`'],
            ['【', '】']
        ]
        end_puns = ['...', '.', ',', ';']
        number = min(n, len(end_puns))

        random_brackets = self.sample_num(bracket_puns, number)
        random_puns = self.sample_num(end_puns, number)

        for i in range(number):
            if self.add_bracket:
                yield [random_brackets[i][0]], [random_puns[i],
                                                random_brackets[i][-1]]
            else:
                yield [], [random_puns[i]]


if __name__ == "__main__":
    from textflint.input.component.sample import UTSample
    sample = UTSample({
        "x": "A paragraph is a collection of words strung together to make a longer unit than a sentence. Several sentences often make a paragraph. There are normally three to eight sentences in a paragraph. Paragraphs can start with a five-space indentation or by skipping a line and then starting over. This makes it simpler to tell when one paragraph ends and the next starts simply it has 3-9 lines. A topic phrase appears in most ordered types of writing, such as essays. This paragraph's topic sentence informs the reader about the topic of the paragraph. In most essays, numerous paragraphs make statements to support a thesis statement, which is the essay's fundamental point. Paragraphs may signal when the writer changes topics. Each paragraph may have a number of sentences, depending on the topic.",
        "y": "human",
    })
    trans = PunctuationAppend(add_bracket=True)
    res_sample = trans._transform(sample)[0]
    print(res_sample.get_text("x"))
