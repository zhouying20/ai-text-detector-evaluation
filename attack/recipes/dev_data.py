from attack.transformation.paragraph.paraphrase import Paraphrase
from attack.transformation.paragraph.back_trans import BackTrans as ParagraphBackTrans
from attack.transformation.sentence.swap_generation import MLMSentence
from attack.transformation.sentence.back_trans import BackTransSentence
from attack.transformation.word.swap_mlm import MLMSuggestion
from attack.transformation.word.insert_adv import InsertAdv
from attack.transformation.word.error_spell import SpellingError
from attack.transformation.word.error_typo import Typos
from attack.transformation.char.word_case import WordCase
from attack.transformation.char.space_removal import SpeaceRemoval
from attack.transformation.char.insert_space import InsertSpace
from attack.transformation.char.punctuation_removal import PunctuationRemoval


# transformation methods
trans_methods = {
    "Paraphrase_l40_o40_no_prefix": {
        "func": Paraphrase,
        "args": {
            "lex_diversity": 40,
            "order_diversity": 40,
            "use_prefix": False,
        }
    },
    "BackTrans_Helsinki_r3": {
        "func": ParagraphBackTrans,
        "args": {
            "trans_iter": 3,
            "from_model_name": "Helsinki-NLP/opus-mt-en-fr",
            "to_model_name": "Helsinki-NLP/opus-mt-fr-en",
        }
    },
    "MLMSentence_large_trans_2_5": {
        "func": MLMSentence,
        "args": {
            "trans_min": 2,
            "trans_max": 5,
            "model_name_or_path": "facebook/bart-large",
        }
    },
    "BackTransSentence_Helsinki_s3_w5": {
        "func": BackTransSentence,
        "args": {
            "trans_sent": 3,
            "trans_sent_window": 5,
            "from_model_name": "Helsinki-NLP/opus-mt-en-fr",
            "to_model_name": "Helsinki-NLP/opus-mt-fr-en",
        }
    },
    "MLMSuggestion_bert_pct20": {
        "func": MLMSuggestion,
        "args": {
            "masked_model": "bert-base-uncased",
            "trans_min": 10,
            "trans_max": None,
            "max_sent_size": 512,
            "trans_p": 0.2,
        },
    },
    "InsertAdv_pct50": {
        "func": InsertAdv,
        "args": {
            "trans_p": 0.5
        },
    },
    "WordMerge_3_10": {
        "func": SpeaceRemoval,
        "args": {
            "removal_min": 3,
            "removal_max": 10,
            "removal_prob": 1.0,
        },
    },
    "WordCase_pct20": {
        "func": WordCase,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.2
        },
    },
    "SpellingError_pct20": {
        "func": SpellingError,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.2,
        },
    },
    "Typos_pct20": {
        "func": Typos,
        "args": {
            "trans_min": 10,
            "trans_max": None,
            "trans_p": 0.2
        },
    },
    "InsertSpace_5_10_prob100": {
        "func": InsertSpace,
        "args": {
            "insert_min": 5,
            "insert_max": 10,
            "insert_prob": 1.0,
        }
    },
    "PunctuationRemoval_p30": {
        "func": PunctuationRemoval,
        "args": {
            "removal_min": 1,
            "removal_max": 10,
            "removal_p": 0.3,
        }
    },
}