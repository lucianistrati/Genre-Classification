from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
# from src.text.multilang.utils import load_spacy_pipeline
from wordsegment import load as load_wordsegment
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from statistics import mean
from wordfreq import word_frequency

import numpy as np

import textstat
import inflect
import string
import random
import spacy
import pdb

# TODO SET THIS AS GOLD STANDARD FOR LINGUISTICAL FEATURE EXTRACTOR ACROSS ALL REPOS
# TODO replace all uses of load_nlp with from src.text.multilang.utils import load_spacy_pipeline
money_symbols = ["$", "£", "€", "lei", "RON", "USD", "EURO", "dolari", "lire", "yeni"]
roman_numerals = "XLVDCMI"

inflect = inflect.engine()

load_wordsegment()

textstat.set_lang("en")

PAD_TOKEN = "__PAD__"

numpy_arrays_path = "data/numpy_data"

encoder_dict = dict()
encoder_cnt = 0

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
GOOD = 0
ERRORS = 0

LEFT_LEFT_TOKEN = -4
LEFT_TOKEN = -3
RIGHT_TOKEN = -1
RIGHT_RIGHT_TOKEN = -2
ALL_LINGUISTICAL_FEATURES_NAMES = ["get_sws_pct", "count_sws", "get_dots_pct", "get_dash_pct", "get_len",
                                   "get_digits_pct", "get_punctuation_pct", "get_phrase_len", "get_spaces_pct",
                                   "get_capital_letters_pct", "get_slashes_pct", "index", "get_roman_numerals_pct",
                                   "get_stopwords_pct", "ner_tags", "dep_tags", "pos_tags", "get_word_frequency",
                                   "count_punctuations", "count_capital_words", "has_money_tag",
                                   "starts_with_capital_letter", "get_phrase_num_tokens", "get_word_position_in_phrase",
                                   "get_num_pos_tags", "count_letters"]

# ALL_LANGUAGES = set(list(np.load(file="data/multilang_utils/massive_all_4749_languages.npy", allow_pickle=True)))


def load_all_stopwords(lang: str, nlp = None):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    all_stopwords = set(list(nlp.Defaults.stop_words))
    return all_stopwords


def load_nlp(lang: str):
    return spacy.load("en_core_web_sm")


def is_there_a_language(doc: spacy.tokens.doc.Doc):
    """
    :return: if the language is supported
    """
    for lang in ALL_LANGUAGES:
        if lang in doc.text.lower():
            return float(True)
    return float(False)


def get_stopwords_pct(doc: spacy.tokens.doc.Doc, all_stopwords: List[str]):
    """
    :return: the percentage of stopwords in the sentence
    """
    characters = set(list(doc.text))
    return len(characters.intersection(all_stopwords)) / len(characters)


def count_letters(doc: spacy.tokens.doc.Doc = None):
    """

    :param text:
    :return:
    """  # TODO look how to check other letters
    return len([t for t in doc.text if t in string.ascii_uppercase or t in string.ascii_lowercase])


def get_phrase_len(doc: spacy.tokens.doc.Doc):
    """

    :param phrase:
    :return:
    """
    return len(doc)


def get_num_pos_tags(doc: spacy.tokens.doc.Doc = None):
    """
    :return: get the number of unique part-of-speech"s that appear in a sentence
    """
    pos_tags = set([tok.pos_ for tok in doc])
    return len(set(pos_tags)) / len(pos_tags)


def get_word_position_in_phrase(doc: spacy.tokens.doc.Doc, start_offset: int):
    """

    :param phrase:
    :param start_offset:
    :return:
    """
    return start_offset / len(doc.text)


def get_phrase_num_tokens(doc: spacy.tokens.doc.Doc):
    """
    :return: number of words in a text
    """
    return len(doc)


def has_money_tag(doc: spacy.tokens.doc.Doc):
    """
    :return: True if there is any money tag in the sentece
    """
    global money_symbols
    for sym in money_symbols:
        if sym.lower() in doc.text.lower():
            return float(True)
    return float(False)


def starts_with_capital_letter(doc: spacy.tokens.doc.Doc):
    """
    :return: True if the word is in Capital case, False otherwise
    """
    if doc.text[0] in string.ascii_uppercase:
        return float(True)
    return float(False)


def get_len(doc: spacy.tokens.doc.Doc):
    """
    :return: return the length of the text
    """
    return len(doc.text)


def get_capital_letters_pct(doc: spacy.tokens.doc.Doc):
    """
    :return: get percentage of capital letters in the text
    """
    return len([c for c in doc.text if c in string.ascii_uppercase]) / len(doc.text)


def get_roman_numerals_pct(doc: spacy.tokens.doc.Doc):
    """
    :return: get percentage of numerals in the text
    """
    global roman_numerals
    return len([c for c in doc.text if c in roman_numerals]) / len(doc.text)


def get_digits_pct(doc: spacy.tokens.doc.Doc):
    """
    :return: get percentage of digits characters in the text
    """
    return len([c for c in doc.text if c in string.digits]) / len(doc.text)


def get_punctuation_pct(doc: spacy.tokens.doc.Doc):
    """
    :return: get percentage of punctuation characters in the text
    """
    return len([c for c in doc.text if c in string.punctuation]) / len(doc.text)


def get_dash_pct(doc: spacy.tokens.doc.Doc):
    """
    :return: get percentage of dash characters in the text
    """
    return len([c for c in doc.text if c == "-"]) / len(doc.text)


def get_spaces_pct(doc: spacy.tokens.doc.Doc):
    """
    :return: get percentage of spaces characters in the text
    """
    return len([c for c in doc.text if c == " "]) / len(doc.text)


def get_slashes_pct(doc: spacy.tokens.doc.Doc):
    """
    :return: get percentage of slash characters in the text
    """
    return len([c for c in doc.text if c == "/" or c == "\\"]) / len(doc.text)


def get_text_similarity(text_1: str, text_2: str):
    """
    :return: the percent of semantical similarity between two texts
    """
    pass  # TODO find the easiest way to get an approximate measure of this


def get_dots_pct(doc: spacy.tokens.doc.Doc):
    """
    :return: get percentage of dots characters in the text
    """
    return len([c for c in doc.text if c == "."]) / len(doc.text)


def count_capital_words(doc: spacy.tokens.doc.Doc):
    """
    :return: get the number of capital words that exist in a sentence
    """
    tokens = [token.text for token in doc]
    return sum(map(str.isupper, tokens))


def count_punctuations(doc: spacy.tokens.doc.Doc):
    """
    :return: number of punctuations signs present in the text
    """
    punctuations = """}!"#/$%"(*]+,->.:);=?&@^_`{<|~["""
    res = []
    for i in punctuations:
        res.append(doc.text.count(i))
    if len(res):
        return mean(res)
    return 0.0


def get_word_frequency(doc: spacy.tokens.doc.Doc):
    """
    :param doc:
    :param tokens:
    :return: mean frequency of the words in the sentence
    """
    return mean([word_frequency(token.text, doc.lang_) for token in doc])


def count_sws(doc: spacy.tokens.doc.Doc):
    """

    :param text:
    :param tokens:
    :return:
    """
    tokens = [token.text for token in doc]
    return len([tok for tok in tokens if tok.lower() in stop_words])


def get_sws_pct(doc: spacy.tokens.doc.Doc):
    """

    :param doc:
    :return:
    """
    tokens = [token.text for token in doc]
    return count_sws(doc) / len(tokens)


def pos_tags(doc: spacy.tokens.doc.Doc, index: int, nlp: spacy.lang = None):
    """

    :param doc:
    :param index:
    :param nlp:
    :return:
    """
    global encoder_dict, encoder_cnt
    context_indexes = list(range(max(index - 2, 0), min(index + 2, len(doc))))
    feats = []
    for idx, nlp_token in enumerate(doc):
        if idx in context_indexes:
            feats.append(nlp_token.pos_)

    if index == 1 or index == len(doc) - 2:
        feats.insert(0, "-2_pos")
        feats.append("-3_pos")

    if index == 0 or index == len(doc) - 1:
        feats.insert(0, "-2_pos")
        feats.insert(0, "-1_pos")
        feats.append("-3_pos")
        feats.append("-4_pos")

    pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
                "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]

    num_pos_tags = len(pos_tags)

    feats_dict = {"-1_pos": -2, "-2_pos": -1, "-3_pos": num_pos_tags, "-4_pos": num_pos_tags + 1}
    for i, tag_pos in enumerate(pos_tags):
        feats_dict[tag_pos] = i
    feats = [feats_dict[feat] for feat in feats_dict]
    res = sum([i * feat for i, feat in enumerate(feats)])

    return float(res)


def dep_tags(doc: spacy.tokens.doc.Doc, index: int, nlp: spacy.lang = None):
    """

    :param doc:
    :param index:
    :param nlp:
    :return:
    """
    global encoder_dict, encoder_cnt
    context_indexes = list(range(max(index - 2, 0), min(index + 2, len(doc))))
    feats = []
    for idx, nlp_token in enumerate(doc):
        if idx in context_indexes:
            feats.append(nlp_token.dep_)
    if index == 1 or index == len(doc) - 2:
        feats.insert(0, "-2_dep")
        feats.append("-3_dep")
    if index == 0 or index == len(doc) - 1:
        feats.insert(0, "-2_dep")
        feats.insert(0, "-1_dep")
        feats.append("-3_dep")
        feats.append("-4_dep")

    dep_tags = ["acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc",
                "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl",
                "intj", "mark", "meta", "neg", "nn", "npmod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp",
                "pobj", "poss", "preconj", "prep", "prt", "punct", "quantmod", "relcl", "root", "xcomp"]
    # dep_tags = [label for label in nlp.get_pipe("tagger").labels]
    num_dep_tags = len(dep_tags)
    feats_dict = {"-1_dep": -2, "-2_dep": -1, "-3_dep": num_dep_tags, "-4_dep": num_dep_tags + 1}
    for i, dep_tag in enumerate(dep_tags):
        feats_dict[dep_tag] = i
    feats = [feats_dict[feat] for feat in feats_dict]
    res = sum([i * feat for i, feat in enumerate(feats)])
    return float(res)


def ner_tags(doc: spacy.tokens.doc.Doc, index: int, nlp: spacy.lang = None):
    """
    :param doc:
    :param index:
    :return:
    """
    global encoder_dict, encoder_cnt
    context_indexes = list(range(max(index - 2, 0), min(index + 2, len(doc))))
    context_tokens = [tok.text for i, tok in enumerate(doc) if i in context_indexes]
    feats = []
    for nlp_token in doc.ents:
        if nlp_token.text in context_tokens:
            feats.append(nlp_token.label_)

    if index == 1 or index == len(doc) - 2:
        feats.insert(0, "-2_ner")
        feats.append("-3_ner")

    if index == 0 or index == len(doc) - 1:
        feats.insert(0, "-2_ner")
        feats.insert(0, "-1_ner")
        feats.append("-3_ner")
        feats.append("-4_ner")

    # ner_tags = [label for label in nlp.get_pipe("tagger").labels]
    ner_tags = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
                "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
    num_ner_tags = len(ner_tags)

    feats_dict = {"-1_ner": -2, "-2_ner": -1, "-3_ner": num_ner_tags, "-4_ner": num_ner_tags + 1}
    for i, ner_tag in enumerate(ner_tags):
        feats_dict[ner_tag] = i
    feats = [feats_dict[feat] for feat in feats_dict]
    res = sum([i * feat for i, feat in enumerate(feats)])
    return (res)


def get_paper_features(text: str, lang: str = "en", doc: spacy.tokens.doc.Doc = None, nlp: spacy.lang = None,
                       index: int = 0, requested_features_names: List[str] = None):
    """

    :param text:
    :param lang:
    :param doc:
    :param nlp:
    :param index:
    :param requested_features_names:
    :return:
    """
    if doc is None:
        # if nlp is None:
        #     pdb.set_trace()
        doc = nlp(text)

    all_stopwords = load_all_stopwords(lang=lang, nlp=nlp)

    FIRST_N_FEATURES = 23

    if requested_features_names is None:
        requested_features_names = ALL_LINGUISTICAL_FEATURES_NAMES[:FIRST_N_FEATURES]

    linguistical_features = []

    for feature_name in requested_features_names:
        match feature_name:
            case 'get_sws_pct':
                linguistical_features.append(get_sws_pct(doc))
            case 'count_sws':
                linguistical_features.append(count_sws(doc))
            case 'get_dots_pct':
                linguistical_features.append(get_dots_pct(doc))
            case 'get_dash_pct':
                linguistical_features.append(get_dash_pct(doc))
            case 'get_len':
                linguistical_features.append(get_len(doc))
            case 'get_digits_pct':
                linguistical_features.append(get_digits_pct(doc))
            case 'get_punctuation_pct':
                linguistical_features.append(get_punctuation_pct(doc))
            case 'get_phrase_len':
                linguistical_features.append(get_phrase_len(doc))
            case 'get_spaces_pct':
                linguistical_features.append(get_spaces_pct(doc))
            case 'get_capital_letters_pct':
                linguistical_features.append(get_capital_letters_pct(doc))
            case 'get_slashes_pct':
                linguistical_features.append(get_slashes_pct(doc))
            case 'index':
                linguistical_features.append(index)
            case 'get_roman_numerals_pct':
                linguistical_features.append(get_roman_numerals_pct(doc))
            case 'get_stopwords_pct':
                linguistical_features.append(get_stopwords_pct(doc, all_stopwords))
            case 'ner_tags':
                linguistical_features.append(ner_tags(doc, index, nlp))  # str
            case 'dep_tags':
                linguistical_features.append(dep_tags(doc, index, nlp))  # str
            case 'pos_tags':
                linguistical_features.append(pos_tags(doc, index, nlp))  # str
            case 'get_word_frequency':
                linguistical_features.append(get_word_frequency(doc))
            case 'count_punctuations':
                linguistical_features.append(count_punctuations(doc))
            case 'count_capital_words':
                linguistical_features.append(count_capital_words(doc))
            case 'has_money_tag':
                linguistical_features.append(has_money_tag(doc))
            case 'starts_with_capital_letter':
                linguistical_features.append(starts_with_capital_letter(doc))
            case 'get_phrase_num_tokens':
                linguistical_features.append(get_phrase_num_tokens(doc))
            case 'get_word_position_in_phrase':
                start_offset = index
                linguistical_features.append(get_word_position_in_phrase(doc, start_offset))
            case 'get_num_pos_tags':
                linguistical_features.append(get_num_pos_tags(doc))
            case 'count_letters':
                linguistical_features.append(count_letters(doc))

    string_feats = pos_tags(doc, index) + dep_tags(doc, index) + ner_tags(doc, index)

    return np.array(linguistical_features), string_feats


def main():
    phrase = "Both China and the Philippines flexed their muscles on Wednesday."
    start_offset = 56 + len("Wednesday")
    end_offset = 56 + len("Wednesday")
    print(phrase[start_offset: end_offset])
    nlp = spacy.load("en_core_web_sm")
    print(get_paper_features(phrase, nlp=nlp))
    for synset in wordnet.synsets("flexed"):
        print(synset)
        print(dir(synset))
        for lemma in synset.lemmas():
            print(lemma.name())


if __name__ == "__main__":
    main()
