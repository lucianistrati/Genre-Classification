from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from langdetect import detect
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.svm import SVC
from copy import deepcopy
from typing import List
from tqdm import tqdm

import seaborn as sns
import tensorflow as tf
import pandas as pd
import xgboost as xgb
import numpy as np

import pyphen
import pronouncing
import nltk
import pdb
import os
import glob
import requests
import uuid
import json
import re
import random
import itertools
import math
import textract
import spacy
import nltk
import docx
import os
import re


def load_data():
    train_df = pd.read_csv("data/Lyrics-Genre-Train.csv")
    test_df = pd.read_csv("data/Lyrics-Genre-Test-GroundTruth.csv")

    return train_df, test_df


from gold_standard_linguistical_feature_extraction import get_paper_features

nlp = spacy.load("en_core_web_sm")


def read_docx(filepath: str):
    """

    :param filepath:
    :return:
    """
    doc = docx.Document(filepath)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText).strip()


def read_txt(filepath: str):
    """

    :param filepath:
    :return:
    """
    with open(filepath, "r") as f:
        content = f.read()
    return content.strip()


def read_pdf(filepath: str):
    """

    :param filepath:
    :return:
    """
    text = textract.process(filepath, method='pdfminer')
    return text.decode('utf-8').strip()


def read_file_content(filepath: str):
    """

    :param filepath:
    :return:
    """
    if filepath.endswith('.txt'):
        return read_txt(filepath)
    if filepath.endswith('.pdf'):
        return read_pdf(filepath)
    if filepath.endswith('.docx'):
        return read_docx(filepath)
    else:
        raise Exception("Wrong extension for the file!")


def get_list_of_stop_words():
    nltk_stopwords = list(stopwords.words("english"))
    spacy_stopwords = list(nlp.Defaults.stop_words)
    all_stopwords = nltk_stopwords + spacy_stopwords
    return all_stopwords


english_stopwords = get_list_of_stop_words()


def remove_stopwords(words: List[str]):
    """

    :param words:
    :return:
    """
    for word in words:
        if word in english_stopwords:
            words.remove(word)
            # print(f"{word} is stopword!")
    return words


stemmer = SnowballStemmer("english")


# print(stemmer.stem("alergare"))


def stemming(text: str):
    """

    :param text:
    :return:
    """
    stemmed_form = stemmer.stem(text)
    return stemmed_form


def lemmatizing(text: str):
    """

    :param text:
    :return:
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    return tokens, lemmas


def get_words_from_text(text: str):
    """

    :param text:
    :return:
    """
    # try:
    return text.split()
    # return nltk.tokenize.word_tokenize(text)
    # except:
    #     print(text, len(text))
    #     pdb.set_trace()

def read_list_of_common_abbreviations():
    path = "data/common_abbreviations_ro.txt"
    abbrev_dict_ = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            words = line.split("=")
            if len(words) == 1:
                break
            words[0] = " " + words[0]
            words[1] = words[1][:-1] + " "
            abbrev_dict_[words[0]] = words[1]
    return abbrev_dict_


abbrev_dict = read_list_of_common_abbreviations()


def normalize_words(words: List[str], text: str = "", option: str = "stemming", do_abbreviate: bool = False,
                    do_remove_stopwords: bool = False, do_get_words_root: bool = False, do_lowercase: bool = False,
                    do_regex_cleaning: bool = False):
    """

    :param words:
    :param text:
    :param option:
    :param do_abbreviate:
    :param do_remove_stopwords:
    :param do_get_words_root:
    :param do_lowercase:
    :param do_regex_cleaning:
    :return:
    """
    if option == "stemming":
        if do_remove_stopwords:
            words = remove_stopwords(words)
        if do_get_words_root:
            words = [stemming(word) for word in words]
        if do_abbreviate:
            words = [abbreviate(word) for word in words]
            words = flatten(words)
    elif option == "lemmatization":
        if do_remove_stopwords:
            _, lemmas = lemmatizing(text)
        if do_get_words_root and do_remove_stopwords:
            words = remove_stopwords(lemmas)
        if do_abbreviate:
            words = [abbreviate(word) for word in words]
            words = flatten(words)
    else:
        raise Exception(f"Wrong option given: {option}!")
    words = [word.replace("\ufeff", "") for word in words]
    if do_lowercase:
        words = [word.lower() for word in words]
    if do_regex_cleaning:
        words = [re.sub('[/(){}\[\]\|@,;]', ' ', word) for word in words]
        words = [re.sub('[^0-9a-z #+_]', '', word) for word in words]

    return words


def get_files_contents(folder_path: str = "docs"):
    """

    :param folder_path:
    :return:
    """
    filepaths_and_filecontents = []
    files_contents = []
    for path, directories, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(path, file)
            file_content = read_file_content(filepath)
            files_contents.append(file_content)
            filepaths_and_filecontents.append((filepath, file_content))
    return files_contents, filepaths_and_filecontents


def abbreviate(word: str, option: str = "stemming", do_abbreviate: bool = False):
    """

    :param word:
    :param option:
    :param do_abbreviate:
    :return:
    """
    words = [word]
    if word in abbrev_dict.keys():
        meaning = abbrev_dict[word]
        # print(f"{word} stands for {meaning}")
        words = get_words_from_text(meaning)
        words = normalize_words(words, do_abbreviate=do_abbreviate, option=option)
    return words


def search(word, word_to_text_idx, idx_to_text, idx_to_filepath):
    """

    :param word:
    :param word_to_text_idx:
    :param idx_to_text:
    :param idx_to_filepath:
    :return:
    """
    text_ids = word_to_text_idx[word]
    retrieved_texts = [idx_to_text[text_id] for text_id in text_ids]
    retrieved_filepaths = [idx_to_filepath[text_id] for text_id in text_ids]
    return retrieved_texts, retrieved_filepaths


def flatten(l):
    """

    :param l:
    :return:
    """
    return [item for sublist in l for item in sublist]


def indexer(texts, filepaths_and_filecontents, do_abbreviate: bool = False, option: str = "stemming"):
    """

    :param texts:
    :param filepaths_and_filecontents:
    :param do_abbreviate:
    :param option:
    :return:
    """
    words_mat = [normalize_words(get_words_from_text(text), do_abbreviate=do_abbreviate, option=option) for text in texts]
    word_to_idx = dict()
    text_to_idx = dict()
    idx_to_text = dict()
    filepath_to_idx = dict()
    idx_to_filepath = dict()

    all_words_list = flatten(words_mat)

    word_to_text_idx = dict()

    for word in all_words_list:
        for i, text in enumerate(texts):
            if word in normalize_words(get_words_from_text(text), do_abbreviate=do_abbreviate, option=option):
                if word not in word_to_text_idx.keys():
                    word_to_text_idx[word] = [i]
                else:
                    word_to_text_idx[word].append(i)

    for (filepath_and_content, text) in zip(filepaths_and_filecontents, texts):
        filepath = filepath_and_content[0]
        val = len(text_to_idx)
        filepath_to_idx[filepath] = val
        idx_to_filepath[val] = filepath
        text_to_idx[text] = val
        idx_to_text[val] = text

    num_all_words = len(all_words_list)
    word_to_frequency = dict()

    for word in all_words_list:
        if word not in word_to_frequency.keys():
            word_to_frequency[word] = 1
        else:
            word_to_frequency[word] += 1

    for (k, v) in word_to_frequency.items():
        word_to_frequency[k] = v / num_all_words

    for words_row in words_mat:
        for word in words_row:
            if word not in word_to_idx.keys():
                word_to_idx[word] = len(word_to_idx)

    return word_to_text_idx, idx_to_text, idx_to_filepath


def get_stop_words_from_text(text: str):
    words = get_words_from_text(text)
    return [word for word in words if word in english_stopwords]


def find_rhyme(text: str):
    """

    :param text:
    :return:
    """
    pass


def find_rhythm(text: str):
    """

    :param text:
    :return:
    """
    pass


from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer('all-MiniLM-L6-v2')


def embed_text(text: str):
    """

    :param text:
    :return:
    """
    emb = emb_model.encode([text])
    return np.reshape(emb, (emb.shape[1]))


# TODO REMOVE MATEI-BEJAN
def compute_average_no_of_syllables(text):
    dic = pyphen.Pyphen(lang='en')

    verses = text.split('\n')
    avg_syllables_len, no_verses = 0, None

    for verse in verses:
        text = verse.lower()
        text = re.sub('[/(){}[]|@,;]!?:.,', ' ', text)
        text = re.sub('[^0-9a-z #+_]', '', text)
        text = ' '.join(text.split())

        tokens = text.split()
        syllables = [dic.inserted(token).split('-') for token in tokens]
        syllables = list(itertools.chain.from_iterable(syllables))

        avg_syllables_len += len(syllables)

    avg_syllables_len /= len(verses)

    return round(avg_syllables_len)


# TODO REMOVE MATEI-BEJAN
def compute_no_of_rhymes(test_str):
    def extract_last_two_sounds(ipa):
        '''
            Words like "germinate" and "frustrate" end in the EY1T and EY2T sounds respectively.
            Since sounds like EY1 and EY2 are extremely similar, we remove the numerical
            character in order to make the comparison of the words more realistic.
        '''
        last_two_sounds = []
        if ipa[-2:-1][0][-1].isnumeric():
            last_two_sounds.append(ipa[-2:-1][0][:-1])
        else:
            last_two_sounds.append(ipa[-2:-1][0])
        if ipa[-1:][0][-1].isnumeric():
            last_two_sounds.append(ipa[-1:][0][:-1])
        else:
            last_two_sounds.append(ipa[-1:][0])
        return last_two_sounds

    dic = pyphen.Pyphen(lang='en')
    punc = '''!()-[]{};:'", <>./?@#$%^&*_~'''

    for ele in test_str:
        if ele in punc:
            test_str = test_str.replace(ele, " ")
    endings = []
    for verse in test_str.split('\n'):
        if len(verse.split()) > 0:
            last_word = verse.split()[-1]
            try:
                endings.append((dic.inserted(last_word).split('-'),
                                extract_last_two_sounds(pronouncing.phones_for_word(last_word)[0].split())))
            except:
                endings.append((last_word, 'NaN'))

    times_rhymes = 0
    for it1 in range(len(endings) - 1):
        for it2 in range(it1, len(endings)):
            if endings[it1][1] == endings[it2][1]:
                times_rhymes += 1
            elif endings[it1][0][-1] == endings[it2][0][-1]:
                times_rhymes += 1
            elif ''.join(endings[it1][0])[-2:] == ''.join(endings[it2][0])[-2:]:
                times_rhymes += 1

    return times_rhymes


# TODO REMOVE MATEI-BEJAN
def extract_lyrics_metadata(df, as_df=True):
    lyrics_metadata_dict = {'Genre': [], 'Avg_Syllables': [],
                            'Rhymes': [], 'Words': [], 'Verses': []}

    for row in df.iterrows():
        text = row[1]['Lyrics']

        avg_no_syllables = compute_average_no_of_syllables(text)

        no_rhymes = compute_no_of_rhymes(text)

        text = row[1]['Lyrics'].lower()
        text = re.sub('[/(){}[]|@,;]!?:.,', ' ', text)
        text = re.sub('[^0-9a-z #+_]', '', text)
        text = ' '.join(text.split())

        no_words = len(text.split())

        no_verses = len(row[1]['Lyrics'].split('\n'))

        lyrics_metadata_dict['Genre'].append(row[1]['Genre'])
        lyrics_metadata_dict['Avg_Syllables'].append(avg_no_syllables)
        lyrics_metadata_dict['Rhymes'].append(no_rhymes)
        lyrics_metadata_dict['Words'].append(no_words)
        lyrics_metadata_dict['Verses'].append(no_verses)

    if as_df:
        return pd.DataFrame(lyrics_metadata_dict)

    return lyrics_metadata_dict


"""
For each song, the dataset contains its title, artist, year, complete lyrics and genre. 
Your task is to build and compare models which predicts the song’s genre using ONLY its lyrics

Try different features: content words:
    (BOW, word embeddings), 
    stylistic markers (stop words), 
    rhyme, 
    rhythm, etc.
Try different learning algorithm ??
Find good combinations OF WHAT?
Don’t overfit (report results on (cross)validation and test) XXXX
"""


def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [' '.join(grams) for grams in n_grams]


def get_n_grams(text: str, n: int = 2, analyzer: str = "word"):
    return extract_ngrams(text, n)


def get_bi_grams(text: str, analyzer: str = "word"):
    return get_n_grams(text, 2)


def get_tri_grams(text: str, analyzer: str = "word"):
    return get_n_grams(text, 3)


def text_similarity(text_1: str, text_2: str):
    return random.random()


def label_text_by_similarity(text, labels_list):
    sims = [text_similarity(text, label) for label in labels_list]
    highest_sim = max(sims)
    highest_index = sims.index(highest_sim)
    return labels_list[highest_index]


def get_features(text: str):
    return len(text)


from statistics import mean
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale(train_data, val_data, test_data, option="standard"):
    if option == "standard":
        scaler = StandardScaler(with_mean=False)
    elif option == "minmax":
        scaler = MinMaxScaler()
    else:
        raise Exception("Wrong scaling option!")
    # print(type(train_data))
    #
    # print(train_data.shape, val_data.shape, test_data.shape)

    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
    return train_data, val_data, test_data


def average_word_len(words: List[str]):
    return mean([len(word) for word in words])


def main():
    use_val_set = False
    use_text_embedding = False
    use_paper_features = True
    use_loaded_data = True

    do_abbreviate = True
    do_remove_stopwords = True
    do_get_words_root = True
    do_lowercase = True
    do_regex_cleaning = True
    words = []
    text = ""
    words = normalize_words(words=words, text=text, do_abbreviate=do_abbreviate, do_remove_stopwords=do_remove_stopwords,
                            do_get_words_root=do_get_words_root, do_lowercase=do_lowercase,
                            do_regex_cleaning=do_regex_cleaning)
    all_labels = ['Metal', 'Hip-Hop', 'Country', 'Jazz', 'Electronic', 'Pop', 'Folk', 'Rock', 'R&B', 'Indie']

    analyzer = ["word", "character"][0]
    model_option = ["cv", "tfidf"][1]
    print("model_option: ", model_option)
    print("analyzer: ", analyzer)
    train_df, test_df = load_data()
    cv = CountVectorizer(analyzer=analyzer)
    tfidf = TfidfVectorizer(analyzer=analyzer)

    if model_option == "cv":
        vectorizer = cv
    elif model_option == "tfidf":
        vectorizer = tfidf
    else:
        raise Exception(f"Wrong model_option given: {model_option}!")

    # print(train_df.columns)
    # print(test_df.columns)

    X_train = train_df["Lyrics"].to_list()
    X_test = test_df["Lyrics"].to_list()

    if use_text_embedding:
        if use_loaded_data:
            X_train = np.load(file="data/X_train_embed_text_default.npy", allow_pickle=True)
            X_test = np.load(file="data/X_test_embed_text_default.npy", allow_pickle=True)
        else:
            X_train = np.array([embed_text(text) for text in tqdm(X_train)])
            X_test = np.array([embed_text(text) for text in tqdm(X_test)])
            np.save(arr=X_train, file="data/X_train_embed_text_default.npy", allow_pickle=True)
            np.save(arr=X_test, file="data/X_test_embed_text_default.npy", allow_pickle=True)
    else:
        if use_paper_features:
            if use_loaded_data:
                X_train = np.load(file="data/X_train_paper_features.npy", allow_pickle=True)
                X_test = np.load(file="data/X_test_paper_features.npy", allow_pickle=True)
                X_train = np.array([X[0] for X in tqdm(X_train)])
                X_test = np.array([X[0] for X in tqdm(X_test)])
            else:
                nlp = spacy.load("en_core_web_sm")
                print("paper_features")
                X_train = np.array([get_paper_features(text=text, nlp=nlp)[0] for text in tqdm(X_train)])
                X_test = np.array([get_paper_features(text=text, nlp=nlp)[0] for text in tqdm(X_test)])
                np.save(arr=X_train, file="data/X_train_paper_features.npy", allow_pickle=True)
                np.save(arr=X_test, file="data/X_test_paper_features.npy", allow_pickle=True)
        # X_train = [" ".join(normalize_words(words=get_words_from_text(text), text=text, do_abbreviate=do_abbreviate,
        #                                     do_remove_stopwords=do_remove_stopwords,
        #                                     do_get_words_root=do_get_words_root, do_lowercase=do_lowercase,
        #                                     do_regex_cleaning=do_regex_cleaning)) for text in X_train]
        #
        # X_test = [" ".join(normalize_words(words=get_words_from_text(text), text=text, do_abbreviate=do_abbreviate,
        #                                    do_remove_stopwords=do_remove_stopwords,
        #                                    do_get_words_root=do_get_words_root, do_lowercase=do_lowercase,
        #                                    do_regex_cleaning=do_regex_cleaning)) for text in X_test]

    # print(X_train.shape, X_test.shape)

    y_train = train_df["Genre"].to_list()
    y_test = test_df["Genre"].to_list()

    labels_set = set(y_train)
    print("Number of possible genres: ", len(labels_set))
    labels_dict = dict()
    for label in labels_set:
        labels_dict[label] = len(labels_dict)

    y_train = [labels_dict[y] for y in y_train]
    y_test = [labels_dict[y] for y in y_test]

    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # y_train, y_test, y_test = scale(y_train, y_test, y_test)

    # print(X_train[0])
    # print(X_train[0].find("\n"))

    if use_val_set:
        val_size = 0.2
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)

    if use_text_embedding is False:
        if use_paper_features is False:
            X_train = vectorizer.fit_transform(X_train)
            if use_val_set:
                X_val = vectorizer.transform(X_val)
            X_test = vectorizer.transform(X_test)

    # X_train, X_val, X_test = scale(X_train, X_val, X_test)

    f1_average = "weighted"
    # model = SVC()  # class_weight="balanced")
    model = xgb.XGBClassifier()

    print(X_train[0], y_train.shape)

    model.fit(X_train, y_train)

    if use_val_set:
        y_pred = model.predict(X_val)
        print("Validation F1 with tfidf and SVC: ", f1_score(y_pred, y_val, average=f1_average))

    y_pred = model.predict(X_test)
    print("Test F1 with tfidf and SVC: ", f1_score(y_pred, y_test, average=f1_average))


if __name__ == "__main__":
    main()
