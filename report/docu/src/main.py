from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.gold_standard_linguistical_feature_extraction import get_paper_features
from sklearn.model_selection import train_test_split
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from gensim.test.utils import common_texts
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import cross_validate
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from langdetect import detect
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.svm import SVC
from copy import deepcopy
from typing import List
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from xgboost import XGBClassifier


import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
import xgboost as xgb
import numpy as np

import pyphen
import multiprocessing
import nltk
import pdb
import pronouncing
import nltk
import pdb
import glob
import fasttext
import spacy
import csv
import os
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


nlp = spacy.load("en_core_web_sm")
all_stop_words = spacy.lang.en.stop_words.STOP_WORDS

SAVING_FOLDER = "data/"

genres = {'Folk', 'Indie', 'Rock', 'Hip-Hop', 'R&B', 'Country', 'Metal', 'Pop', 'Jazz', 'Electronic'}


def create_df_from_arrays(X, Y, saving_filename):
    dict_df = {"text": [], "label": []}

    for (x, y) in list(zip(X, Y)):
        dict_df['text'].append(x)
        dict_df['label'].append(y)

    df = pd.DataFrame.from_dict(dict_df)

    df.to_csv(saving_filename)


def preprocess(X, preprocessing_method=""):
    if preprocessing_method == "":
        return X
    for i in range(len(X)):
        preprocessed_x = []
        doc = nlp(str(X[i]))
        for token in doc:
            if preprocessing_method == "STOPWORDS_ELIM":
                if token.text.lower() not in all_stop_words:
                    preprocessed_x.append(token.text)
            elif preprocessing_method == "PART_OF_SPEECH":
                preprocessed_x.append(token.text + "_" + token.dep_)
            elif preprocessing_method == "DEPENDENCY":
                preprocessed_x.append(token.text + "_" + token.dep_)

        X[i] = " ".join(preprocessed_x)

    return np.array(X)


def baseline_models(X_train, y_train, X_test, y_test, subject_feature=False):
    models = [BernoulliNB(), MultinomialNB(), Perceptron(),
              AdaBoostClassifier(), RandomForestClassifier(),
              DecisionTreeClassifier(), XGBClassifier()]
    models_names = ["BernoulliNB()", "MultinomialNB()",
                    "Perceptron()", "AdaBoostClassifier()",
                    "RandomForestClassifier()",
                    "DecisionTreeClassifier()", "XGBClassifier()"]


    if subject_feature:
        create_df_from_arrays(X_train, y_train, "data/bert_train.csv")
        create_df_from_arrays(X_test, y_test, "data/bert_test.csv")
        X_train = preprocess(X_train)
        X_test = preprocess(X_test)
        cv = CountVectorizer()
        X_train = cv.fit_transform(X_train)
        X_test = cv.transform(X_test)


    for (model, model_name) in list(zip(models, models_names)):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("*" * 10)
        print(model_name)
        print("ACC: ", accuracy_score(y_pred, y_test))
        print("F1: ", f1_score(y_pred, y_test, average="weighted"))


def load_tokenizer():
    """
    :return:
    """
    return BertTokenizer.from_pretrained('bert-base-uncased')

tokenizer = load_tokenizer()

# @profile
def load_model():
    """
    :return:
    """
    return BertModel.from_pretrained("bert-base-uncased")


model = load_model()


def embed_text_bert(text: str, model: Any, tokenizer: Any, vectorization_method: str = "transformer"):
    """
    :param text:
    :param model:
    :param tokenizer:
    :param vectorization_method:
    :return:
    """
    # print(type(text))
    max_len = 512
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input["input_ids"] = encoded_input["input_ids"][:, :max_len]
    encoded_input["token_type_ids"] = encoded_input["token_type_ids"][:, :max_len]
    encoded_input["attention_mask"] = encoded_input["attention_mask"][:, :max_len]
    output = model(**encoded_input)
    # print(output["pooler_output"].shape)
    return output["pooler_output"].detach().numpy()

def save_data(df, train_size, target_column, content_column):
    # for i in range(len(df)):
    #     df.at[i, target_column] = '__label__' + df.iloc[i][
    #         target_column].replace(" ", "_")

    df = df[[target_column, content_column]]
    # the column order needs to be changed for processing with the FastText
    # model
    df = df.reindex(columns=[target_column, content_column])

    df[:int(train_size * len(df))].to_csv(
        os.path.join(SAVING_FOLDER, "train.txt"),
        index=False,
        sep=' ',
        header=False,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar=" ")

    df[int(train_size * len(df)):].to_csv(
        os.path.join(SAVING_FOLDER, "test.txt"),
        index=False,
        sep=' ',
        header=False,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar=" ")

def train_fasttext_model(train_size, test_size):
    model = fasttext.train_supervised(input=os.path.join("data",
                                                         "train.txt"),
                                      autotuneValidationFile="data/test.txt")
    model.save_model(
        "model_news_" +
        str(train_size) +
        "-" +
        str(test_size) +
        ".bin")
    return model

def train_fasttext(df, target_column, content_column):
    model_name = "Fasttext Model"
    train_size = 0.8
    test_size = 0.2
    save_data(df, train_size, target_column, content_column)
    # trains the fast-text model on the first (train_size * 100) % of the data
    model = train_fasttext_model(train_size, test_size)
    # tests the fast-text model accuracy on the last ((train_size -
    # test-size) * 100) % of the data


def fasttext_model():
    try:
        os.mkdir("data")
    except FileExistsError:
        pass
    df = pd.read_csv("data/data.csv")
    train_fasttext(df, "label", "text")



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


def document_preprocess(document):
    """
    :param document:
    :return:
    """
    return word_tokenize(document)


def train_word2vec(texts):
    print
    print(type(texts[0]))
    preprocessed_texts = []
    for text in texts:
        try:
            preprocessed_texts.append(document_preprocess(text))
        except:
            print("*", type(text)) # , text)
    texts = preprocessed_texts
    # texts = [document_preprocess(text) ]

    print(multiprocessing.cpu_count())

    print(texts[0])
    print(texts[-1])
    model = Word2Vec(sentences=texts, vector_size=300, window=5, min_count=1, workers=multiprocessing.cpu_count())
    model.save("checkpoints/word2vec.model")  # checkpoint is saved
    print("Saved word2vec model")


def load_word2vec():
    model = Word2Vec.load("checkpoints/word2vec.model")
    return model


def predict_word2vec(text, model):
    preproc_text = document_preprocess(text)
    return mean([model.wv[word] for word in preproc_text])


def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [' '.join(grams) for grams in n_grams]


def get_n_grams(text: str, n: int = 2, analyzer: str = "word"):
    return extract_ngrams(text, n)


def get_bi_grams(text: str, analyzer: str = "word"):
    return get_n_grams(text, 2)


def text_similarity(text_1, text_2):
    if isinstance(text_1, str):
        emb_1 = embed_text(text_1) #, model, tokenizer)
    else:
        emb_1 = text_1
    if isinstance(text_2, str):
        emb_2 = embed_text(text_2) #, model, tokenizer)
    else:
        emb_2 = text_2
    return cosine_similarity(emb_1, emb_2)


def train_by_text_similarity(X_train, y_train, X_test, y_test):
    print("Text similarity training...")
    # idea 1 similarity of test lyrics with train lyrics for a certain category
    # idea 2 similarity of the class labels with the lyrics
    print(type(X_train[0]))
    X_train = [embed_text(text) for text in tqdm(X_train)]
    X_test = [embed_text(text) for text in tqdm(X_test)]

    labels_avg_embs = dict()
    for (emb, label) in tqdm((X_train, y_train)):
        if label not in labels_avg_embs.keys():
            labels_avg_embs[label] = [emb]
        else:
            labels_avg_embs[label].append(emb)

    for (k, v) in labels_avg_embs.items():
        labels_avg_embs[k] = mean(v)

    y_pred = []
    for (emb, label) in X_test:
        values = []
        for (label_, avg_emb) in labels_avg_embs.items():
            values.append((label_, text_similarity(emb, avg_emb)))
        values = sorted(values, key=lambda x: x[1])
        y_pred.append(values[-1][0])

    print("F1:", f1_score(y_test, y_pred))

    return X_train, X_test


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


def train_bert_for_seq_classif(sentences, labels):
    import tensorflow as tf

    # Checking for the GPU
    device_name = tf.test.gpu_device_name()
    print(device_name)

    import torch

    device = torch.device("cpu")

    # Load the BERT tokenizer.

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    MAX_LEN = 64

    # Padding the input to the max length that is 64
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    # Creating the attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    # We will call the train_test_split() function from sklearn
    from sklearn.model_selection import train_test_split

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=2018, test_size=0.1)
    # Performing same steps on the attention masks
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                           random_state=2018, test_size=0.1)

    # Converting the input data to the tensor , which can be feeded to the model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

    # Creating the DataLoader which will help us to load data into the GPU/CPU
    batch_size = 32

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    from transformers import BertForSequenceClassification, AdamW, BertConfig

    print("num possible labels: ", len(set(labels)))
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(set(labels)),
        output_attentions=False,
        output_hidden_states=False, )

    # AdamW is an optimizer which is a Adam Optimzier with weight-decay-fix
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )

    from transformers import get_linear_schedule_with_warmup

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 4

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)



    # https://www.kaggle.com/code/akshat0007/bert-for-sequence-classification whole function

    import numpy as np

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    # Creating the helper function to have a watch on elapsed time

    import time
    import datetime

    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # Let's start the training process

    import random

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    # torch.dev.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")

    print(loss_values)  # Having a view of stored loss values in the list

    [0.09573989999538511, 0.07170906540825654, 0.07369904678423128, 0.09808315255551665]

    # Loading the test data and applying the same preprocessing techniques which we performed on the train data
    import pandas as pd

    # Load the dataset into a pandas dataframe.
    df = pd.read_csv("./cola_public/raw/out_of_domain_dev.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    # Create sentence and label lists
    sentences = df.sentence.values
    labels = df.label.values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        )

        input_ids.append(encoded_sent)

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

        # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    # Set the batch size.
    batch_size = 32

    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Evaluating our model on the test set

    # Prediction on test set

    print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)


    print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))

    from sklearn.metrics import matthews_corrcoef

    matthews_set = []

    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating Matthews Corr. Coef. for each batch...')

    # For each input batch...
    for i in range(len(true_labels)):
        # The predictions for this batch are a 2-column ndarray (one column for "0"
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

        # Calculate and store the coef for this batch.
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)

    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

    print('MCC: %.3f' % mcc)

    MCC: 0.529


def main():
    use_val_set = False
    use_text_embedding = True
    use_paper_features = False
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

    y_train = train_df["Genre"].to_list()
    y_test = test_df["Genre"].to_list()

    labels_set = set(y_train)
    print("Number of possible genres: ", len(labels_set))
    labels_dict = dict()
    for label in labels_set:
        labels_dict[label] = len(labels_dict)

    y_train = [labels_dict[y] for y in y_train]
    y_test = [labels_dict[y] for y in y_test]

    sentences = X_train + X_test
    labels = y_train + y_test

    # train_word2vec(sentences)
    word2vec_model = load_word2vec()

    train_size = 0.8
    test_size = 0.2
    content_column = "Lyrics"
    target_column = "Genre"

    df = train_df + test_df
    # baseline_models(X_train, y_train, X_test, y_test, subject_feature=False)
    # save_data(df, train_size, target_column, content_column)
    # train_fasttext_model(train_size, test_size)
    # train_fasttext(df, target_column, content_column)
    # fasttext_model()

    # train_bert_for_seq_classif(sentences, labels)

    use_word2vec_features = False
    do_words_normalization = False

    if do_words_normalization is True:
        use_paper_features = False

    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    if use_text_embedding:
        print("text embedding...")
        if use_loaded_data:
            print('loading data...')
            X_train = np.load(file="data/X_train_embed_text_default.npy", allow_pickle=True)
            X_test = np.load(file="data/X_test_embed_text_default.npy", allow_pickle=True)
        else:
            X_train = np.array([embed_text(text) for text in tqdm(X_train)])
            X_test = np.array([embed_text(text) for text in tqdm(X_test)])
            np.save(arr=X_train, file="data/X_train_embed_text_default.npy", allow_pickle=True)
            np.save(arr=X_test, file="data/X_test_embed_text_default.npy", allow_pickle=True)
    else:
        if use_paper_features:
            print("paper features...")
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
        elif do_words_normalization:
                print("words normalization...")
                X_train = [" ".join(normalize_words(words=get_words_from_text(text), text=text, do_abbreviate=do_abbreviate,
                                                    do_remove_stopwords=do_remove_stopwords,
                                                    do_get_words_root=do_get_words_root, do_lowercase=do_lowercase,
                                                    do_regex_cleaning=do_regex_cleaning)) for text in X_train]

                X_test = [" ".join(normalize_words(words=get_words_from_text(text), text=text, do_abbreviate=do_abbreviate,
                                                   do_remove_stopwords=do_remove_stopwords,
                                                   do_get_words_root=do_get_words_root, do_lowercase=do_lowercase,
                                                   do_regex_cleaning=do_regex_cleaning)) for text in X_test]
        elif use_word2vec_features:
            print("word2vec features...")
            X_train = [predict_word2vec(text, word2vec_model) for text in X_train]
            X_test = [predict_word2vec(text, word2vec_model) for text in X_test]


    # print(X_train.shape, X_test.shape)

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
    model = SVC()  # class_weight="balanced")

    # train_by_text_similarity(X_train, y_train, X_test, y_test)
    print("fitting...")
    v_results = cross_validate(model, X_train, y_train, cv=3)
    print("results from cross validation...", v_results)

    # model = xgb.XGBClassifier()

    print(X_train[0], y_train.shape)

    model.fit(X_train, y_train)

    if use_val_set:
        y_pred = model.predict(X_val)
        print("Validation F1 with tfidf and SVC: ", f1_score(y_pred, y_val, average=f1_average))

    y_pred = model.predict(X_test)
    print("Test F1 with tfidf and SVC: ", f1_score(y_pred, y_test, average=f1_average))


if __name__ == "__main__":
    main()
