# python imports
import os
from pathlib import Path
import string
import re

# matplot imports
import matplotlib.pyplot as plt
from matplotlib import pyplot
# numpy imports
import numpy as np
# pandas imports
import pandas as pd
# sklearn imports
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
# gensim imports
from gensim.models import word2vec
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_documents
from gensim import utils
from gensim.parsing.porter import PorterStemmer
from gensim.parsing import stem_text
# nltk imports
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
# misc imports
from traceback import print_tb
from string import ascii_lowercase


# local imports
import DIIM_config as config

# variables
logger = config.logger

EN_STOP_WORDS = stopwords.words('english')
EN_STOP_WORDS.extend(['is', 'may', 'also', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                     'eight', 'nine', 'ten', 'across', 'among', 'beside', 'however', 'yet', 'within']+list(ascii_lowercase))

# Test word from the IoT data on water quality monitoring
IoT_TEST_WORDS = ['sensor', 'system', 'time', 'measurement', 'digital', 'platform',
                  'actuator', 'data', 'value', 'property', 'determinand',
                  'process', 'location', 'position', 'temperature', 'description', 'pH', 'Clorine']  # , 'temp'


def clean_sentence(val):
    # replace chars that are not letters or numbers with space and downcase
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub(' ', val).lower()

    # replace digits with space
    sentence = replace_digits_with_space(sentence)
    sentence = sentence.split(" ")

    # remove stop words
    for word in list(sentence):
        if word in EN_STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


def filter_stopwords(words):
    # remove it later
    filtered_sentence = []
    en_stopwords = set(stopwords.words('english'))
    for w in words:
        if w not in en_stopwords:
            filtered_sentence.append(w)

    Stem_words = []
    ps = PorterStemmer()
    for w in filtered_sentence:
        rootWord = ps.stem(w)
        Stem_words.append(rootWord)
    # print(filtered_sentence)
    # print(Stem_words)
    return filtered_sentence


def build_corpora_from_DIR(ONTO_DIR_Path):
    # return a dataframe
    data = []
    # logger.info('starting with NLP of files ending with ' + fileType)
    # look up JSONFiles in current dir and analyze them
    config.logger.info('Looking for input data in dir ' + str(ONTO_DIR_Path))
    for currentpath, folders, files in os.walk(ONTO_DIR_Path):
     # for file in filter(lambda file: file.endswith('.txt'), files):
     # iot or examples for ontologies xml, rdf, csv, json
        for file in filter(lambda file: file.endswith('.owl')
                           or file.endswith('.rdf')
                           or file.endswith('.xml')
                           or file.endswith('.txt')
                           or file.endswith('.json')
                           or file.endswith('.csv')
                           or file.endswith('.html'), files):
            filePath = os.path.join(currentpath, file)
            config.logger.info('Creating corpus from ' + filePath)
            # data.append(build_corpora(filePath))
            build_corpora(data, filePath)

    # Calling DataFrame constructor on list
    corpora_df = pd.DataFrame(data)

    return corpora_df


# def get_w2v_word_similarities(model, words):
#     similar_words = {}
#     for word in words:
#         similar_words(model, word)

# def save_DIIM_Dataframe_Corpus(diim_dataframe):


def get_similar_words(ontology_name, model, words, topn_limit):
    # Finding most similar words
    # return set object
    # key = ontology_name + '_'+ word
    # vlaue = most similarwords as a list of tuples (word , int)
    similar_words = {}

    for word in words:

        if word == 'None':
            logger.warn(
                'TODO: None found in IoT list of words refactore code in: ', ontology_name)
            continue

        if word is None:
            logger.warn(
                'TODO: None found in IoT list of words refactore code in: ', ontology_name)
            continue

        elif word in model.wv:
            logger.info(ontology_name,
                        " has following 10 words similar to: " + word)
            found_similarities = model.wv.most_similar(word, topn=topn_limit)
            key = ontology_name + '_' + word
            similar_words[key] = found_similarities
            logger.info(similar_words)
        else:
            # this should not happen as the word was trained in the model
            logger.warn(ontology_name,
                        ' Word2Vec model doesn\'t contain ', word)

    return similar_words


def drop_all_rows_with_all_none_values(corpus):

    new_corpus = corpus.dropna(how='all')
    return new_corpus


def drop_all_columns_with_all_none_values(corpus):

    new_corpus = corpus.dropna(axis=1, how='all')
    return new_corpus


def drop_all_rows_and_columns_with_all_none_values(corpus):

    corpus = drop_all_rows_with_all_none_values(corpus)
    corpus = drop_all_columns_with_all_none_values(corpus)
    return corpus


def drop_empty_rowsandcolums(corpora_df):

    corpora_df.replace(
        to_replace='None', value=np.nan, inplace=True)
    #nan_value = float("NaN")
    #corpora_df.replace(0, nan_value, inplace=True)

    # Drop columns that has all NaN values
    corpora_df.dropna(how='all', axis=1, inplace=True)

    # Drop rows that has all NaN values
    corpora_df.dropna(how='all')

    return corpora_df


# def drop_emptyrows(corpora_df):
#     # print(corpora_df.head)
#     index_of_emptyrows = []
#     # print(corpora_df.isna().sum())
#     # for i in range(len(corpora_df.index)):
#     nr_of_rows = corpora_df.shape[0]
#     nr_of_cols = corpora_df.shape[1]
#     print(range(nr_of_rows))
#     for i in range(nr_of_rows):
#         # count all nulls in the row
#         nr_of_Nulls = corpora_df.iloc[i].isnull().sum()
#         # if the nr of nulls equals nr of cols --> mark rows
#         if (nr_of_Nulls >= nr_of_cols - 1):  # remove lines with single word or no words at all
#             # print("Nan in row ", i, " : ", nr_of_Nulls)
#             index_of_emptyrows.append(i)

#     corpora_df = corpora_df.drop(index_of_emptyrows)

#     return corpora_df


def tokenize_file(input_path):
    lines = []
    with open(input_path, encoding="utf8") as f:
        lines = f.readlines()

    # print('*** printing file content')
    # for sentence in lines:
    #     print(sent_tokenize(sentence))

    # Preprocessing data to lowercase all words and remove single punctuation words
    # data = preprocess_documents(lines)
    # print(data)

    data = []
    for sent in lines:
        new_sent = []
        # clean sentence
        sent = clean_sentence(sent)
        # tokenize the sentence
        words = word_tokenize(sent)
        data.append(words)
    #     for word in words:
    #         new_word = word.lower()
    #         if new_word[0] not in string.punctuation:
    #             new_sent.append(new_word)
    #     if len(new_sent) > 0:
    #         data.append(new_sent)
    # print(data)

    return data


def replace_digits_with_space(string):
    pattern = r'[0-9]'
    # Match all digits in the string and replace them with an empty string
    new_string = re.sub(pattern, ' ', string)
    return new_string


# def replace_punctuation_with_space(string):
#     pattern = r'([^\s\w]|_)+'
#     # Match all digits in the string and replace them with an empty string
#     new_string = re.sub(pattern, ' ', string)
#     return new_string


def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub(' ', val).lower()
    sentence = replace_digits_with_space(sentence)
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in EN_STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


def filter_stopwords(words):
    # remove it later
    filtered_sentence = []
    en_stopwords = set(stopwords.words('english'))
    for w in words:
        if w not in en_stopwords:
            filtered_sentence.append(w)

    Stem_words = []
    ps = PorterStemmer()
    for w in filtered_sentence:
        rootWord = ps.stem(w)
        Stem_words.append(rootWord)
    # print(filtered_sentence)
    # print(Stem_words)
    return filtered_sentence


def show_scatter_plot(model, words):
    X = model.wv[words]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


def print_vocabulary(wv_model):
    wv_model_size = len(wv_model.wv.index_to_key)
    for index, word in enumerate(wv_model.wv.index_to_key):
        # print first 100 words
        if index == 100:
            break
        print(f"word #{index}/{wv_model_size} is {word}")


def build_corpora(data, input_path):

    more_sentences = tokenize_file(input_path)
    for sent in more_sentences:
        data.append(sent)
    return data


def remove_punctuation(data):
    # list(list[str]) --> list(list[str])
    filtered_data = []
    for list_string in data:
        filtered_list = []
        for a_string in list_string:
            # treat url
            a_string.replace('.', '. ')
            a_string.replace('//', '// ')
            # remove puncuation
            new_string = a_string.translate(
                str.maketrans('', '', string.punctuation))
            filtered_list.append(new_string)
        filtered_data.append(filtered_list)
    print(filtered_data)
    return filtered_data


def split_ontology_iot_word_name(ontology_iot_word_name):
    ontology_name = ''
    iot_name = ''
    label = ''
    title_words = ontology_iot_word_name.split('_')

    if 'WHO_Drinking' in ontology_iot_word_name:  # it contains more then 4 '_'
        ontology_name1 = title_words[0]
        ontology_name2 = title_words[1]
        ontology_name = ontology_name1 + '_' + ontology_name2

        iot_name = title_words[2]
        if (len(title_words) == 3):
            label = title_words[2]
        elif (len(title_words) == 4):
            label = title_words[3]

    else:

        ontology_name = title_words[0]
        iot_name = title_words[1]
        if (len(title_words) == 3):
            label = title_words[2]

    return ontology_name, iot_name, label


drop_all_rows_with_all_none_values

# converts a dataframe corpus to a document corpus i.e. a list of sentences
# removes empty rows and columns from dataframe corpus


def convert_dataframe_to_documents(corpus):

    new_corpus = drop_all_rows_and_columns_with_all_none_values(corpus)
    # print(new_corpus)
    corpus_lists = new_corpus.values.tolist()
    documents = []
    none_counter = 0
    for corpus_list in corpus_lists:
        doc = ''
        for word in corpus_list:
            if word is None:
                none_counter = none_counter + 1
            else:
                doc = word + ' ' + doc
        documents.append(doc)

    # print(documents)

    return documents
