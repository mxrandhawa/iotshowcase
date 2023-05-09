#from typing import List
#from curses.ascii import isalpha
import spacy
import numpy as np
import pandas as pd
import DIIM_config as config
import logging
import random
import os.path
#import csv

#nlp = spacy.load('en_core_web_sm')
logger = logging.getLogger(config.LOGAPPLICATION_NAME)

ids = []


def create_matrix(rows_list, col_list):
   # print(rows_list)
   # print("rows " + str(len(rows_list)))
   # print(col_list)
   # print("cols " + str(len(col_list)))

    # word_similarity_matrix = np.chararray([matrix_size, matrix_size], unicode=True)
    word_similarity_matrix = np.zeros(
        [len(rows_list), len(col_list)], dtype='object')

    return word_similarity_matrix


def create_dataframe_with_cols_and_index_names(rows_list, col_list):
    # create an empty matrix
    narray = create_matrix(rows_list, col_list)

    # creating the dataframe
    df = pd.DataFrame(data=narray,
                      index=rows_list,
                      columns=col_list)
    return df


def create_matrix_with_cols(rows_list, col_list):
   # print(rows_list)
   # print("rows " + str(len(rows_list)))
   # print(col_list)
   # print("cols " + str(len(col_list)))

    # word_similarity_matrix = np.chararray([matrix_size, matrix_size], unicode=True)
    word_similarity_matrix = np.zeros(
        [len(rows_list) + 1, len(col_list) + 1], dtype='object')

    # set values in first col of all rows, starting from 1
    for i in range(0, len(rows_list), 1):
        #mystr = str(i) + " " + rows_list[i]
        word_similarity_matrix.itemset((i + 1, 0), str(rows_list[i]))

    # set values in all columns of first row, starting from 1
    for i in range(0, len(col_list), 1):
        word_similarity_matrix.itemset((0, i + 1), str(col_list[i]))

    return word_similarity_matrix


def create_matrix_with_cols_and_rows(rows_list, col_list):
   # print(rows_list)
   # print("rows " + str(len(rows_list)))
   # print(col_list)
   # print("cols " + str(len(col_list)))

    # word_similarity_matrix = np.chararray([matrix_size, matrix_size], unicode=True)
    word_similarity_matrix = np.zeros(
        [len(rows_list) + 1, len(col_list) + 1], dtype='object')

    # set values in first col of all rows, starting from 1
    for i in range(0, len(rows_list), 1):
        #mystr = str(i) + " " + rows_list[i]
        word_similarity_matrix.itemset((i + 1, 0), str(rows_list[i]))

    # set values in all columns of first row, starting from 1
    for i in range(0, len(col_list), 1):
        word_similarity_matrix.itemset((0, i + 1), str(col_list[i]))

    return word_similarity_matrix


def write_dataframe_obj_to_csv(csv_filename, df_obj):

    if os.path.isfile(csv_filename):
        logger.info('File ' + csv_filename +
                    ' already exists, therefore skipping writing file!')
        return

    logger.info('Writing matrix to file ' + csv_filename)
    df_obj.to_csv(csv_filename)
    # write comments first
    logger.info('Done !')


def write_matrix_to_csv(csv_filename, matrix):

    if os.path.isfile(csv_filename):
        logger.info('File ' + csv_filename +
                    ' already exists, therefore skipping writing file!')
        return

    logger.info('Writing matrix to file ' + csv_filename)
    if (len(matrix) > 0):
        df = pd.DataFrame(matrix)
        # df.to_csv(comment_in_csv)
        df.to_csv(csv_filename, header=False, index=False)
    # write comments first
    logger.info('Done !')


def write_matrix_with_index_and_info(csv_filename, matrix, commentlist):
    logger.info('Writing matrix to file ' + csv_filename)

    if (len(matrix) > 0):
        df = pd.DataFrame(matrix)
        # df.to_csv(comment_in_csv)
        df.to_csv(csv_filename)
    # write comments first
    with open(csv_filename, 'a') as f:
        for comment in commentlist:
            f.write(comment)
    f.close()

    logger.info('Done !')


def create_nlp_corpus(list):
    corpus = ''
    for word in list:
        corpus += word + ' '
    tokens = nlp(corpus)
    # print(corpus)
    #print('list: ' + str(list))
    #print('corpus: ' + corpus)
    #print('tokens: ' + str(tokens))
    return tokens


def split_words_in_list(list, delimiter):
    new_list = []
    for i in list:
        splitted_words = str(i).split(delimiter)
        for sp_word in splitted_words:
            # if(sp_word.isalpha()):
            new_list.append(sp_word)

    new_list.sort()
    return new_list


def create_unique_list_of_words(list):
    list_set = {''}
    for i in list:
        list_set.add(i)
    list_set.remove('')
    list_unique = sorted(list_set)
    return list_unique


def get_random_id():
    id = random.randint(0, 1000)
    while (id in ids):
        id = random.randint(0, 1000)
    ids.add(id)
    return id


def get_fileName(filePath):
    folder, filename = os.path.split(filePath)
    return filename


def get_fileName_without_extension(fileName):
    substrings = str(fileName).split('.')
    if len(substrings) >= 2:
        return substrings[0]
    return fileName


def get_fileNames_from_filePath(diim_similarity_matrix_filePath):
    dataset_folder_path, fileName = os.path.split(
        diim_similarity_matrix_filePath)
    diim_similarity_fileName = get_fileName_without_extension(fileName)
    fileNames = diim_similarity_fileName.split('_')
    firstFileName = fileNames[0]
    secondFileName = fileNames[1]
    return firstFileName, secondFileName


def get_filePath_with_similarity_algorithmPattern(diim_dataset1, diim_dataset2, algorithName, fileExtension):
    # add first dataset 1 name without extenstion to filepath
    filePath = config.storeDir + \
        get_fileName_without_extension(diim_dataset1.get_name())
    # add first dataset 1 id to filepath
    filePath = filePath + str(diim_dataset1.get_id())
    # print(filePath)
    # add delimiter '_' and first dataset 2 id to filepath
    # add first dataset 2 name without extenstion delimiter '_' and algorith name to filepath
    filePath += '_' + str(diim_dataset2.get_id()) + \
        get_fileName_without_extension(diim_dataset2.get_name())
    filePath += '_' + algorithName + '.' + fileExtension

    return filePath
