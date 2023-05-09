import numpy as np
import nltk
import pandas as pd
# local imports
import utility as ut
ALGORITHM_NAME = 'Levenshtein'


def generate_similarity_matrix_with_cols_names(row_names, col_names):

    dict_obj = {}
    x_label = 'token'
    # add token names in first column with name as 'token'
    dict_obj[x_label] = row_names
    # print(dict_obj[x_label])

    for col_word in col_names:
        dict_obj[col_word] = calculate_similarity(col_word, row_names)
        # print(dict_obj[col_word])
    # df_obj = pd.DataFrame(dict_obj, index=list2) when tokens are created
    # the list has more elements becuase it split after white spaces
    return dict_obj


def calculate_similarity(col_word, row_words):
    vals = []
    for word in row_words:
        # INFO: Calculate the Levenshtein edit-distance between two strings.
        # The edit distance is the number of characters that need to be
        # substituted, inserted, or deleted, to transform s1 into s2.
        # For example, transforming \"rain\" to \"shine\" requires three
        # steps, consisting of two substitutions and one insertion:
        # \"rain\" -> \"sain\" -> \"shin\" -> \"shine\".
        # These operations could have been done in other orders, but at
        # least three steps are needed. Allows specifying the cost of
        # substitution edits (e.g., \"a\" -> \"b\"), because sometimes
        # it makes sense to assign greater penalties to substitutions.
        vals.append(int(nltk.edit_distance(col_word, word)))
    return vals


def write_matrix_to_CSV(matrix, csvFilepath):
    ut.write_matrix_to_csv(csvFilepath, matrix)


def get_lib_info():

    # comment_in_csv = "# INFO: Calculate the Levenshtein edit-distance between two strings.
    # The edit distance is the number of characters that need to be substituted, inserted,
    # or deleted, to transform s1 into s2.  For example, transforming \"rain\" to \"shine\"
    # requires three steps, consisting of two substitutions and one insertion: \"rain\" ->
    # \"sain\" -> \"shin\" -> \"shine\".  These operations could have been done in other
    # orders, but at least three steps are needed. Allows specifying the cost of substitution
    # edits (e.g., \"a\" -> \"b\"), because sometimes it makes sense to assign greater
    # penalties to substitutions."
    comment_in_csv_1 = "# INFO: Calculate the Levenshtein edit-distance between two strings. \n"
    comment_in_csv_2 = "# INFO: The edit distance is the number of characters that need to be substituted inserted or deleted to transform s1 into s2."
    comment_in_csv_3 = "# INFO: For example  transforming \"rain\" to \"shine\" requires three steps  consisting of two substitutions and one insertion: \"rain\" -> \"sain\" -> \"shin\" -> \"shine\". \n"
    comment_in_csv_4 = "# INFO: These operations could have been done in other orders but at least three steps are needed. \n"
    comment_in_csv_5 = "# INFO: Allows specifying the cost of substitution edits (e.g.  \"a\" -> \"b\") because sometimes it makes sense to assign greater penalties to substitutions."
    commentlist = [comment_in_csv_1, comment_in_csv_2,
                   comment_in_csv_3, comment_in_csv_4, comment_in_csv_5]

    return commentlist
