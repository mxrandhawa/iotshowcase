from turtle import title
import matplotlib.pyplot as plt
import pandas as pd


def show_diim_similarity_matrix(diim_similarity_matrix):
    # show just one word
    df_obj = pd.DataFrame(diim_similarity_matrix.get_similarity_matrix())
    x_label = df_obj.columns[0]
    #y_label = df_obj.columns[1]
    # show once
    #df_obj.plot(kind='scatter', x=x_label, y=y_label)
    # plt.show()

    # gca stands for 'get current axis'
    ax = plt.gca()
    y_labels = df_obj.columns[1:]
    for label_y in y_labels:
        df_obj.plot(kind='scatter', x=x_label, y=label_y, ax=ax)

    title_arg1 = "Similarity Algorithm: " + \
        diim_similarity_matrix.get_algorithm_name()

    dataset1_terms = diim_similarity_matrix.get_dataset_1_terms_cleaned()
    #title_arg2 = "X-axis: " + str(dataset1_terms)
    #title_arg2 = title_arg2 + ' Length: ' + str(len(dataset1_terms))
    title_arg2 = construct_Ylabel_compact(
        'X-axis', diim_similarity_matrix.get_dataset1_name(), dataset1_terms, len(dataset1_terms))

    dataset2_terms = diim_similarity_matrix.get_dataset_2_terms_cleaned()
    title_arg3 = construct_Ylabel_compact(
        'Y-axis', diim_similarity_matrix.get_dataset2_name(), dataset2_terms, 10)  # len(dataset1_terms))

    #title_arg3 = "\nY-axis: " + str(diim_similarity_matrix.get_dataset_2_terms_cleaned())
    subtitle_string = title_arg2 + title_arg3

    #plt.suptitle(title_arg1, y=1.05, fontsize=10)
    plt.title(title_arg1 + subtitle_string, fontsize=14)
    plt.show()


def construct_Ylabel_compact(axis, dataset_name, namelist, limit):
    title_arg = '\n' + axis + ': ' + dataset_name + ' ['
    for i in range(limit):
        title_arg = title_arg + ' ' + namelist[i] + ', '
    title_arg = title_arg + '...]'
    title_arg = title_arg + ' Term count: ' + str(len(namelist))
    return title_arg


def construct_Ylabel_from_all(namelist):
    i = 1
    title_arg3 = '\nY-axis: ['
    for term in namelist:
        if (i % 20 == 0):
            title_arg3 = title_arg3 + '\n' + term + ', '
        else:
            title_arg3 = title_arg3 + term + ', '
        i += 1
    title_arg3 = title_arg3 + ']'
    return title_arg3


def show_diim_similarity_matrices(diim_similarity_matrices):
    for diim_similarity_matrix in diim_similarity_matrices:
        show_diim_similarity_matrix(diim_similarity_matrix)
