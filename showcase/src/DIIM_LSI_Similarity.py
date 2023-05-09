
from pprint import pprint
from gensim import corpora
from gensim import models
from gensim import similarities
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric


#import DIIM_LSI_Onto as lsi_onto
import DIIM_Corpora_Onto as corpora_onto
import DIIM_Corpora_IoT as corpora_iot
import DIIM_config as config

logger = config.logger


# def main():

#     texts_list = corpora_onto.get_DIIM_Ontos_Corpora()
#     # print(texts_list[0])
#     # print('*')
#     # print(texts_list[1])
#     docs = corpora_iot.get_DIIM_IoT_Corpora()
#     for texts in texts_list:
#         # äremove words that appear only once
#         frequency = defaultdict(int)
#         for text in texts:
#             for token in text:
#                 frequency[token] += 1

#         texts = [
#             [token for token in text if frequency[token] > 1]
#             for text in texts
#         ]

#         dictionary = corpora.Dictionary(texts)
#         corpus = [dictionary.doc2bow(text) for text in texts]

#         lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

#         # doc = "Human computer interaction"
#         # vec_bow = dictionary.doc2bow(doc.lower().split())

#         for doc in docs:
#             vec_bow = dictionary.doc2bow(doc)
#             vec_lsi = lsi[vec_bow]  # convert the query to LSI space
#             print(vec_lsi)


# main()


def build_LSI_Models_of_Ontos_and_IoT():

    ontos_corpora = corpora_onto.get_DIIM_Ontos_Corpora()
    # print(texts_list[0])
    # print('*')
    # print(texts_list[1])
    iot_corpora = corpora_iot.get_DIIM_IoT_Corpora()
    onto_index = 0

    for onto_corpus in ontos_corpora:
        print(type(onto_corpus))
        # print(onto_corpus)
        text_tokens, onto_bow_corpus, ontology_dictionary = build_TextTokens_BoWCorpus_Dictionary_for_LSI(
            onto_corpus)

        lsi_model_onto = models.LsiModel(
            onto_bow_corpus, id2word=ontology_dictionary)
        ontology_name = corpora_onto.get_Ontology_Names()[onto_index]
        print('Printing topics of LSI model ', ontology_name)
        onto_index += 1
        pprint(lsi_model_onto.print_topics())
        print()

        iot_index = 0
        for iot_corpus in iot_corpora:
            iot_name = corpora_iot.get_IoT_Names()[iot_index]

            text_tokens, iot_bow_corpus, iot_dictionary = build_TextTokens_BoWCorpus_Dictionary_for_LSI(
                iot_corpus)

            iot_vec_bow = iot_dictionary.doc2bow(iot_corpus)
            # convert the query to LSI space
            vec_lsi = lsi_model_onto[iot_vec_bow]
            # print('\nPrinting vec lsi Query')
            # print(vec_lsi)

            index = similarities.MatrixSimilarity(
                lsi_model_onto[onto_bow_corpus])
            sims = index[vec_lsi]
            print('\nPrinting LSI Similarity : ', ontology_name, iot_name)
            # pprint(sims)
            # sort the similarity with the highest value first
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            # print(list(enumerate(sims)))
            logger.info("printing sorted list ...")
            print("\nPrinting sorted list ...")
            for pos, (doc_position, doc_score) in list(enumerate(sims)):
                # print(sim)
                # for doc_position, doc_score in sim:
                # pprint(doc_position, doc_score)
                # logger.info(str(doc_score), str(doc_position))
                print("Ontology " + str(ontology_dictionary[doc_position]) +
                      " has similarity " + str(doc_score) + " to IoT Data ")
                # print_LSI_topics_of_merged_Onto_IoT_LSI(
                #    lsi_model_onto, iot_bow_corpus)


def print_LSI_topics_of_merged_Onto_IoT_LSI(lsi_model_onto, iot_bow_corpus):
    lsi_model_onto.add_documents(iot_bow_corpus)
    # pprint(lsi_model.print_topics())
    lsi_topics = lsi_model_onto.show_topics(num_words=10)

    topics = []
    filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]

    for topic in lsi_topics:
        # print(topic)
        topics.append(preprocess_string(topic[1], filters))

    pprint(topics)
    print('-------------')


def build_TextTokens_BoWCorpus_Dictionary_for_LSI(diim_corpus):
    text_tokens = []
    # It is a list of lists. A list contains tokens as strings
    # e.g.:
    # [['human', 'interface', 'computer'],
    # ['trees'],
    # ['graph', 'trees'],
    # ......
    # ['graph', 'minors', 'survey']]
    for text in diim_corpus:
        tokens = text.split()
        text_tokens.append(tokens)

    dictionary = corpora.Dictionary(text_tokens)

    # new_doc = "Human computer interaction"
    # new_corpus = new_doc.lower().split()
    # print(type(new_corpus))
    # print(new_corpus)
    # vec_bow = dictionary.doc2bow(new_corpus)
    # print(vec_bow)

    bow_corpus = [dictionary.doc2bow(text) for text in text_tokens]

    return text_tokens, bow_corpus, dictionary


def calculate_LSI_Similarities_of_Ontos_and_IoT(tfidf_onto, tfidf_iot):

    index = similarities.SparseMatrixSimilarity(tfidf_onto, num_features=12)
    sims = index[tfidf_iot]
    pprint(sims)


# def test():
#     new_doc = "Human computer interaction"

#     new_vec = dictionary.doc2bow(new_doc.lower().split())
#     print(new_vec)
