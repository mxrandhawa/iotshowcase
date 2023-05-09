import logging
import pandas as pd
import warnings
import bs4 as bs
import urllib.request
import nltk
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models, matutils
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from operator import itemgetter
from spacy import displacy
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
# import local modules
import DIIM_config as config


def parseURL(inputURL, outputFilenname):
    logger = logging.getLogger(config.LOGAPPLICATION_NAME)
    raw_html = urllib.request.urlopen(inputURL)
    raw_html = raw_html.read()
    article_html = bs.BeautifulSoup(raw_html, 'lxml')
    article_paragraphs = article_html.find_all('p')
    article_text = ''
    for para in article_paragraphs:
        article_text += para.text
        # print(article_text)
        # saving the parsed text in txt file
        file = open(outputFilenname, 'w')
        file.write(article_text)
        file.close()
    # End message
    logger.info('Wiki text has been successufully parsed!')


def existsFile(fileName):
    for currentpath, folders, files in os.walk('.'):
        for file in files:
            if (file.endswith(fileName)):
                return True  # file found

    # file not found
    return False


def createCorpus(fileName):
    with open(fileName) as f:
        # put the path to your dataset
        contents = f.read()
        corpus = nltk.sent_tokenize(contents)
    f.close()
    return corpus
#   print(corpus)


def formatCorpus(corpus, cleanedDatasetFilename):
    # Step1: Sentence Segmentation
    for i in range(len(corpus)):
        # Convert text into lower case;
        corpus[i] = corpus[i].lower()
        # Remove empty spaces;
        corpus[i] = re.sub(r'\W', ' ', corpus[i])
        # Remove punctuations;
        corpus[i] = re.sub(r'\s+', ' ', corpus[i])
        # Remove short words, with length equal to 1
        shortword = re.compile(r'\W*\b\w{1}\b')
        corpus[i] = re.sub(shortword, ' ', corpus[i])
    # Step 2: Tokenisation
    # split text into tokens(words) using word_tokenize
    for sentence in corpus:
        tokens = nltk.word_tokenize(sentence)
    # Step 3: Stop words removal
    # list of stopwords
    stop_words = set(stopwords.words('english'))
    # print(stop_words)
    # save the cleaned text to a file
    fileClean = open(cleanedDatasetFilename, 'a')
    for sentence in corpus:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            # remove numeric␣values
            token = ''.join([i for i in token if not i.isdigit()])
            # check for stopwords
            if not token in stop_words:
                # save the non stopwords in the file
                fileClean.write(" " + token)
    fileClean.close()


def getStemmedText(fileName):
    logger = logging.getLogger(config.LOGAPPLICATION_NAME)
    with open(fileName) as f:
        contents = f.read()
    tokens = nltk.word_tokenize(contents)
    # Stemming
    porter = PorterStemmer()
    stem_words = np.vectorize(porter.stem)
    stemed_text = ' '.join(stem_words(tokens))
    logger.info(f"nltk stemed text: {stemed_text}")
    f.close()
    return stemed_text


def getLemmatizedText(fileName):
    logger = logging.getLogger(config.LOGAPPLICATION_NAME)

    with open(fileName) as f:
        contents = f.read()
    tokens = nltk.word_tokenize(contents)
    # Lemmatisation
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatize_words = np.vectorize(wordnet_lemmatizer.lemmatize)
    lemmatized_text = ' '.join(lemmatize_words(tokens))
    logger.info(f"nltk lemmatized text: {lemmatized_text}")
    f.close()
    return lemmatized_text


def doPOSTagging(cleanedDatasetFilename, posTaggedWikiDatasetFileName):
    sp = spacy.load('en_core_web_sm')
    # Open the cleaned text file
    with open(cleanedDatasetFilename) as f:
        contents = f.read()
    '''
    words = []
    for word in sp(contents):
        words.append(word)
    words.sort()
    '''
    # file to store the result of POS tagging
    filePOS = open(posTaggedWikiDatasetFileName, 'w')
    headerStr = str(f'WORD \t\t\t\t' 'TYPE \t' 'ACRYNOM \t' 'DESCRIPTION')
    headerStr += '\n--------------------------------------------------\n'
    # print(headerStr)
    filePOS.write(headerStr)
    for word in sp(contents):
        writeStr = f'{word.text:{12}}\t\t {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}'
      #  print(writeStr)
        filePOS.write(writeStr + '\n')
    filePOS.close()


def visualizePOSTags():
    sp = spacy.load('en_core_web_sm')
    sen = sp(u"Biomedicine also can relate to many other categories in health. It has been the dominant system of medicine in the Western world for more than a century")
    displacy.render(sen, style='dep', jupyter=True, options={'distance': 85})


# This step refers to the identification of words in a sentence as an entity
# e.g. the name of a person, place, organization, etc. The spaCy library is
# used to perform named entity recognition as shown in the following code
def recognizeNamedEntities(cleandDatasetFileName, NERDatasetFilename):
    logger = logging.getLogger(config.LOGAPPLICATION_NAME)
    sp = spacy.load('en_core_web_sm')
    # Open the cleaned text file
    with open(cleandDatasetFileName) as f:
        contents = f.read()
    # file to store the NER result
    fileNER = open(NERDatasetFilename, 'w')
    for entity in sp(contents).ents:
        entityStr = entity.text + ' - ' + entity.label_ + ' - ' + str(spacy.explain(entity.label_))
        logger.info(entityStr)
        fileNER.write(entity.text + '\n')
    f.close()
    fileNER.close()


def extractTermsWithTF_IDF(cleanDatasetFilename):
    logger = logging.getLogger(config.LOGAPPLICATION_NAME)
    with open(cleanDatasetFilename) as f:
        contents = f.read()
    corpus = nltk.sent_tokenize(contents)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    logger.info(vectorizer.get_feature_names_out())
    # print(X.shape)
    tf_idf_model = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
    # save results in a text file
    tFIDFDatasetFilename = cleanDatasetFilename + "TFIDF.txt"
    fileTFIDF = open(tFIDFDatasetFilename, 'w')
    for key, val in tf_idf_model.items():
        # print (str(key) + ':' + str(val))
        fileTFIDF.write(str(key) + ':' + str(val) + '\n')
    fileTFIDF.close()
    # sort the dictionary of words by tf-idf and save the results in a text file
    fileTFIDF_Sorted = open('WikiDatasetTFIDF_Sorted.txt', 'w')
    listofwords = sorted(tf_idf_model.items(), reverse=True, key=lambda x: x[1])
    for elem in listofwords:
        logger.info(elem[0], " ::", elem[1])
        fileTFIDF_Sorted.write(str(elem[0]) + ':' + str(elem[1]) + '\n')
    fileTFIDF_Sorted.close()

# Word Embedding is a language modeling technique used for mapping words to
# vectors of real numbers. It represents words or phrases in vector space
# with several dimensions. The basic idea of word embedding is words that
# occur in similar context tend to be closer to each other in vector space.
# Therefore, word embedding can be used to extract synonyms of the ontology
# terms. Word2Vec consists of models for generating word embedding. These
# models are shallow two layer neural networks having one input layer,
# one hidden layer and one output layer. Word2Vec utilises two architectures:
# CBOW (Continuous Bag of Words) and Skip-Gram. In this activity, we will
# use the Skip_Gram architecture because it has been experimentally proven
# that it performs better that the CBOW architecture. Skip gram predicts
# the surrounding context words within specific window given current word.
# For generating word vectors in Python, module needed is gensim. Run this
# command in terminal to install gensim.


def doWordEmbeddingWithWord2Vec():
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
    logger = logging.getLogger(config.LOGAPPLICATION_NAME)
    input_filename = 'WikiDatasetCleaned.txt'  # text file to train on
    model_filename = 'WikiDatasetCleaned.model'  # name for saving trained model
    # train using skip-gram
    skip_gram = True
    # create vocabulary
    logger.info('building vocabulary...')
    model = models.Word2Vec()
    sentences = models.word2vec.LineSentence(input_filename)
    model.build_vocab(sentences)
    # train model
    logger.info('training model...')
    if skip_gram:
        model.train(sentences, total_examples=model.corpus_count, epochs=3)
    else:
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    # and save the trained model
    logger.info('- saving model...')
    model.save(model_filename)
    # bye message
    logger.info('all done, whew!')
    # find similar words
    logger.info(model.wv.most_similar(config.word, topn=10))

# visualise these similar words using dimensionality reduction algorithms
# such as t-SNE (tdistributed Stochastic Neighbor Embedding). The
# visualisation can be useful to understand how Word2Vec works and how
# to interpret relations between vectors captured from your texts


def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0, 100), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.wv.most_similar(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array(model.wv.__getitem__([word])), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array(model.wv.__getitem__([word])), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # display scatter plot
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset.points')

    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


def clusterConcepts(datasetFilename):
    logger = logging.getLogger(config.LOGAPPLICATION_NAME)
    # from sklearn.cluster import KMeans
    with open(datasetFilename) as f:
        content = f.read()
    corpus = nltk.sent_tokenize(content)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    df_text = pd.DataFrame(X.toarray())
    logger.info(df_text)
    # Creating the model
    agg_clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    # predicting the labels
    labels = agg_clustering.fit_predict(X.toarray())
    # Linkage Matrix
    Z = linkage(X.toarray()[:, 1:20], method='ward')
    # plotting dendrogram
    dendro = dendrogram(Z)
    plt.title('Dendrogram')
    plt.ylabel('Euclidean distance')
    plt.show()
    plt.figure(figsize=(10, 7))
    plt.scatter(X.toarray()[:, 0], X.toarray()[:, 1], c=agg_clustering.labels_, cmap='rainbow')


def clusterConceptsWithK2(wikiDatsetName):
    logger = logging.getLogger(config.LOGAPPLICATION_NAME)
    with open(wikiDatsetName) as f:
        content = f.read()
    corpus = nltk.sent_tokenize(content)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    true_k = 2
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    logger.info("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(true_k):
        logger.info("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            logger.info(' %s' % terms[ind]),
        logger.info()


def runNLPAnalysis(datasetFilename):
    # config.initialize()
    logger = logging.getLogger(config.LOGAPPLICATION_NAME)
    # logger.info the current working directory
    logger.info("Current working directory: {0}".format(os.getcwd()))
    logger.info("Project working directory: {0}".format(config.inputDir))

    # Change the current working directory
    # os.chdir(os.path.join(os.getcwd(), '/src/ex/NLP-Word2Vec'))
    os.chdir(config.inputDir)
    # Print the current working directory
    logger.info("Current working directory: {0}".format(os.getcwd()))

    # if output doesn't exits, parse the url and create a parsed output
    # if (not existsFile(config.datasetFilename)):
    #   parseURL(config.inputURL, config.datasetFilename)

    # Step 0: create Corpus from a text
    # Step 1: Sentence Segmentation
    # Step 2: Tokenisation
    # Step 3: Stop words Removal
    corpus = createCorpus(datasetFilename)

    # Step 4: Lemmatisation and Stemming
    cleanedDatasetFilename = datasetFilename + "CLEAN.txt"
    formatCorpus(corpus, cleanedDatasetFilename)
    logger.info("-------------------------")
    getStemmedText(cleanedDatasetFilename)
    logger.info("-------------------------")
    getLemmatizedText(cleanedDatasetFilename)

    # Step 5: POS Tagging (Part-Of-Speech Tagging)
    posTaggedWikiDatasetFileName = datasetFilename + "POSTag.txt"
    doPOSTagging(cleanedDatasetFilename, posTaggedWikiDatasetFileName)
    visualizePOSTags()

    # Step 6: Named Entity Recognition
    nERDatasetFilename = datasetFilename + "NER.txt"
    recognizeNamedEntities(cleanedDatasetFilename, nERDatasetFilename)

    # Step 7: TF-IDF (Term Frequency – Inverse Document Frequency)
#    extractTermsWithTF_IDF()

    # Step 8: Word Embeddings – Word2Vec
 #   doWordEmbeddingWithWord2Vec()

    # name of the saved trained model

    logger.info('loading the model ...')
    modelFilename = datasetFilename + ".model"
    #model = Word2Vec.load(modelFilename)

    # display the model
    #display_closestwords_tsnescatterplot(model, config.word)

    # visualize clusteering of words
    # hierachical clustering
    # clusterConcepts(datasetFilename)

    # The output of the hierarchical clustering shows that our current text
    # can be devided into two separate clusters. We can then apply K-means
    # with K=2 to know the terms’ clusters.
    # clusterConceptsWithK2(datasetFilename)


def runAnalysis():
    logger = logging.getLogger(config.LOGAPPLICATION_NAME)

    # look up JSONFiles in current dir and analyze them
    fileType = ".txt"
    logger.info('starting with NLP of files ending with ' + fileType)
    for currentpath, folders, files in os.walk(config.inputDir):
        for file in files:
            if (file.endswith(fileType)):
                filePath = os.path.join(currentpath, file)
               # print(filePath)
                logger.info('starting with NLP analysis of file ' + filePath)
                runNLPAnalysis(filePath)
    logger.info('done with NLP)')
