# Import spaCy
from pathlib import Path
import numpy as np
from spacy.tokens import Doc
import spacy
# Import Matcher
from spacy.matcher import Matcher
# Import the Doc and Span classes
from spacy.tokens import Doc, Span
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os

# local imports
import DIIM_config as config

# Logging
logger = logging.getLogger(config.LOGAPPLICATION_NAME)
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

PROJECT_DIR = os.getcwd() + "\\"

#####################
# IoT Data related variables
#############################
iot_data_dict = {}  # set of iot data documents, which will be used to find similarity
iot_data_dict_counter = 0  # counter for iot data documents
IOT_DATASET_PATH = PROJECT_DIR + 'res\\data\\input\\iot\\'
IoT_dataset = {}  # key is filename, value is iot data in the file

# Test Data
sentence = "WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement."
iot_data = "Meta data: Name:S1, Description: The sensor measures water temperature in Fahrenheit, serial:00-14-22-01-23-45, model: BFG9000, mac:50:8c:b1:77:e8:e6, latitude:51.75543, longitude:-1.03248 Measurement data:4baa-a2ff-8741efad4e63: temp:timestamp:2021-08-09T17:01:28.796Z, values: value :20, timestamp:2021-08-09T17:01:38.792Z, values:value:24"


def some_matcher_test():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(iot_data.lower())
    for token in doc:
        if token.pos_ == "PROPN":
            print(token)

    # Initialize the Matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)
    # Create a pattern matching
    pattern = [
        # {"IS_DIGIT": True},
        {"LOWER": "temperature"},
        # {"LOWER": "Fahrenheit"},
        # {"LOWER": "sensor"},
        # {"IS_PUNCT": True}
    ]

    # Use the matcher on the doc
    matches = matcher(doc)
    print("Matches:", [doc[start:end].text for match_id,
          start, end in matches])

    # doc = nlp("Get busy living or get busy dying.")


def analyze(data):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(data)

    print(f"{'Count':{5}} {'text':{30}} {'POS':{6}} {'TAG':{6}} {'Dep':{6}} {'POS explained':{20}} {'tag explained'} ")
    counter = 0
    for token in doc:
        if (token.pos_ == 'PROPN' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'ADP' or token.pos_ == 'VERB'):
            counter += 1
            print(f'{counter:{5}} {token.text:{30}} {token.pos_:{6}} {token.tag_:{6}} {token.dep_:{6}} {spacy.explain(token.pos_):{20}} {spacy.explain(token.tag_)}')


def is_Temperature_Sensor_Data(data):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(data)
    # Initialize the Matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)

    # Create a pattern matching two tokens: "iPhone" and "X"
    # pattern = [{"TEXT": "sensor"}, {"TEXT": "temperature"}]
    pattern = [{"TEXT": "sensor"}, {"TEXT": "temperature"}]

    # Add the pattern to the matcher
    matcher.add("temperature_sensor", [pattern])

    # Use the matcher on the doc
    matches = matcher(doc)
    print("Matches:", [doc[start:end].text for match_id,
          start, end in matches])


def analyse_doc(document):
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000000
    # or any large value, as long as you don't run out of RAM
    #doc_strs = document.split('\n')
    logger.info('Starting NLP analyze of data ..')
    #doc = nlp.pipe(doc_strs)
    doc = nlp(document)
    logger.info('NLP analyze of data done!')
    # tag_lst = nlp.pipe_labels['tagger']
    # pos_lst = nlp.pipe_labels['ner']
    # print(pos_lst)
    # print(len(tag_lst))
    # for tag in tag_lst:
    #     print(tag, spacy.explain(tag))

    # # Counting the frequencies of different POS tags:
    POS_counts = doc.count_by(spacy.attrs.POS)
    logger.info(POS_counts)
    POS_counts_data = {}
    skip_count = 0
    for k, v in sorted(POS_counts.items()):
        # print(f'{k:{4}}. {doc.vocab[k].text:{5}}: {v}')
        if (doc.vocab[k].text == 'SPACE' or doc.vocab[k].text == 'PUNCT' or doc.vocab[k].text == 'NUM'):
            skip_count += 1
        else:
            POS_counts_data[doc.vocab[k].text] = v

    logger.info('POS Conuts data ', POS_counts_data)
    logger.info('No. of POS skipped ', skip_count)
    return POS_counts_data


def visualize_POS_data(filename, data):

    pos_names = data.keys()
    # pos_names.add('NUM')
    pos_counts = data.values()
    # pos_counts.add(20)
    POS_column_name = 'POS of ' + filename
    Values_column_name = 'Values of ' + filename
    data_as_lst = {POS_column_name: pos_names, Values_column_name: pos_counts}
    data_as_df = pd.DataFrame(data_as_lst)
    print(data_as_df)
    # data_as_df.groupby('POS')['name'].nunique().plot(kind='bar')
    data_as_df.plot(kind='barh', x=POS_column_name,
                    title='POS Analysis of ' + filename)
    plt.show()


def visualize_POS_data_with_bar_labels(filename, data):
    # better way to show bars with values
    pos_names = data.keys()
    pos_counts = data.values()

    df = pd.DataFrame({filename: pos_counts}, index=pos_names)
    ax = df.plot.barh()

    ax.bar_label(ax.containers[0])

    plt.show()


def initialize_IoT_dataset():
    logger.info("Initializing IoT datasets from " + IOT_DATASET_PATH)
    global IoT_dataset
    for root, dirs, files in os.walk(IOT_DATASET_PATH):
        # for file in filter(lambda file: file.endswith('.txt'), files):
        # for file in filter(lambda file: file.endswith('.csv') or file.endswith('.json'), files):
        for file in filter(lambda file: file.endswith('.small.csv') or file.endswith('.small.json'), files):
            # for file in filter(lambda file: file.endswith('drinking-water-quality-monitoring.json'), files):
            # read the entire document, as one big string
            document_path = Path(os.path.join(root, file))
            document = open(document_path).read()

            logger.info(
                "Adding " + file + " as key to IoT datasets with document word length " + str(len(document)))
            IoT_dataset[file] = document

    logger.info("IoT_dataset initilized with size " + str(len(IoT_dataset)))
    return IoT_dataset


def run_NER_analysis(document):
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000000

    # Getting the pipeline component
    ner = nlp.get_pipe("ner")
    LABEL = 'MICROBIOLOGICAL_PARAMETERS'
    WQ_TRAIN_DATA = [
        ("Coliform bacteria are organisms that are present in the environment and in \
                  the feces of all warm-blooded animals and humans. Coliform bacteria will not \
                  likely cause illness. However, their presence in drinking water indicates \
                  that disease-causing organisms (pathogens) could be in the water system.",
         {"entities": [(0, 8, LABEL)]}),
        ("Coliform bacteria will not likely cause illness. However, their presence in drinking water indicates \
                  that disease-causing organisms (pathogens) could be in the water system.",
         {"entities": [(0, 8, LABEL)]}),
        ("Coliform bacteria are defined as either motile or non-motile Gram-negative non-spore \
                  forming Bacilli that possess β-galactosidase to produce acids and gases under their \
                  optimal growth temperature of 35-37°C. \
                  They can be aerobes or facultative aerobes, and are a commonly used indicator of sanitary quality of foods and water.",
         {"entities": [(0, 8, LABEL)]}),

        ("Coliforms can be found in the aquatic environment, in soil and on vegetation; \
                  they are universally present in large numbers in the feces of warm-blooded \
                  animals as they are known to inhabit the gastrointestinal system.[1] ",
         {"entities": [(0, 9, LABEL)]}),
        ("While coliform bacteria are not normally causes of serious illness, they are easy to culture,\
                   and their presence is used to infer that other pathogenic organisms of fecal origin \
                   may be present in a sample, or that said sample is not safe to consume",
         {"entities": [(6, 14, LABEL)]}),
    ]

    # Add the new label to ner
    ner.add_label(LABEL)

    # Resume training
    optimizer = nlp.resume_training()
    move_names = list(ner.move_names)

    # List of pipes you want to train
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    # List of pipes which should remain unaffected in training
    other_pipes = [
        pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Adding new labels to the `ner`
    # for _, annotations in WQ_TRAIN_DATA:
    #     for ent in annotations.get("entities"):
    #         ner.add_label(ent[2])
    #         logger.info('Starting NER analyze of data ..')
    #         doc = nlp(document)
    doc = nlp(document)
    for ent in doc.ents:
        print(ent.text, ent.label_)
    logger.info('NER analyze of data done!')


def run_NLP_demo():
    for iot_data_file in IoT_dataset.keys():
        nlp_analsis_data = analyse_doc(IoT_dataset[iot_data_file])
        visualize_POS_data_with_bar_labels(iot_data_file, nlp_analsis_data)


def run_NER_demo():
    for iot_data_file in IoT_dataset.keys():
        print('Entities found in ', iot_data_file)
        run_NER_analysis(IoT_dataset[iot_data_file])


def main():
    initialize_IoT_dataset()
    # run_NLP_demo()  # working
    run_NER_demo()

    # analyze(iot_data)
    # is_Temperature_Sensor_Data(iot_data)
    # analyse_doc(iot_data)
    # analyse_doc(sentence)
    # visualize_POS_data(analyse_doc(sentence))


main()
