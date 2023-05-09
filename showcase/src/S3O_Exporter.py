##############################
#  DIIM Similarities Ontology (DSO)
#
#############################

# python imports
from ast import Not
from cProfile import label
import sys
from rdflib import Graph
# Import the requirements modules
from rdflib.namespace import OWL, RDF, RDFS, SDO, XSD
from rdflib import Graph, Literal, Namespace, URIRef
import difflib


# local imports
import DIIM_word2vec_utils as w2v_utils
import DIIM_config as config
import S3O_Schema_Exporter as dso_schema

logger = config.logger

DSO_FILEPATH_PREFIX = config.DIIM_SIM_STORE_PATH + '\\DIIM_IoT_Similarities.'
DSO_DEMO_FILEPATH_PREFIX = config.DIIM_SIM_STORE_PATH + \
    '\\DIIM_IoT_Similarities_Demo.'
# DSO_CORRECTED_FILEPATH_PREFIX = config.DIIM_SIM_STORE_PATH + \
#    '\\DIIM_IoT_Similarities_Corrected.'

DIIM_RDF_XML_FORMAT = "xml"
DIIM_RDF_TURTTLE_FORMAT = "ttl"
DIIM_RDF_ALL = 'all'

DSO_URL = 'http://www.iot4win.org.uk/diim/ontologies/dso'
DSO_NAMESPACE = DSO_URL + '#'


# print(RDF_HEAD)
graph = Graph()
# create diim namespace
DSON = Namespace(DSO_NAMESPACE)
# Bind prefix to namespace to make it more readable
graph.bind('dso', DSON)
graph.bind('schema', SDO)
graph.bind('owl', OWL)

#################################
# Entities of DIIM Similarity Ontology (DSO)
#################################
# Classes in DSO
DSO_Term = 'Term'
DSO_IoT_Data = 'IoT_Data'
DSO_Ontology = 'Ontology'
DSO_Similarity = 'Similarity'
DSO_W2V_Similarity = 'W2V_Similarity'
DSO_SSM_Similarity = 'SSM_Similarity'


# Object properties in DSO
DSO_uses_Term = 'uses_Term'
DSO_term_Used_by = 'term_Used_by'
# DSO_uses_Ontology = 'uses_Ontology'
DSO_has_Similarity = 'has_Similarity'
DSO_ref_Term = 'ref_Term'
DSO_ref_Ontology = 'ref_Ontology'
DSO_ref_IoT_Data = 'ref_IoT_Data'

# Data properties in DSO
DSO_serialization_format = 'serialization_format'
DSO_similarity_value = 'similarity_value'
DSO_uri = 'uri'


def save_DSO_Similarities_of_IOT_with_Ontos(diim_Similarities_of_IOT_with_Ontos, format, similarity_nodes_limit):
    dso_schema.create_S3O_schema_entities()
    # fix_DSO_DataObjectProperty_issue()

    # add individuals of similarities
    for onto_name_iot_name in diim_Similarities_of_IOT_with_Ontos:
        diim_Similarities_of_IOT_with_Onto = diim_Similarities_of_IOT_with_Ontos[
            onto_name_iot_name]

        ontology_name, iot_name, label = w2v_utils.split_ontology_iot_word_name(
            onto_name_iot_name)

        # print(ontology_name)
        # print(iot_name)
        # create IoT_Data individual
        iot_node = DSON[iot_name]
        graph.add((iot_node, RDF.type, DSON.IoT_Data))

        # create Ontology individual
        onto_node = DSON[ontology_name]
        graph.add((onto_node, RDF.type, DSON.Ontology))

        # create similarity nodes  for ontology x iot
        create_similarity_nodes(
            diim_Similarities_of_IOT_with_Onto, iot_node, onto_node, similarity_nodes_limit)

    # save graph
    if similarity_nodes_limit != sys.maxsize:
        save_graph_demo(format)
    else:
        save_graph(format)


def create_similarity_nodes(diim_Similarities_of_IOT_with_Onto, iot_node, onto_node, similarity_nodes_limit):
    nr_of_similarity_nodes = 0

    for onto_name_iot_name_label in diim_Similarities_of_IOT_with_Onto:
        if nr_of_similarity_nodes > similarity_nodes_limit:
            return
        nr_of_similarity_nodes = nr_of_similarity_nodes + 1

        # logger.info('spliting ', onto_name_iot_name_label)
        ontology_name, iot_name, label = w2v_utils.split_ontology_iot_word_name(
            onto_name_iot_name_label)

        # create Term individual for label
        label_word_node = DSON[label]
        graph.add((label_word_node, RDF.type, DSON.Term))

        # create relation between label and IoT Data
        graph.add((iot_node, DSON.uses_Term, label_word_node))
        # add Ontology to Term
        graph.add((label_word_node, DSON.term_Used_in, iot_node))

        similarities_list = diim_Similarities_of_IOT_with_Onto[onto_name_iot_name_label]

        for sim_word, sim_value in similarities_list:
            # logger.info(str(ontology_name), str(sim_word), str(iot_name), str(label))
            if sim_word is None:
                continue
            else:
                # create a Word2Vec similarity node based on the values from word2vec models
                create_similarity_node(DSO_W2V_Similarity, iot_node, label_word_node,
                                       onto_node, ontology_name, sim_word, iot_name, label, sim_value)

                # create a String Sequence Match similarity node based on the string matching pattern manually
                create_StringSequenceMatcher_similarity_node(
                    iot_node, label_word_node, onto_node, ontology_name, sim_word, iot_name, label)


def create_StringSequenceMatcher_similarity_node(iot_node, label_word_node,
                                                 onto_node, ontology_name, sim_word, iot_name, label):
    # # create a similarity node with similarity_value 1 if the label exists itself in the ontology
    if (label_word_node, DSON.term_Used_in, onto_node) in graph and (label_word_node, DSON.term_Used_in, iot_node) in graph:
        logger.info(label_word_node, DSON.term_Used_in, onto_node)
        logger.info(label_word_node, DSON.term_Used_in, iot_node)

    #     # create W2V_Similarity_Term individual
    #     # sim_name = ontology_name + '_' + label + '_' + iot_name + '_' + label
    #     # sim_node = DSON[sim_name]
    #     # graph.add((sim_node, RDF.type, DSON.W2V_Similarity))
        str_seq_match_val = 1.0
        logger.info('INPUT:', DSO_SSM_Similarity, iot_node,
              label_word_node, onto_node, ontology_name, label, iot_name, label, str_seq_match_val)
        node = create_similarity_node(DSO_SSM_Similarity, iot_node,
                                      label_word_node, onto_node, ontology_name, label, iot_name, label, str_seq_match_val)
        logger.info(node)
        logger.info('')
    # else:
    seq = difflib.SequenceMatcher(None, label, sim_word)
    str_seq_match_val = seq.ratio()
    #print(label, sim_word, str_seq_match_val)
    create_similarity_node(DSO_SSM_Similarity, iot_node,
                           label_word_node, onto_node, ontology_name, sim_word, iot_name, label, str_seq_match_val)


def create_similarity_node(sim_type, iot_node, label_word_node, onto_node, ontology_name, sim_word, iot_name, label, sim_value):

    if (sim_type == DSO_SSM_Similarity):
        # create W2V_Similarity_Term individual
        sim_name = 'SSM' + '_' + ontology_name + \
            '_' + sim_word + '_' + iot_name + '_' + label
        sim_node = DSON[sim_name]
        graph.add((sim_node, RDF.type, DSON.SSM_Similarity))

    elif (sim_type == DSO_W2V_Similarity):
        # create W2V_Similarity_Term individual
        sim_name = 'W2V' + '_' + ontology_name + \
            '_' + sim_word + '_' + iot_name + '_' + label
        sim_node = DSON[sim_name]
        graph.add((sim_node, RDF.type, DSON.W2V_Similarity))

    # create Term individual for sim_word
    sim_word_node = DSON[sim_word]
    graph.add((sim_word_node, RDF.type, DSON.Term))

    # create relation (has_Similarity) between terms and sim_node
    # create has_Similarity relation for label_word
    graph.add(
        (label_word_node, DSON.has_Similarity, sim_node))
    # create has_Similarity relation for sim_word_node
    graph.add(
        (sim_word_node, DSON.has_Similarity, sim_node))

    # create relation (ref_Term) between sim_node and terms
    graph.add((sim_node, DSON.ref_Term, label_word_node))
    graph.add((sim_node, DSON.ref_Term, sim_word_node))

    # add the value of similarity to the similairty node
    sim_word_value = Literal(sim_value, datatype=XSD['float'])
    graph.add(
        (sim_node, DSON.similarity_value, sim_word_value))

    # add ref to ontology
    graph.add(
        (sim_node, DSON.ref_Ontology, onto_node))

    # create ref_IoT_Data between sim_node and IoT Data
    graph.add(
        (sim_node, DSON.ref_IoT_Data, iot_node))

    # # add the label of the similarity to the similairty object
    # sim_word_label = DSON[sim_word]
    # graph.add((sim_word_node, DSON.Label, sim_word_label))

    create_term_onto_uses_relation(onto_node, sim_word_node)

    return sim_node


def create_term_onto_uses_relation(onto_node, sim_word_node):
    # maybe adding it is not necessary
    # Add bi-directinal relation between ontology and similar word
    # add Ontology uses similar word
    graph.add((onto_node, DSON.uses_Term, sim_word_node))
    # add similar word  is used in Ontology
    graph.add((sim_word_node, DSON.term_Used_in, onto_node))


def save_graph_demo(format):

    if format == DIIM_RDF_TURTTLE_FORMAT:
        filepath = DSO_DEMO_FILEPATH_PREFIX + DIIM_RDF_TURTTLE_FORMAT
        graph.serialize(filepath, format=format, encoding="utf-8")
        logger.info("Graph saved: ", filepath, format, "utf-8")
        
    elif format == DIIM_RDF_XML_FORMAT:
        filepath = DSO_DEMO_FILEPATH_PREFIX + DIIM_RDF_XML_FORMAT
        graph.serialize(filepath, format=format, encoding="utf-8")
        logger.info("Graph saved: ", filepath, format, "utf-8")

    elif format == DIIM_RDF_ALL:
        save_graph_demo(DIIM_RDF_TURTTLE_FORMAT)
        save_graph_demo(DIIM_RDF_XML_FORMAT)

    else:
        logger.warn(format, ' is unknown for saving DIIM Graph!')


def save_graph(format):

    if format == DIIM_RDF_TURTTLE_FORMAT:
        filepath = DSO_FILEPATH_PREFIX + DIIM_RDF_TURTTLE_FORMAT
        graph.serialize(filepath, format=format, encoding="utf-8")

    elif format == DIIM_RDF_XML_FORMAT:
        filepath = DSO_FILEPATH_PREFIX + DIIM_RDF_XML_FORMAT
        graph.serialize(filepath, format=format, encoding="utf-8")

    elif format == DIIM_RDF_ALL:
        save_graph(DIIM_RDF_TURTTLE_FORMAT)
        save_graph(DIIM_RDF_XML_FORMAT)

    else:
        logger.warn(format, ' is unknown for saving DIIM Graph!')
