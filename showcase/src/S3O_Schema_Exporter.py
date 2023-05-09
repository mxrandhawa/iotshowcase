##############################DSO
#  DIIM Similarities Ontology (S3O)
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

logger = config.logger

S3O_SCHEMA_FILEPATH_PREFIX = config.DIIM_SIM_KG_STORE_PATH + '\\S3O_Schema'

DIIM_RDF_XML_FORMAT = "xml"
DIIM_RDF_TURTTLE_FORMAT = "ttl"
DIIM_RDF_ALL = 'all'

S3O_SCHEMA_FILEPATH_TURTTLE = S3O_SCHEMA_FILEPATH_PREFIX + \
    '.' + DIIM_RDF_TURTTLE_FORMAT
S3O_SCHEMA_FILEPATH_XML = S3O_SCHEMA_FILEPATH_PREFIX + '.' + DIIM_RDF_XML_FORMAT

S3O_URL = 'http://www.iot4win.org.uk/diim/ontologies/s3o'
S3O_NAMESPACE = S3O_URL + '#'


# print(RDF_HEAD)
graph = Graph()
# create diim namespace
S3ON = Namespace(S3O_NAMESPACE)
# Bind prefix to namespace to make it more readable
graph.bind('s3o', S3ON)
graph.bind('schema', SDO)
graph.bind('owl', OWL)

#################################
# Entities of DIIM Similarity Ontology (S3O)
#################################
# Classes in S3O
S3O_Term = 'Term'
S3O_IoT_Data = 'IoT_Data'
S3O_Ontology = 'Ontology'
S3O_Similarity = 'Similarity'
S3O_W2V_Similarity = 'W2V_Similarity'
S3O_SSM_Similarity = 'SSM_Similarity'


# Object properties in S3O
S3O_uses_Term = 'uses_Term'
S3O_term_Used_by = 'term_Used_by'
# S3O_uses_Ontology = 'uses_Ontology'
S3O_has_Similarity = 'has_Similarity'
S3O_ref_Term = 'ref_Term'
S3O_ref_Ontology = 'ref_Ontology'
S3O_ref_IoT_Data = 'ref_IoT_Data'

# Data properties in S3O
S3O_serialization_format = 'serialization_format'
S3O_similarity_value = 'similarity_value'
S3O_uri = 'uri'


def add_S3O_class_node(node):
    # Add node to the graph
    graph.add((node, RDF.type, OWL.Class))
    graph.add((node, RDFS.subClassOf, OWL.Thing))


def add_S3O_objectproperty_node(node):
    # Add node to the graph
    graph.add((node, RDF.type, OWL.ObjectProperty))
    # graph.add((node, RDFS.subClassOf, OWL.Thing))


def add_term_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_Term])
    add_S3O_class_node(node)


def add_ontology_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_Ontology])
    add_S3O_class_node(node)


def add_iotdata_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_IoT_Data])
    add_S3O_class_node(node)


def add_similarity_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_Similarity])
    add_S3O_class_node(node)


def add_w2v_similarity_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_W2V_Similarity])
    # Add node to the graph
    graph.add((node, RDF.type, OWL.Class))

    # add as sub class of the w2v_similarity
    sim_node = URIRef(S3ON[S3O_Similarity])
    graph.add((node, RDFS.subClassOf, sim_node))


def add_ssm_similarity_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_SSM_Similarity])
    # Add node to the graph
    graph.add((node, RDF.type, OWL.Class))

    # add as sub class of the w2v_similarity
    sim_node = URIRef(S3ON[S3O_Similarity])
    graph.add((node, RDFS.subClassOf, sim_node))


def create_S3O_classes():
    # add S3O classes
    add_term_node()
    add_iotdata_node()
    add_ontology_node()
    add_similarity_node()
    add_w2v_similarity_node()
    add_ssm_similarity_node()


################################
# S3O Object properties
################################

def add_has_similarity_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_has_Similarity])
    add_S3O_objectproperty_node(node)

    # add domain
    term_node = URIRef(S3ON[S3O_Term])
    graph.add((node, RDFS.domain, term_node))

    # add range
    sim_node = URIRef(S3ON[S3O_Similarity])
    graph.add((node, RDFS.range, sim_node))


def add_uses_term_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_uses_Term])
    add_S3O_objectproperty_node(node)

    # no more required
    # add domain
    # sim_node = URIRef(S3ON[S3O_Similarity])
    # graph.add((node, RDFS.domain, sim_node))

    # add domain
    iot_node = URIRef(S3ON[S3O_IoT_Data])
    graph.add((node, RDFS.domain, iot_node))

    # add domain
    onto_node = URIRef(S3ON[S3O_Ontology])
    graph.add((node, RDFS.domain, onto_node))

    # add range
    term_node = URIRef(S3ON[S3O_Term])
    graph.add((node, RDFS.range, term_node))

    # add inverse
    inverse_node = URIRef(S3ON[S3O_term_Used_by])
    graph.add((node, OWL.inverseOf, inverse_node))


def add_term_used_by_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_term_Used_by])
    add_S3O_objectproperty_node(node)

    # add domain
    term_node = URIRef(S3ON[S3O_Term])
    graph.add((node, RDFS.domain, term_node))

    # no more required
    # # add range
    # sim_node = URIRef(S3ON[S3O_Similarity])
    # graph.add((node, RDFS.range, sim_node))

    # add domain
    iot_node = URIRef(S3ON[S3O_IoT_Data])
    graph.add((node, RDFS.range, iot_node))

    # add range
    onto_node = URIRef(S3ON[S3O_Ontology])
    graph.add((node, RDFS.range, onto_node))

    # add inverse
    inverse_node = URIRef(S3ON[S3O_uses_Term])
    graph.add((node, OWL.inverseOf, inverse_node))

def add_ref_ontology_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_ref_Ontology])
    add_S3O_objectproperty_node(node)

    # graph.add(node, RDF.type, OWL.DatatypeProperty)

    # add domain
    sim_node = URIRef(S3ON[S3O_Similarity])
    graph.add((node, RDFS.domain, sim_node))

    # add range
    onto_node = URIRef(S3ON[S3O_Ontology])
    graph.add((node, RDFS.range, onto_node))


def add_ref_iot_data_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_ref_IoT_Data])
    add_S3O_objectproperty_node(node)

    # graph.add(node, RDF.type, OWL.DatatypeProperty)

    # add domain
    sim_node = URIRef(S3ON[S3O_Similarity])
    graph.add((node, RDFS.domain, sim_node))

    # add range
    iot_node = URIRef(S3ON[S3O_IoT_Data])
    graph.add((node, RDFS.range, iot_node))


def add_ref_term_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_ref_Term])
    add_S3O_objectproperty_node(node)

    # graph.add(node, RDF.type, OWL.DatatypeProperty)

    # add domain
    sim_node = URIRef(S3ON[S3O_Similarity])
    graph.add((node, RDFS.domain, sim_node))

    # add range
    term_node = URIRef(S3ON[S3O_Term])
    graph.add((node, RDFS.range, term_node))

#################################################################
#
#    Data properties
#
#################################################################


def add_serialization_format_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_serialization_format])
    add_S3O_datatypeproperty(node)

#    graph.add(node, RDF.type, OWL.DatatypeProperty)

    # add domain
    graph.add((node, RDFS.domain, OWL.Thing))


def add_S3O_datatypeproperty(node):
    graph.add((node, RDF.type, OWL.DatatypeProperty))


def add_similarity_value_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_similarity_value])
    add_S3O_datatypeproperty(node)

    # graph.add(node, RDF.type, OWL.DatatypeProperty)

    # add domain
    sim_node = URIRef(S3ON[S3O_Similarity])
    graph.add((node, RDFS.domain, sim_node))





def add_uri_node():
    # Create the node to add to the Graph
    node = URIRef(S3ON[S3O_uri])
    add_S3O_datatypeproperty(node)

    # graph.add(node, RDF.type, OWL.DatatypeProperty)

    # add domain
    graph.add((node, RDFS.domain, OWL.Thing))


def create_S3O_data_properties():
    add_similarity_value_node()
    add_uri_node()
    add_serialization_format_node()
   


def create_S3O_object_properties():
    # add S3O object properties
    add_has_similarity_node()
    add_uses_term_node()
    add_term_used_by_node()
    
    # for similarity node
    add_ref_iot_data_node()
    add_ref_ontology_node()
    add_ref_term_node()


def create_S3O_schema_entities():
    # add classes
    create_S3O_classes()
    # add object properties
    create_S3O_object_properties()
    create_S3O_data_properties()

    # add S3O domain range relations
    # create_domain_range_of_entities()

    # print(graph.serialize(format=DIIM_RDF_TURTTLE_FORMAT))

    # iot_node = S3ON['IoTData']
    # #
    # graph.add((iot_node, RDF.type,  OWL.Class))
    # print(graph)

    # has_Similar_Term_node = S3ON['has_Similar_Term']
    # # s3o:has_Similar_Term rdf:type owl:ObjectProperty ;
    # graph.add(has_Similar_Term_node, RDF.type, OWL.ObjectProperty)

    # # rdfs: range s3o: Similar_Term
    # graph.add(has_Similar_Term_node, RDF.range, OWL.ObjectProperty)

    # # Turtle repesenation is below
    # # rdfs: domain s3o: Term .
    # graph.add(has_Similar_Term_node, RDF.domain, S3ON.Term)


# def fix_S3O_DataObjectProperty_issue():

#     save_graph(format)
#     # remove the dataobject properties which are created as Object properties
#     # remove_replaced_DataObjectProperties()

#     replace_object_with_data_in_properties(
#         S3O_FILEPATH_PREFIX + DIIM_RDF_TURTTLE_FORMAT, S3O_CORRECTED_FILEPATH_PREFIX + DIIM_RDF_TURTTLE_FORMAT)
#     load_S3O(S3O_CORRECTED_FILEPATH_PREFIX + DIIM_RDF_TURTTLE_FORMAT)


# def save_S3O_Similarities_of_IOT_with_Ontos(diim_Similarities_of_IOT_with_Ontos, format, similarity_nodes_limit):
#     create_S3O_schema_entities()
#     # fix_S3O_DataObjectProperty_issue()

#     # add individuals of similarities
#     for onto_name_iot_name in diim_Similarities_of_IOT_with_Ontos:
#         diim_Similarities_of_IOT_with_Onto = diim_Similarities_of_IOT_with_Ontos[
#             onto_name_iot_name]

#         ontology_name, iot_name, label = w2v_utils.split_ontology_iot_word_name(
#             onto_name_iot_name)

#         # print(ontology_name)
#         # print(iot_name)
#         # create IoT_Data individual
#         iot_node = S3ON[iot_name]
#         graph.add((iot_node, RDF.type, S3ON.IoT_Data))

#         # create Ontology individual
#         onto_node = S3ON[ontology_name]
#         graph.add((onto_node, RDF.type, S3ON.Ontology))

#         # create similarity nodes  for ontology x iot
#         create_similarity_nodes(
#             diim_Similarities_of_IOT_with_Onto, iot_node, onto_node, similarity_nodes_limit)

#     # save graph
#     if similarity_nodes_limit != sys.maxsize:
#         save_graph_demo(format)
#     else:
#         save_graph(format)


# def create_similarity_nodes(diim_Similarities_of_IOT_with_Onto, iot_node, onto_node, similarity_nodes_limit):
#     nr_of_similarity_nodes = 0

#     for onto_name_iot_name_label in diim_Similarities_of_IOT_with_Onto:
#         if nr_of_similarity_nodes > similarity_nodes_limit:
#             return
#         nr_of_similarity_nodes = nr_of_similarity_nodes + 1

#         # logger.info('spliting ', onto_name_iot_name_label)
#         ontology_name, iot_name, label = w2v_utils.split_ontology_iot_word_name(
#             onto_name_iot_name_label)

#         # create Term individual for label
#         label_word_node = S3ON[label]
#         graph.add((label_word_node, RDF.type, S3ON.Term))

#         # create relation between label and IoT Data
#         graph.add((iot_node, S3ON.uses_Term, label_word_node))
#         # add Ontology to Term
#         graph.add((label_word_node, S3ON.term_Used_in, iot_node))

#         similarities_list = diim_Similarities_of_IOT_with_Onto[onto_name_iot_name_label]

#         for sim_word, sim_value in similarities_list:
#             # logger.info(str(ontology_name), str(sim_word), str(iot_name), str(label))
#             if sim_word is None:
#                 continue
#             else:
#                 # create a Word2Vec similarity node based on the values from word2vec models
#                 create_similarity_node(S3O_W2V_Similarity, iot_node, label_word_node,
#                                        onto_node, ontology_name, sim_word, iot_name, label, sim_value)

#                 # create a String Sequence Match similarity node based on the string matching pattern manually
#                 create_StringSequenceMatcher_similarity_node(
#                     iot_node, label_word_node, onto_node, ontology_name, sim_word, iot_name, label)


# def create_StringSequenceMatcher_similarity_node(iot_node, label_word_node,
#                                                  onto_node, ontology_name, sim_word, iot_name, label):
#     # create a similarity node with similarity_value 1 if the label exists itself in the ontology
#     if (label_word_node, S3ON.term_Used_in, onto_node) in graph and (label_word_node, S3ON.term_Used_in, iot_node) in graph:
#         # create W2V_Similarity_Term individual
#         # sim_name = ontology_name + '_' + label + '_' + iot_name + '_' + label
#         # sim_node = S3ON[sim_name]
#         # graph.add((sim_node, RDF.type, S3ON.W2V_Similarity))
#         str_seq_match_val = 1.0
#         create_similarity_node(S3O_SSM_Similarity, iot_node,
#                                label_word_node, onto_node, ontology_name, label, iot_name, label, str_seq_match_val)

#     else:
#         seq = difflib.SequenceMatcher(None, label, sim_word)
#         str_seq_match_val = seq.ratio()
#     print(label, sim_word, str_seq_match_val)
#     create_similarity_node(S3O_SSM_Similarity, iot_node,
#                            label_word_node, onto_node, ontology_name, sim_word, iot_name, label, str_seq_match_val)


# def create_similarity_node(sim_type, iot_node, label_word_node, onto_node, ontology_name, sim_word, iot_name, label, sim_value):

#     if (sim_type == S3O_SSM_Similarity):
#         # create W2V_Similarity_Term individual
#         sim_name = 'SSM' + '_' + ontology_name + \
#             '_' + sim_word + '_' + iot_name + '_' + label
#         sim_node = S3ON[sim_name]
#         graph.add((sim_node, RDF.type, S3ON.SSM_Similarity))

#     elif (sim_type == S3O_W2V_Similarity):
#         # create W2V_Similarity_Term individual
#         sim_name = 'W2V' + '_' + ontology_name + \
#             '_' + sim_word + '_' + iot_name + '_' + label
#         sim_node = S3ON[sim_name]
#         graph.add((sim_node, RDF.type, S3ON.W2V_Similarity))

#     # create Term individual for sim_word
#     sim_word_node = S3ON[sim_word]
#     graph.add((sim_word_node, RDF.type, S3ON.Term))

#     # create relation (has_Similarity) between terms and sim_node
#     # create has_Similarity relation for label_word
#     graph.add(
#         (label_word_node, S3ON.has_Similarity, sim_node))
#     # create has_Similarity relation for sim_word_node
#     graph.add(
#         (sim_word_node, S3ON.has_Similarity, sim_node))

#     # create relation (ref_Term) between sim_node and terms
#     graph.add((sim_node, S3ON.ref_Term, label_word_node))
#     graph.add((sim_node, S3ON.ref_Term, sim_word_node))

#     # add the value of similarity to the similairty node
#     sim_word_value = Literal(sim_value, datatype=XSD['float'])
#     graph.add(
#         (sim_node, S3ON.similarity_value, sim_word_value))

#     # add ref to ontology
#     graph.add(
#         (sim_node, S3ON.ref_Ontology, onto_node))

#     # create ref_IoT_Data between sim_node and IoT Data
#     graph.add(
#         (sim_node, S3ON.ref_IoT_Data, iot_node))

#     # # add the label of the similarity to the similairty object
#     # sim_word_label = S3ON[sim_word]
#     # graph.add((sim_word_node, S3ON.Label, sim_word_label))

#     create_term_onto_uses_relation(onto_node, sim_word_node)

#     return sim_node


# def create_term_onto_uses_relation(onto_node, sim_word_node):
#     # maybe adding it is not necessary
#     # Add bi-directinal relation between ontology and similar word
#     # add Ontology uses similar word
#     graph.add((onto_node, S3ON.uses_Term, sim_word_node))
#     # add similar word  is used in Ontology
#     graph.add((sim_word_node, S3ON.term_Used_in, onto_node))


# def save_graph_demo(format):

#     if format == DIIM_RDF_TURTTLE_FORMAT:
#         filepath = S3O_DEMO_FILEPATH_PREFIX + DIIM_RDF_TURTTLE_FORMAT
#         graph.serialize(filepath, format=format, encoding="utf-8")

#     elif format == DIIM_RDF_XML_FORMAT:
#         filepath = S3O_DEMO_FILEPATH_PREFIX + DIIM_RDF_XML_FORMAT
#         graph.serialize(filepath, format=format, encoding="utf-8")

#     elif format == DIIM_RDF_ALL:
#         save_graph_demo(DIIM_RDF_TURTTLE_FORMAT)
#         save_graph_demo(DIIM_RDF_XML_FORMAT)

#     else:
#         logger.warn(format, ' is unknown for saving DIIM Graph!')


def save_S3O_schema(format):

    if format == DIIM_RDF_TURTTLE_FORMAT:
        filepath = S3O_SCHEMA_FILEPATH_PREFIX + DIIM_RDF_TURTTLE_FORMAT
        graph.serialize(filepath, format=format, encoding="utf-8")

    elif format == DIIM_RDF_XML_FORMAT:
        filepath = S3O_SCHEMA_FILEPATH_PREFIX + DIIM_RDF_XML_FORMAT
        graph.serialize(filepath, format=format, encoding="utf-8")

    elif format == DIIM_RDF_ALL:
        save_S3O_schema(DIIM_RDF_TURTTLE_FORMAT)
        save_S3O_schema(DIIM_RDF_XML_FORMAT)

    else:
        logger.warn(format, ' is unknown for saving DIIM Graph!')


def load_S3O(filepath):
    if filepath:
        graph.parse(filepath)
    counter = 0
    for s, o, p in graph:
        # print(s, o, p)
        counter = counter + 1
    print('Loading sucessfull, loaded *', str(len(graph)), '* statements')


# def replace_line(line, serialization_format_flag, uri_flag, similarity_value_flag):

#     if serialization_format_flag and line == 's3o:serialization_format a owl:ObjectProperty ;\n':
#         line = line.replace('s3o:serialization_format a owl:ObjectProperty ;\n',
#                             's3o:serialization_format a owl:DatatypeProperty ;\n')
#         serialization_format_flag = False  # done
#     elif uri_flag and line == 's3o:uri a owl:ObjectProperty ;\n':
#         line = line.replace('s3o:uri a owl:ObjectProperty ;\n',
#                             's3o:uri a owl:DatatypeProperty ;\n')
#         uri_flag = False  # done
#     elif similarity_value_flag and line == 's3o:similarity_value a owl:ObjectProperty ;\n':
#         line = line.replace('s3o:similarity_value a owl:ObjectProperty ;\n',
#                             's3o:similarity_value a owl:DatatypeProperty ;\n')
#         similarity_value_flag = False  # done

#     return line, similarity_value_flag, uri_flag, similarity_value_flag


# def replace_object_with_data_in_properties(input_filepath, output_filepath):

#     fin = open(input_filepath, "rt")
#     # output file to write the result to
#     fout = open(output_filepath, "wt")

#     serialization_format_flag = True
#     uri_flag = True
#     similarity_value_flag = True

#     for line in fin:
#         # read replace the string and write to output file
#         # for each line in the input file check and replace once occurance
#         if serialization_format_flag or uri_flag or similarity_value_flag:
#             line, serialization_format_flag, uri_flag, similarity_value_flag = replace_line(line, serialization_format_flag,
#                                                                                             uri_flag, similarity_value_flag)

#         fout.write(line)

#     # close input and output files
#     fin.close()
#     fout.close()


# def remove_replaced_DataObjectProperties():
#     node = URIRef(S3ON[S3O_similarity_value])
#     # remove all triples about bob
#     for s, p, o in graph.triples((None, RDF.type, None)):
#         s_str = str(s)
#         p_str = str(p)
#         o_str = str(o)
#         if ('s3o#serialization_format' in s_str and 'owl#ObjectProperty' in o_str):
#             graph.remove(s, RDF.type, o)
#             graph.add(
#                 s, RDF.type, 'http://www.w3.org/2002/07/owl#DataObjectProperty')

#     #graph.remove(S3ON[S3O_similarity_value], RDF.type, OWL.ObjectProperty)
#     # graph.remove(node)
#     for s, p, o in graph.triples((None, RDF.type, None)):
#         print(s, p, o)

#     exit

####################################
# Tesing of the current file
############################


def test_create_S3O_schema_entities():
    create_S3O_schema_entities()

    # input_filepath = 's3o_test.ttl'
    # output_filepath = 's3o_test1.ttl'

    # Check saving of the graph
    graph.serialize(
        S3O_SCHEMA_FILEPATH_TURTTLE, format=DIIM_RDF_TURTTLE_FORMAT, encoding="utf-8")

    # Check loading of the graph
    print('loading file ********** ', S3O_SCHEMA_FILEPATH_TURTTLE)
    load_S3O(S3O_SCHEMA_FILEPATH_TURTTLE)

    # check replacement of DataProperties
    # replace_object_with_data_in_properties(input_filepath, output_filepath)


# To test uncomment the following method
test_create_S3O_schema_entities()
