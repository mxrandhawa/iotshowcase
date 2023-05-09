import json
from dataclasses import dataclass, field
from typing import List
from typing import Set


@dataclass
class DIIM_Dataset(dict):
    '''Object for tracking DIIM Datasets in a collection.'''

    def __init__(self, id=0, name='', source_filepath='', size=0, type='', terms_sourced=[], terms_sourced_count=0, terms_cleaned=[], terms_cleaned_count=0, wordnet_synonyms=[], wordnet_synonyms_count=0, wordnet_syn_lemmas=[], wordnet_syn_lemmas_count=0):
        self['id'] = id  # int
        self['name'] = name  # str  # file name
        # str  # source filepath, from where data has been read
        self['source_filepath'] = source_filepath
        self['size'] = size  # str
        self['type'] = type  # str
        self['terms_sourced'] = terms_sourced  # List[str]
        self['terms_sourced_count'] = terms_sourced_count  # int
        self['terms_cleaned'] = terms_cleaned  # List[str]
        self['terms_cleaned_count'] = terms_cleaned_count  # int
        # Set[term_cleaned : wordnet_synonyms]
        self['wordnet_synonyms'] = wordnet_synonyms
        self['wordnet_synonyms_count'] = wordnet_synonyms_count  # int
        # Set[wordnet_synonym : wordnet_lemmas]
        self['wordnet_syn_lemmas'] = wordnet_syn_lemmas  # list[str]
        self['wordnet_syn_lemmas'] = wordnet_syn_lemmas_count  # int

# getter functions
    def get_id(self):
        return self['id']

    def get_name(self):
        return self['name']

    def get_source_filepath(self):
        return self['source_filepath']

    def get_size(self):
        return self['size']

    def get_type(self):
        return self['type']

    def get_terms_sourced(self):
        return self['terms_sourced']

    def get_terms_sourced_count(self):
        return self['terms_sourced_count']

    def get_terms_cleaned(self):
        return self['terms_cleaned']

    def get_terms_cleaned_count(self):
        return self['terms_cleaned_count']

    def get_wordnet_synonyms(self):
        return self['wordnet_synonyms']

    def get_wordnet_synonyms_count(self):
        return self['wordnet_synonyms_count']

    def get_wordnet_syn_lemmas(self):
        return self['wordnet_syn_lemmas']

    def get_wordnet_syn_lemmas_count(self):
        return self['wordnet_syn_lemmas_count']

# serialization functions
    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def create_DIIM_Dataset(serialized_dict_obj):
        diim_dataset_obj = DIIM_Similarity()
        for key in serialized_dict_obj.keys():
            diim_dataset_obj[key] = serialized_dict_obj[key]

        return diim_dataset_obj


'''
    def toJsonDict(self):
        dict_obj = {}
        dict_obj['id'] = self.id  # int
        dict_obj['name'] = self.name   # str  # file name
        dict_obj['source_filepath'] = self.source_filepath
        dict_obj['size'] = self.size
        dict_obj['type'] = self.type
        dict_obj['terms_sourced'] = self.terms_sourced
        dict_obj['terms_cleaned'] = self.terms_cleaned
        dict_obj['wordnet_synonyms'] = self.wordnet_synonyms
        dict_obj['wordnet_syn_lemmas'] = self.wordnet_syn_lemmas

        return json.dumps(dict_obj)
'''


@dataclass
class DIIM_Terms(list):
    def __init__(self, diim_terms):
        self.diim_terms = diim_terms


@dataclass
class DIIM_Similarity_Matrices(list):
    def __init__(self, similarity_matrices):
        self.similarity_matrices = similarity_matrices


@dataclass
class DIIM_Similarity(dict):
    '''Object for evaluating word similarity in a collection.'''

    def __init__(self, id=0, algorithm_name='', algorithm_info='', filePath='', dataset1_name='', dataset_1_terms_cleaned=[], dataset2_name='', dataset_2_terms_cleaned=[], similarity_matrix=[]):
        self['id'] = id  # int
        self['algorithm_name'] = algorithm_name  # str algorithm_name
        self['algorithm_info'] = algorithm_info  # str
        self['dataset1_name'] = dataset1_name  # str
        self['dataset_1_terms_cleaned'] = dataset_1_terms_cleaned  # List[str]
        self['dataset2_name'] = dataset2_name  # str
        self['dataset_2_terms_cleaned'] = dataset_2_terms_cleaned  # List[str]
        self['similarity_matrix'] = similarity_matrix  # List[str]
        # str path of the stored matrix in a csv file
        self['filePath'] = filePath
        #similarity_matrix_with_details: List[str]

    def get_id(self):
        return self['id']

    def get_algorithm_name(self):
        return self['algorithm_name']

    def get_algorithm_info(self):
        return self['algorithm_info']

    def get_dataset1_name(self):
        return self['dataset1_name']

    def get_dataset_1_terms_cleaned(self):
        return self['dataset_1_terms_cleaned']

    def get_dataset2_name(self):
        return self['dataset2_name']

    def get_dataset_2_terms_cleaned(self):
        return self['dataset_2_terms_cleaned']

    def get_similarity_matrix(self):
        return self['similarity_matrix']

    def get_filePath(self):
        return self['filePath']

    def toJson(self):
        print(self)
        return json.dumps(self, default=lambda o: o.__dict__)

    def create_DIIM_Similarity(serialized_dict_obj):
        diim_similarity_obj = DIIM_Similarity()
        for key in serialized_dict_obj.keys():
            diim_similarity_obj[key] = serialized_dict_obj[key]

        return diim_similarity_obj


@dataclass
class DIIM_Term:
    '''Object for tracking DIIM Terms in a collection.'''

    def __init__(self, word, wordnet_synonyms, synonym_type):
        self.word = word  # int = 0
        # list of synonyms object from wordnet
        self.wordnet_synonyms = wordnet_synonyms
        self.synonym_type = synonym_type


@dataclass
class DIIM_IoT_Semantic_Model:
    '''Object for tracking DIIM Terms in a collection.'''
    id: int = 0
    terms_as_class: List[str] = field(default_factory=list)  # list of classes
    terms_as_attribute: List[str] = field(
        default_factory=list)  # list of attributes
    terms_as_relation: List[str] = field(
        default_factory=list)  # list of relations
# todo
# tupel of subject relation object
