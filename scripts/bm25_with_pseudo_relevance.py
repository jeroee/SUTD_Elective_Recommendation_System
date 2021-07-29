import pandas as pd 
import numpy as np
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import operator
import time

from utils.association_matrix import  get_top_k_associated_words, get_associated_words
from utils.query_processing import get_wordnet_pos, process_query, expand_query
from basic_bm25 import bm25_basic

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()


tf = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_trained.csv', header=0, index_col=0)
tf_norm = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_norm_trained.csv', header=0, index_col=0)
idf = pd.read_csv('../data/trained_scores/course_info_with_survey_idf_trained.csv', header=0, index_col=0)
df = pd.read_csv('../data/trained_scores/course_info_with_survey_df_trained.csv', header=0, index_col=0)
glove_kv = '../pretrained_corpus/glove_6B_300d.kv'   # pretrained vectors for query expansion
query_val = pd.read_csv('../data/survey/vaildation_sample_query.csv', header = 0 , index_col = 0)
association_matrix = pd.read_csv('../data/trained_scores/association_matrix_trained.csv', header = 0, index_col = 0)
norm_association_matrix = pd.read_csv('../data/trained_scores/norm_association_matrix_trained.csv', header = 0, index_col = 0)


def bm25_prediction(query, tf, tf_norm, df, idf, vocab, avg_doc_len, reformulated, relevant_courses=[]):
    '''
    ranks the documents based on the scores of all the documents
    reformulated: if true run bm25_reformulated algorithm, else run bm25_basic algorithm.
    '''
    courses = tf.columns.tolist()
    result = {}
    for course in courses:
        result[course] = bm25_basic(query, course, tf, tf_norm, idf, vocab, avg_doc_len)
    sorted_result = dict(sorted(result.items(), key=operator.itemgetter(1), reverse=True))
    ls = []
    for k, v in sorted_result.items():
        ls.append(k)
        
    return sorted_result, ls

def bm25_pseudo_relevance_back(query, tf, tf_norm, df, idf, vocab, avg_doc_len, k=10):
    query = process_query(query) # lemitize query
    query = expand_query(query,glove_kv,topn=3)  # expand query by including words from pretrianed w2v corpus

    result_initial, ls_initial = bm25_prediction(query=query, df=df, tf=tf, tf_norm=tf_norm, idf=idf, vocab=vocab, avg_doc_len=avg_doc_len, reformulated=False)
    relevant_courses = ls_initial[k:]

    associated_words = get_top_k_associated_words(relevant_courses, tf, norm_association_matrix, k=3)
    query = query + associated_words 
    
    result_reformed, ls_reformed = bm25_prediction(query=query, df=df, tf=tf, tf_norm=tf_norm, idf=idf, vocab=vocab, avg_doc_len=avg_doc_len, reformulated=False)
    return result_reformed, ls_reformed




if __name__ == '__main__':
    vocab = tf.index.tolist()  # unique words
    total_length = tf.to_numpy().sum()
    avg_doc_len = total_length / len(tf.columns) # average document length across all courses


    query = "I am planning to go 'ESD' interested in Finances and wanting to learn python and R"
    result, ls = bm25_pseudo_relevance_back(query=query, df=df, tf=tf, tf_norm=tf_norm, idf=idf, vocab=vocab, avg_doc_len=avg_doc_len, k=10)
    for k, v in result.items():
        print(f"{k}: {v}")