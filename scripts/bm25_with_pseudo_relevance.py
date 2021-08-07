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
from basic_bm25 import bm25_basic, get_result
from bm25_with_relevance_feedback import bm25_prediction

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()



glove_kv = '../pretrained_corpus/glove_6B_300d.kv'   # pretrained vectors for query expansion


def bm25_pseudo_relevance_back(query, df, tf, tf_norm, idf, norm_association_matrix, vocab, avg_doc_len, k=5):
    query = process_query(query) # lemitize query
    query = expand_query(query,glove_kv,topn=3)  # expand query by including words from pretrianed w2v corpus

    #result_initial, ls_initial = get_result(query=query,tf=tf,tf_norm=tf_norm,idf=idf,vocab=vocab,avg_doc_len=avg_doc_len)
    result_initial, ls_initial = bm25_prediction(query=query, df=df, tf=tf, tf_norm=tf_norm, idf=idf, vocab=vocab, avg_doc_len=avg_doc_len, reformulated=False, relevant_courses=[] )
    relevant_courses = ls_initial[k:]
    associated_words = get_top_k_associated_words(relevant_courses, tf, norm_association_matrix, k=3)
    query = query + associated_words 
    #query = expand_query(query,glove_kv,topn=3)
    # change get_result to bm25 reformulated
    result_reformed, ls_reformed = bm25_prediction(query=query, df=df, tf=tf, tf_norm=tf_norm, idf=idf, vocab=vocab, avg_doc_len=avg_doc_len, reformulated=True, relevant_courses=relevant_courses ) # bm25 reformulated
    return result_reformed, ls_reformed


if __name__ == '__main__':

    tf = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_trained.csv', header=0, index_col=0)
    tf_norm = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_norm_trained.csv', header=0, index_col=0)
    idf = pd.read_csv('../data/trained_scores/course_info_with_survey_idf_trained.csv', header=0, index_col=0)
    df = pd.read_csv('../data/trained_scores/course_info_with_survey_df_trained.csv', header=0, index_col=0)
    
    query_val = pd.read_csv('../data/survey/vaildation_sample_query.csv', header = 0 , index_col = 0)
    association_matrix = pd.read_csv('../data/trained_scores/association_matrix_trained.csv', header = 0, index_col = 0)
    norm_association_matrix = pd.read_csv('../data/trained_scores/norm_association_matrix_trained.csv', header = 0, index_col = 0)


    vocab = tf.index.tolist()  # unique words
    total_length = tf.to_numpy().sum()
    avg_doc_len = total_length / len(tf.columns) # average document length across all courses


    #query = "I am planning to go 'ESD' interested in Finances and wanting to learn python and R"
    query = "network, term, model, technology, probability"
    result, ls = bm25_pseudo_relevance_back(query=query, df=df, tf=tf, tf_norm=tf_norm, idf=idf, norm_association_matrix=norm_association_matrix, vocab=vocab, avg_doc_len=avg_doc_len, k=5)
    print(result)
    for k, v in result.items():
        print(f"{k}: {v}")