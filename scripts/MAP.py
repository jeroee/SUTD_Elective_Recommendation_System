import pandas as pd 
import numpy as np
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


from utils.association_matrix import  get_top_k_associated_words, get_associated_words
from utils.query_processing import get_wordnet_pos, process_query, expand_query
from basic_bm25 import bm25_basic, get_result
from bm25_with_pseudo_relevance import bm25_pseudo_relevance_back
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()


def clean_elective_names(relevant_results):
    '''
        Change the few course names in the survey to be same as the module names scraped
    '''
    # clean up the relevant course names 

    #https://stackoverflow.com/questions/2582138/finding-and-replacing-elements-in-a-list
    replacements = {
        ' 50.035 Computer Vision': '50.035 Computer Vision',
        '50.043 Database Systems / Database and Big Data Systems (for class 2021)': '50.043 Database Systems',
    }

    relevant_results = [replacements.get(x, x) for x in relevant_results]
    
    if '40.302 Advanced Optim/ 40.305 Advanced Stochastic' in relevant_results:
        relevant_results.remove('40.302 Advanced Optim/ 40.305 Advanced Stochastic')
        relevant_results.append('40.302 Advanced Topics in Optimisation#')
        #relevant_results.append('40.305 Advanced Topics in Stochastic Modelling#')
    return relevant_results


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.

    return np.mean(out)

def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def get_map_no_query_expansion(query_val, tf, tf_norm, idf):

    '''
        Get the MAP score for basic bm25 given the input tf, tf_norm and idf
    '''
    top_retrieved = 10
    rs = []
    for index, row in query_val.iterrows():

        vocab = tf.index.tolist()  # unique words
        total_length = tf.to_numpy().sum()
        avg_doc_len = total_length / len(tf.columns) # average document length across all courses
        query_original = row['querySample']  # take in query from training sample
        query = process_query(query=query_original)

        result, ls = get_result(query=query,tf=tf,tf_norm=tf_norm,idf=idf,vocab=vocab,avg_doc_len=avg_doc_len)
        predicted = ls[:top_retrieved] # retrieve top 10 courses from predictions
        relevant_results = eval(row['expectedElectivesInOrder'])
        relevant_results = clean_elective_names(relevant_results)  
        
        #print('rpedicted: ', predicted)
        #print('relevant: ', relevant_results)
        r = []
        for query_result in predicted:
            if query_result in relevant_results:
                r.append(1)
            else:
                r.append(0)

        ap = round(average_precision(r), 5)
        print(f"query: {query_original}".ljust(100, " "), f"Average Precision {ap}")
        rs.append(r)

    map = mean_average_precision(rs)
    #print("Mean Average Precision on validation query: ", map)
    return map

def get_map(query_val, tf, tf_norm, idf):

    '''
        Get the MAP score for basic bm25 (with query expansion) given the input tf, tf_norm and idf
    '''
    glove_kv = '../pretrained_corpus/glove_6B_300d.kv'   # pretrained vectors for query expansion
    top_retrieved = 10
    rs = []
    for index, row in query_val.iterrows():

        vocab = tf.index.tolist()  # unique words
        total_length = tf.to_numpy().sum()
        avg_doc_len = total_length / len(tf.columns) # average document length across all courses
        query_original = row['querySample']  # take in query from training sample
        query = process_query(query=query_original)
        query = expand_query(query,glove_kv,topn=3)
        result, ls = get_result(query=query,tf=tf,tf_norm=tf_norm,idf=idf,vocab=vocab,avg_doc_len=avg_doc_len)
        predicted = ls[:top_retrieved] # retrieve top 10 courses from predictions
        relevant_results = eval(row['expectedElectivesInOrder'])
        relevant_results = clean_elective_names(relevant_results)  
        
        #print('rpedicted: ', predicted)
        #print('relevant: ', relevant_results)
        r = []
        for query_result in predicted:
            if query_result in relevant_results:
                r.append(1)
            else:
                r.append(0)

        ap = round(average_precision(r), 5)
        print(f"query: {query_original}".ljust(100, " "), f"Average Precision {ap}")
        rs.append(r)

    map = mean_average_precision(rs)
    #print("Mean Average Precision on validation query: ", map)
    return map


def get_map_pseudo_relevance(query_val, df, tf, tf_norm, idf, norm_association_matrix):

    '''
        Get the MAP score for BM25 with pseudo relevance feedback
    '''
    top_retrieved = 10
    rs = []
    for index, row in query_val.iterrows():

        vocab = tf.index.tolist()  # unique words
        total_length = tf.to_numpy().sum()
        avg_doc_len = total_length / len(tf.columns) # average document length across all courses
        query_original = row['querySample']  # take in query from training sample
        #query = process_query(query=query_original)
        result, ls = bm25_pseudo_relevance_back(query=query_original, df=df, tf=tf, tf_norm=tf_norm, idf=idf, norm_association_matrix=norm_association_matrix, vocab=vocab, avg_doc_len=avg_doc_len, k=10)
        predicted = ls[:top_retrieved] # retrieve top 10 courses from predictions
        relevant_results = eval(row['expectedElectivesInOrder'])
        relevant_results = clean_elective_names(relevant_results)  
        
        #print('rpedicted: ', predicted)
        #print('relevant: ', relevant_results)
        r = []
        for query_result in predicted:
            if query_result in relevant_results:
                r.append(1)
            else:
                r.append(0)

        ap = round(average_precision(r), 5)
        print(f"query: {query_original}".ljust(100, " "), f"Average Precision {ap}")
        rs.append(r)

    map = mean_average_precision(rs)
    #print("Mean Average Precision on validation query: ", map)
    return map


if __name__ == '__main__':
    # print("#"*200)
    # print('Calculating Mean Average Precision for bm25 without survey data added')
    # query_val = pd.read_csv('../data/survey/vaildation_sample_query.csv', header = 0 , index_col = 0)

    # # scores without survey
    # tf = pd.read_csv('../data/course_info_scores/course_info_tf.csv', header=0, index_col=0)
    # tf_norm = pd.read_csv('../data/course_info_scores/course_info_tf_norm.csv', header=0, index_col=0)
    # idf = pd.read_csv('../data/course_info_scores/course_info_idf.csv', header=0, index_col=0)
    # map = get_map(query_val, tf=tf, tf_norm=tf_norm, idf=idf)
    # print("Mean Average Precision on validation query (bm25 with no survey): ", map)

    # print("#"*200)
    # print('Calculating Mean Average Precision for bm25 with survey data added')
    # tf_with_survey = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_tf.csv', header=0, index_col=0)
    # tf_norm_with_survey = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_tf_norm.csv', header=0, index_col=0)
    # idf_with_survey = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_idf.csv', header=0, index_col=0)
    # map_with_survey = get_map(query_val, tf=tf_with_survey, tf_norm=tf_norm_with_survey, idf=idf_with_survey)
    # print("Mean Average Precision on validation query (bm25 after relevance feedback training): ", map_with_survey)


    # print("#"*200)
    # print('Calculating Mean Average Precision for bm25 after training with relevance feedback')
    # tf_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_trained.csv', header=0, index_col=0)
    # tf_norm_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_norm_trained.csv', header=0, index_col=0)
    # idf_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_idf_trained.csv', header=0, index_col=0)
    # map_relevance_feedback = get_map(query_val, tf=tf_relevance_feedback, tf_norm=tf_norm_relevance_feedback, idf=idf_relevance_feedback)
    # print("Mean Average Precision on validation query (bm25 after relevance feedback training): ", map_relevance_feedback)

    # df_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_df_trained.csv', header=0, index_col=0)
    # norm_association_matrix = pd.read_csv('../data/trained_scores/norm_association_matrix_trained.csv', header = 0, index_col = 0)
    # print("#"*200)
    # print('Calculating Mean Average Precision for bm25 with pseudo relevance feedback')
    # map_pseudo_relevance_feedback = get_map_pseudo_relevance(query_val, df=df_relevance_feedback, tf=tf_relevance_feedback, tf_norm=tf_norm_relevance_feedback, idf=idf_relevance_feedback, norm_association_matrix=norm_association_matrix)
    # print("Mean Average Precision on validation query (bm25 with pseudo relevance feedback): ", map_pseudo_relevance_feedback)
    query_val = pd.read_csv('../data/survey/vaildation_sample_query.csv', header = 0 , index_col = 0)
    




    print("#"*200)
    print('Calculating Mean Average Precision for bm25 (with no query expansion) without survey data added')
    tf = pd.read_csv('../data/course_info_scores/course_info_tf.csv', header=0, index_col=0)
    tf_norm = pd.read_csv('../data/course_info_scores/course_info_tf_norm.csv', header=0, index_col=0)
    idf = pd.read_csv('../data/course_info_scores/course_info_idf.csv', header=0, index_col=0)
    map_no_query_expansion = get_map_no_query_expansion(query_val, tf=tf, tf_norm=tf_norm, idf=idf)
    print("Mean Average Precision on validation query (bm25 without query expansion and with no survey): ", map_no_query_expansion)


    print("#"*200)
    print('Calculating Mean Average Precision for bm25 (with query expansion) without survey data added')

    # scores without survey

    map = get_map(query_val, tf=tf, tf_norm=tf_norm, idf=idf)
    print("Mean Average Precision on validation query (bm25 with no survey): ", map)

    print("#"*200)
    print('Calculating Mean Average Precision for bm25 with survey data added')
    tf_with_survey = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_tf.csv', header=0, index_col=0)
    tf_norm_with_survey = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_tf_norm.csv', header=0, index_col=0)
    idf_with_survey = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_idf.csv', header=0, index_col=0)
    map_with_survey = get_map(query_val, tf=tf_with_survey, tf_norm=tf_norm_with_survey, idf=idf_with_survey)
    print("Mean Average Precision on validation query (bm25 after relevance feedback training): ", map_with_survey)

    print("#"*200)
    print('Calculating Mean Average Precision for bm25 after training with relevance feedback')
    tf_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_trained.csv', header=0, index_col=0)
    #tf_relevance_feedback = pd.read_csv('../data/course_info_scores/course_info_tf.csv', header=0, index_col=0)
    tf_norm_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_norm_trained.csv', header=0, index_col=0)
    idf_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_idf_trained.csv', header=0, index_col=0)
    map_relevance_feedback = get_map(query_val, tf=tf_relevance_feedback, tf_norm=tf_norm_relevance_feedback, idf=idf_relevance_feedback)
    print("Mean Average Precision on validation query (bm25 after relevance feedback training): ", map_relevance_feedback)
    
    # for index, row in idf_with_survey.iterrows():
    #     if row['idf'] == idf_relevance_feedback['idf'][index]:
    #         print(index)
    df_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_df_trained.csv', header=0, index_col=0)
    norm_association_matrix = pd.read_csv('../data/trained_scores/norm_association_matrix_trained.csv', header = 0, index_col = 0)
    print("#"*200)
    print('Calculating Mean Average Precision for bm25 with pseudo relevance feedback')
    map_pseudo_relevance_feedback = get_map_pseudo_relevance(query_val, df=df_relevance_feedback, tf=tf_relevance_feedback, tf_norm=tf_norm_relevance_feedback, idf=idf_relevance_feedback, norm_association_matrix=norm_association_matrix)
    print("Mean Average Precision on validation query (bm25 with pseudo relevance feedback): ", map_pseudo_relevance_feedback)