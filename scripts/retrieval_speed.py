import pandas as pd
import time
from utils.query_processing import get_wordnet_pos, process_query, expand_query
from basic_bm25 import bm25_basic, get_result
from utils.query_processing import get_wordnet_pos, process_query, expand_query
from bm25_with_pseudo_relevance import bm25_pseudo_relevance_back

tf_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_trained.csv', header=0, index_col=0)
tf_norm_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_norm_trained.csv', header=0, index_col=0)
idf_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_idf_trained.csv', header=0, index_col=0)
vocab = tf_relevance_feedback.index.tolist()  # unique words
total_length = tf_relevance_feedback.to_numpy().sum()
avg_doc_len = total_length / len(tf_relevance_feedback.columns) # average document length across all courses
glove_kv = '../pretrained_corpus/glove_6B_300d.kv'




query = "network, term, model, technology, probability"

# basic bm25 without query expansion
start_time = time.time()
query_1 = process_query(query=query)
result, ls = get_result(query=query_1,tf=tf_relevance_feedback,tf_norm=tf_norm_relevance_feedback,idf=idf_relevance_feedback,vocab=vocab,avg_doc_len=avg_doc_len)
print(f"time taken for basic bm25 without query expansion: {time.time() - start_time}")



# basic bm25 with query expansion
start_time = time.time()
query_2 = process_query(query=query)
query_2 = expand_query(query,glove_kv,topn=3)
result, ls = get_result(query=query,tf=tf_relevance_feedback,tf_norm=tf_norm_relevance_feedback,idf=idf_relevance_feedback,vocab=vocab,avg_doc_len=avg_doc_len)
print(f"time taken for basic bm25 with query expansion: {time.time() - start_time}")



# bm25 with pseudo relevance feedback
start_time = time.time()
df_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_df_trained.csv', header=0, index_col=0)
norm_association_matrix = pd.read_csv('../data/trained_scores/norm_association_matrix_trained.csv', header = 0, index_col = 0)

map_pseudo_relevance_feedback = result, ls = bm25_pseudo_relevance_back(query=query, df=df_relevance_feedback, tf=tf_relevance_feedback, tf_norm=tf_norm_relevance_feedback, idf=idf_relevance_feedback, norm_association_matrix=norm_association_matrix, vocab=vocab, avg_doc_len=avg_doc_len, k=10)
print(f"time taken for basic bm25 with pseudo relevance feedback: {time.time() - start_time}")

