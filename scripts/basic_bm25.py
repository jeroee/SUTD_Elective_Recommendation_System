import numpy as np
import pandas as pd
import operator

def bm25_basic(query, doc, tf, tf_norm, idf, vocab, avg_doc_len, k1=1.5, b=0.75):
    ''''
    returns the score for a document given a query based on the basic bm25 algorithm
    '''
    score = 0.0
    for term in query:
        if term not in vocab:
            continue
        numerator = idf['idf'][term] * tf_norm[doc][term] * (k1 + 1)
        doc_len = tf[doc].to_numpy().sum()
        denominator = tf_norm[doc][term] + k1 * \
            (1 - b + b * doc_len / avg_doc_len)
        score += (numerator / denominator)
    return score

def get_result(query,tf,tf_norm,idf,vocab,avg_doc_len):
    '''
    Sort and return the bm25 scores for all the SUTD Courses

    return
        sorted_result : {module_name: bm25_score}
        ls: a ranked list of module names in descending bm25 score 
    '''
    courses = tf.columns.tolist()
    result = {}
    for course in courses:
        result[course] = bm25_basic(query, course,tf,tf_norm,idf,vocab,avg_doc_len)

    sorted_result = dict(
        sorted(result.items(), key=operator.itemgetter(1), reverse=True))
    #print(result)
    ls=[]
    for k, v in sorted_result.items():
        #print(f"{k}: {v}")
        ls.append(k)
    #print(sorted_result)
    return sorted_result, ls