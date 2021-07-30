from tqdm import tqdm
import pandas as pd
import itertools
import time

def create_association_matrix(tf):
    '''
    takes in tf dataframe to get association matrix df
    '''
    unique_words = tf.index.values.tolist()
    ls = list(itertools.combinations(tf.index, 2))
    for idx in tf.index:
        ls.append((idx,idx))
    associations = {}
    for item in ls:
        k = item[0]+item[1]
        associations[k] = sum(tf.loc[item[0]]*tf.loc[item[1]])
    association_matrix = pd.DataFrame(0,index=unique_words,columns=unique_words)
    for i in tqdm(unique_words):
        for j in unique_words:
            # association_matrix.loc[i,j] = sum(tf.loc[i]*tf.loc[j])
            if i+j in associations:
                association_matrix.loc[i,j] = associations[i+j]
            elif j+i in associations:
                association_matrix.loc[i,j] = associations[j+i]
    return association_matrix,unique_words

def create_norm_association_matrix(association_matrix, unique_words):
    norm_association_matrix = pd.DataFrame(index=unique_words,columns=unique_words)
    for i in tqdm(unique_words):
        for j in unique_words:
            if i!=j:
                norm_association_matrix.loc[i,j] = association_matrix.loc[i,j]/(association_matrix.loc[i,i]+association_matrix.loc[j,j]-association_matrix.loc[i,j])
    return norm_association_matrix

def get_associated_words(query, matrix):
    '''
    for each word in original query, add on a word which is best associated with it via the corpus
    '''
    extended_query = query.copy()
    for word in query:
        if word in matrix.index:
            new_word = matrix[word].astype(float).idxmax()
            extended_query.append(new_word)
    return extended_query

def get_top_k_associated_words(relevant_courses, tf, matrix, k):
    '''
    get top k associated words based on relevant courses and reformulate back into the query
    '''
    total_new_words = []
    for course in relevant_courses:
        words = tf.index[tf[course]>0].tolist()
        # filtered_matrix = matrix[matrix.index.isin(words)]
        associated_words = {}
        for word in words:
            associated_words[matrix[word].astype(float).idxmax()] = matrix[word].astype(float).max()
        new_words = sorted(associated_words, key=associated_words.get, reverse=True )[:k]
        total_new_words += new_words
    return total_new_words





'''
uncomment to formulate association matrix and normalised association matrix from course_info_with_survey_tf.csv
'''
# tf_path = '../data/course_info_with_survey_scores/course_info__with_survey_tf.csv'
# association_matrix_path = '../data/course_info_with_survey_scores/association_matrix.csv'
# norm_association_matrix_path = '../data/course_info_with_survey_scores/norm_association_matrix.csv'
# tf = pd.read_csv(tf_path, header=0, index_col=0)
# start_time = time.time()
# association_matrix, unique_words = create_association_matrix(tf)
# norm_association_matrix = create_norm_association_matrix(association_matrix, unique_words)
# association_matrix.to_csv(association_matrix_path)
# norm_association_matrix.to_csv(norm_association_matrix_path)
# print(f'time elapsed: {(time.time()-start_time)//60}min {(time.time()-start_time)%60}s') 