import numpy as np
import pandas as pd
import math
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

def bm25_reformulated(query, doc, relevant_courses, tf, df, vocab, tf_norm, avg_doc_len, k1=1.5, b=0.75, k3=1.5):
    score = 0.0
    vr = len(relevant_courses)  # total retrieved relevant docs
    for term in query:
        if term not in vocab:
            continue
        vr_t = 0  # total retrieved relevant docs where term t appears
        for course in relevant_courses:
            words = tf.index[tf[course]>0].tolist()
            if term in words:
                vr_t+=1
        vnr_t = vr-vr_t  # total retrieved relevant docs where term t does not appear
        df_t = float(df.loc[term]) # document frequency with given term 
        doc_len = tf[doc].to_numpy().sum()
        tf_d = float(tf_norm[doc][term]) + 0.0001 # term frequency in document (include 0.0001 smoothing if not code will break lol)
        tf_q = query.count(term)/len(query) # term frequency in reformulated query normalised
        N = len(tf.columns) # total number of documents
        part_a = ((abs(vr_t)+0.5)/(abs(vnr_t)+0.5))/((df_t-abs(vr_t)+0.5)/(N-df_t-abs(vr)+abs(vr_t)+0.5))
        part_b = ((k1+1)*tf_d)/((k1*((1-b)+b*(doc_len/avg_doc_len)))+tf_d)
        part_c = ((k3+1)*tf_q)/(k3+tf_q)

        score += math.log10(part_a*part_b*part_c)
    return score

def bm25_prediction(query, tf, tf_norm, df, idf, vocab, avg_doc_len, reformulated, relevant_courses=[]):
    '''
    ranks the documents based on the scores of all the documents
    reformulated: if true run bm25_reformulated algorithm, else run bm25_basic algorithm.
    '''
    courses = tf.columns.tolist()
    result = {}
    if reformulated is False:
        print(f'running basic bm25')
        for course in courses:
            result[course] = bm25_basic(query, course, tf, tf_norm, idf, vocab, avg_doc_len)
    elif reformulated is True:
        print(f'running bm25 for reformulated query')
        for course in courses:
            result[course] = bm25_reformulated(query, course, relevant_courses, tf, df, vocab, tf_norm, avg_doc_len)      
    sorted_result = dict(
        sorted(result.items(), key=operator.itemgetter(1), reverse=True))
    ls = []
    for k, v in sorted_result.items():
        ls.append(k)
        # print(f"{k}: {v}")
    return result, ls

def update_scores(tf, relevant_courses, associated_words):
    courses = tf.columns.tolist()
    for course in relevant_courses:
        for word in associated_words:
            if word in tf.index.tolist():
                tf[course][word] += 1
            else:
                idx = courses.index(course)
                temp_lst = [1 if i == idx else 0 for i in range(len(courses))]
                temp_df = pd.DataFrame([temp_lst], columns=courses, index=[word])
                tf = tf.append(temp_df)
    tf.sort_index()

    tf_norm = tf.apply(lambda x: x/x.max(), axis=0) # normalise each column by dividing the max frequency
    tf_norm = tf_norm.replace(np.nan, 0)

    df_arr = np.count_nonzero(tf, axis=1) # compute df: count the nunber of non zero columns in each role 
    df = pd.DataFrame(data={'df':df_arr}, index = tf.index.tolist())

    idf = df.copy()
    idf['df'] = idf['df'].apply(lambda freq: math.log10((len(tf.columns.tolist())) / (freq))) # calc idf
    idf = idf.rename({'df': 'idf'}, axis=1, inplace=False)
    idf = idf.replace(np.nan, 0)
    
    return tf, tf_norm, df, idf

# will fix this later
def train():
    # entire pipeline for training phase
    top_retrieved = 10
    start_time = time.time()

    for idx,row in added_data.iterrows(): # accessing all sample queries from training set
        print(f'Training iteration: {idx+1}')
        vocab = tf.index.tolist()  # unique words
        total_length = tf.to_numpy().sum()
        avg_doc_len = total_length / len(tf.columns) # average document length across all courses
        query = row['querySample']  # take in query from training sample
        print(f'original query from training sample: {query}')
        query = process_query(query) # lemitize query 
        query = get_associated_words(query, norm_association_matrix) # included associated terms
        print(f'query expansion with correlation matrix: {query}')
        query = expand_query(query,glove_kv,topn=3)  # expand query by including words from pretrianed w2v corpus
        print(f'query expansion with pretrained corpus: {query}')
        result, ls = bm25_prediction(query=query, df=df, tf=tf, tf_norm=tf_norm, idf=idf, vocab=vocab, avg_doc_len=avg_doc_len, reformulated=False)  # intial bm25
        predicted = ls[:top_retrieved] # retrieve top 10 courses from predictions
        print(f'initial predicted courses: ')
        for course in predicted: 
            print(course)
        print('')
        gold_standard = added_data['expectedElectivesInOrder'][idx]
        relevant_courses = []
        for i in range(1,top_retrieved+1):
            if gold_standard[i-1] == predicted[i-1]:
                relevant_courses.append(predicted[i-1])
                print(f'Relevant & Retrieved: Rank {i} {predicted[i-1]}') # Documents which are relevant are retrieved, comparing with gold standard training sample
        associated_words = get_top_k_associated_words(relevant_courses, tf, norm_association_matrix, k=3)  # get top 3 associated words from each relevant and retrieved course 
        query+=associated_words  # add associated words to orginal query to form the eformulated query
        print('')
        print(f'reformulated query:{query}')
        result, ls = bm25_prediction(query=query, df=df, tf=tf, tf_norm=tf_norm, idf=idf, vocab=vocab, avg_doc_len=avg_doc_len, relevant_courses=relevant_courses, reformulated=True) # bm25 reformulated
        predicted_reformulated = ls[:top_retrieved] # retrieve top 10 courses from predictions
        print(f'prediction after query reformulation')
        for course in predicted_reformulated:
            print(course)
        
        tf, tf_norm, df, idf = update_scores(tf=tf, relevant_courses=relevant_courses, associated_words=associated_words)  # update tf, tf_norm, idf, df
        association_matrix,unique_words = create_association_matrix(tf=tf)
        norm_association_matrix = create_norm_association_matrix(association_matrix = association_matrix,unique_words = unique_words) # update correlation matrix
        print('updated tf, tf_norm, df, idf, asociation matrix, association matrix norm scores...')
        print(f'time elapsed: {(time.time()-start_time)//60}min {(time.time()-start_time)%60}s')
        print('')



    print(f'time elapsed: {(time.time()-start_time)//60}min {(time.time()-start_time)%60}s') 
    # saving trained scores 
    tf.to_csv('../../../data/bm25/bm25_relevance_feedback/merged_courses_tf_trained.csv')
    tf_norm.to_csv('../../../data/bm25/bm25_relevance_feedback/merged_courses_tf_norm_trained.csv')
    df.to_csv('../../../data/bm25/bm25_relevance_feedback/merged_courses_df_trained.csv')
    idf.to_csv('../../../data/bm25/bm25_relevance_feedback/merged_courses_idf_trained.csv')
    association_matrix.to_csv('../../../data/bm25/bm25_relevance_feedback/association_matrix_trained.csv')
    norm_association_matrix.to_csv('../../../data/bm25/bm25_relevance_feedback/association_matrix_norm_trained.csv')