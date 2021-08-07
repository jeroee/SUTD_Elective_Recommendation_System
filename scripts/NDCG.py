# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
## from week 6 lab
def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    import numpy as np
    r = np.asfarray(r)[:k]
    if r.size: ## why is this r.size? when will this be false?
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


# %%
## from week 6 lab
def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Normalized discounted cumulative gain
    """
    import numpy as np

    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    # print('For k is {}, DCG scorce is {}'.format(k,dcg_at_k(r, k, method)))
    # print('For k is {}, IDCG scorce is {}'.format(k,dcg_max))
    return dcg_at_k(r, k, method) / dcg_max


# %%
def assigningBM25ScoreToRelevantAndRetrieved(_bm25ScoreDf, relevantDocsList):
    """[summary]
    This function helps to assign zero values to those non-relevant and retrieved documents
    It retents the score of those relevant and retrieved

    Args:
        _bm25ScoreDf ([dataframe]): [a dataframe where rows are modules and columns is the bm25 scores]
        relevantAndRetrievedDocs ([list]): [list of modules based on the golden standard(idea outcome based on survey)]
    """
    df = _bm25ScoreDf.copy(deep = False)
    irrelevantAndRetrievedDocsList = list(set(df.index) - set(relevantDocsList))
    
    for relevantAndRetrievedDoc in irrelevantAndRetrievedDocsList:
        df.loc[relevantAndRetrievedDoc]['bm25Score'] = 0
    """[summary]
    output is a df with score that are retrieved and relevant(relevant depends on the gold standard)
    """
    return(df)


# %%
def NDCGWithVariousK(retrievedDocsDf,listOfRelevantDocs, exportResults = 0, queryNum = '', fileName = 'test'):
    """[summary]
    This function compute the NDGC at vaious K

    Args:
        retrievedDocsDf ([dataframe]): [dataframe of retrieved documents and it's bm25 score]
        listOfRelevantDocs ([list]): [list of relevant Documents based on gold standard]
        exportResults (int, optional): [to determine to export ndcg results]. Defaults to 0 and 1 to export ndgc score
        fileName (str, optional): [fileName to be exported ideally it should be the "ndcg_score_'model name']. Defaults to 'test'.
    """
    ## assign zero values to those non-relevant and retrieved documents, It retain the score of those relevant and retrieved
    BM25ScoreToRelevantAndRetrieved = assigningBM25ScoreToRelevantAndRetrieved(retrievedDocsDf,listOfRelevantDocs)
    ## obtain the score of the BM25 of the relevant and retrieved modules
    BM25ScoreToRelevantAndRetrievedScoreList = list(BM25ScoreToRelevantAndRetrieved.bm25Score)
    
    ## dict to save NDCGScore ie {k(ranking):NDCG Score}
    NDCGScoreDict = {}
    for i in range(1,len(BM25ScoreToRelevantAndRetrievedScoreList)+1):
        ndcg_at_kScore = ndcg_at_k(BM25ScoreToRelevantAndRetrievedScoreList,i)
        # print('For k is {}, NDCG scorce is {}\n'.format(i,ndcg_at_kScore))
        NDCGScoreDict[i] = ndcg_at_kScore
    
    ## convert dict to df for easier sorting analysis of the scores and exporting it to csv
    import pandas as pd
    NDCGDf = pd.DataFrame.from_dict(NDCGScoreDict,orient='index',columns=['NDCGScore{}'.format(queryNum)])
    NDCGDf.reset_index(inplace = True)
    ## rename the column to k columns 
    NDCGDf.rename(columns={"index": "k"}, inplace = True)
    
    ## to export the ndcg scores to csv if exportResults == 1
    if exportResults == 1:
        fileName = 'ndcg_score_{}.csv'.format(fileName)
        NDCGDf.to_csv('../results/ndcg_score/{}'.format(fileName))
    return(NDCGDf)
    

# %% [markdown]
# # Toy Problem formulation

# %%
if __name__ == "__main__":
    "Test Case : the retrievedDocScore"
    ## assume docs are not in bm25 scorce order
    retrievedDocs = ['D','C', 'B','A'] 
    retrievedDocsScore = [0.43, 0.26, 0.03, 0.37]
    ## I realised that the score should be in ascending order of bm25 score hence I made some changes to fit our use case
    # retrievedDocsScore = [0.43,  0.37, 0.26, 0.03]

    ## creating a retrievedDocsDf for test cases
    ## this should be the same format of the bm25 output
    retrievedDocsDict = {}
    for index in range(len(retrievedDocs)):
        retrievedDocsDict[retrievedDocs[index]] = retrievedDocsScore[index]
    import pandas as pd
    retrievedDocsDf1 = pd.DataFrame.from_dict(retrievedDocsDict,orient='index',columns = ['bm25Score'])

    print('BM25 output:')
    retrievedDocsDf1

    "Test Case : the retrievedDocScore"
    ## assume docs are not in bm25 scorce order
    retrievedDocs = ['C','D', 'B','A'] 
    retrievedDocsScore = [0.5, 0.3, 0.2, 0.1]
    ## I realised that the score should be in ascending order of bm25 score hence I made some changes to fit our use case
    # retrievedDocsScore = [0.43,  0.37, 0.26, 0.03]

    ## creating a retrievedDocsDf for test cases
    ## this should be the same format of the bm25 output
    retrievedDocsDict = {}
    for index in range(len(retrievedDocs)):
        retrievedDocsDict[retrievedDocs[index]] = retrievedDocsScore[index]
    import pandas as pd
    retrievedDocsDf2 = pd.DataFrame.from_dict(retrievedDocsDict,orient='index',columns = ['bm25Score'])

    print('BM25 output:')
    retrievedDocsDf2
    "Test Case : The Relevant Docs"
    relevantDocs1 = ['B','D','E']
    print('List of relevant Docs: {}'.format(relevantDocs1))
    relevantDocs2 = ['A','C']
    print('List of relevant Docs: {}'.format(relevantDocs2))

    retrievedlist = [retrievedDocsDf1,retrievedDocsDf2]
    relevantlist =[relevantDocs1,relevantDocs2]
    
    ## test case
    import pandas as pd
    ## this index is meant to keep track of the NDCG score of each query
    queryIndex = 0
    for retrieved in retrievedlist:
    ## to compute the NDCG of a single query
        NDCGWithVariousKdf = NDCGWithVariousK(retrieved,relevantlist[queryIndex],0,queryIndex)
    ## if this is 1st NDCG score been compute, make it's df to NDCG df else merge with the current overall NDGC df
        if queryIndex == 0:
            NDCGDf = NDCGWithVariousKdf
        else:
            NDCGDf = pd.merge(NDCGDf, NDCGWithVariousKdf, on=["k"])
        queryIndex += 1


# %%
def clean_elective_names(relevant_results):
    # clean up the relevant course names 

    #https://stackoverflow.com/questions/2582138/finding-and-replacing-elements-in-a-list
    try:
        relevant_results = relevant_results.split(',')
        relevant_results = [x.replace("'",'') for x in relevant_results]
        relevant_results = [x.replace("[",'') for x in relevant_results]
        relevant_results = [x.replace("]",'') for x in relevant_results]
    ## this is required as apart from the index 0 module the other modules still retain a space inform of them
        relevant_results2 = [x.replace(" ",'',1) for x in relevant_results if x != relevant_results[0]]
    ## thus the next 2 lines of code help to reinsert the 0th index modules and reassign relevant_results2 to relevant_results
        relevant_results2.insert(0,relevant_results[0])
        relevant_results = relevant_results2
    except:
        pass
    replacements = {
        ' 50.035 Computer Vision': '50.035 Computer Vision'
        ,'50.043 Database Systems / Database and Big Data Systems (for class 2021)': '50.043 Database Systems'
        }

    relevant_results = [replacements.get(x, x) for x in relevant_results]
    
    if '40.302 Advanced Optim/ 40.305 Advanced Stochastic' in relevant_results:
        relevant_results.remove('40.302 Advanced Optim/ 40.305 Advanced Stochastic')
        relevant_results.append('40.302 Advanced Topics in Optimisation#')
        relevant_results.append('40.305 Advanced Topics in Stochastic Modelling#')
    return relevant_results


# %%
def displayNdcgAt10(queryCount,query_val,NDCGWithVariousKdf):
    """[summary]
    to print the result of ndcg at 10 for every query

    Args:
        queryCount ([int]): [description] counter to indicate wht query it is at wihin query_val's querySample columns
        NDCGWithVariousKdf ([dataframe]): [description] df with a single query's ndcg score at various k
    """
    queryColumn = 'NDCGScore{}'.format(queryCount)
    ndcgAt10 = NDCGWithVariousKdf[queryColumn][9]
    ndcgAt10 = round(ndcgAt10, 5)
    print(f"query: {query_val['querySample'][queryCount]}".ljust(100, " "), f"NDCG@10 {ndcgAt10}")


# %%
## function to compute the NDCG for cosine simliarities for model 1
def get_NDCG_cosine_no_expan(query_val,tf,queryExpansion = 0):
    """[summary]
    To obtain the NDCG @ for cosine simliarities model without query expansion

    Args:
        query_val ([dataframe]): [description] queries for vaildation
        tf ([dataframe]): [description] tf of courpus
        queryExpansion (int, optional): [description]. Defaults to 0.
    """
    import CosineSimilarity
    ## compute Cosine simliarities score
    cosineSimDf = CosineSimilarity.rankedModuleOfCosineSim(query_val,tf,queryExpansion)
    cosineSimDf = cosineSimDf.T

    ## this section help to compute and obtain the NDCG for each query and store in df
    import pandas as pd
    queryCount = 0
    NDCGDf = 0
    for query,row in cosineSimDf.iterrows():
        ## create the df for retrieved docs and it's score
            retrievedDocsDict = {}
            cleanedElectives = clean_elective_names(row['topModules'])
            for index in range(len(row['topModules'])):
                retrievedDocsDict[cleanedElectives[index]] = row['topModulesScore'][index]
            import pandas as pd
            retrievedDocsDf = pd.DataFrame.from_dict(retrievedDocsDict,orient='index',columns = ['bm25Score'])
        
        ## cleaned golden/vaildation set modules
            validModules = clean_elective_names(query_val['expectedElectivesInOrder'][queryCount])
        ## to compute the NDCG of a single query
            NDCGWithVariousKdf = NDCGWithVariousK(retrievedDocsDf,validModules,0,queryCount)
        ## if this is 1st NDCG score been compute, make it's df to NDCG df else merge with the current overall NDGC df
            if queryCount == 0:
                NDCGDf = NDCGWithVariousKdf
            else:
                NDCGDf = pd.merge(NDCGDf, NDCGWithVariousKdf, on=["k"])
                
        ## print query and NDCG score at k =10
            displayNdcgAt10(queryCount,query_val,NDCGWithVariousKdf)
        ## update the query index by 1
            queryCount += 1
            
    ## return a df with all the ndcg results
    
    return(NDCGDf)


# %%
## function to compute the NDCG for cosine simliarities for model 2 and 3
def get_NDCG_cosine(query_val,tf,queryExpansion = 1):
    """[summary]
    To obtain the NDCG @ for cosine simliarities model with query expansion

    Args:
        query_val ([dataframe]): [description] queries for vaildation
        tf ([dataframe]): [description] tf of courpus
        queryExpansion (int, optional): [description]. Defaults to 1.
    """
    import CosineSimilarity
    
    ## compute Cosine simliarities score
    cosineSimDf = CosineSimilarity.rankedModuleOfCosineSim(query_val,tf,queryExpansion)
    cosineSimDf = cosineSimDf.T

    ## this section help to compute and obtain the NDCG for each query and store in df
    import pandas as pd
    queryCount = 0
    NDCGDf = 0
    for query,row in cosineSimDf.iterrows():
        ## create the df for retrieved docs and it's score
            retrievedDocsDict = {}
            cleanedElectives = clean_elective_names(row['topModules'])
            for index in range(len(row['topModules'])):
                retrievedDocsDict[cleanedElectives[index]] = row['topModulesScore'][index]
            import pandas as pd
            retrievedDocsDf = pd.DataFrame.from_dict(retrievedDocsDict,orient='index',columns = ['bm25Score'])
        
        ## cleaned golden/vaildation set modules
            validModules = clean_elective_names(query_val['expectedElectivesInOrder'][queryCount])
        ## to compute the NDCG of a single query
            NDCGWithVariousKdf = NDCGWithVariousK(retrievedDocsDf,validModules,0,queryCount)
        ## if this is 1st NDCG score been compute, make it's df to NDCG df else merge with the current overall NDGC df
            if queryCount == 0:
                NDCGDf = NDCGWithVariousKdf
            else:
                NDCGDf = pd.merge(NDCGDf, NDCGWithVariousKdf, on=["k"])
                
        ## print query and NDCG score at k =10
            displayNdcgAt10(queryCount,query_val,NDCGWithVariousKdf)
        ## update the query index by 1
            queryCount += 1
            
    ## return a df with all the ndcg results
    
    return(NDCGDf)


# %%
## function to compute the NDCG for bm25basic for model 4
## model 4 BM25 Basic (without query expansion, course information)
def get_NDCG_BM25BasicNoExpan(query_val,tf,tf_norm,idf):
    """[summary]
    To obtain NDCG at 10 for BM25 using basic formale with no query expansion
    Args:
        query_val ([df]): [description] queries for vaildation
        tf ([df]): [description] tf of corpus
        tf_norm ([df]): [description] tf_norm of corpus
        idf ([df]): [description] idf of corpus
    """
    import time
    import basic_bm25
    import utils.query_processing
    
    ## vairables required for basic_bm25 function
    vocab = tf.index.tolist()
    total_length = tf.to_numpy().sum()
    avg_doc_len = total_length / len(tf.columns) # average document length across all courses
    
    ## this section help to compute and obtain the NDCG for each query and store in df
    import pandas as pd
    queryCount = 0
    NDCGDf = 0
    totalTime = 0
    for index,row in query_val.iterrows():
        query = row['querySample']
        query = utils.query_processing.process_query(query)
        
    ## compute basic_bm25 score
        start = time.time()
        retrievedDocs, rankedLs = basic_bm25.get_result(query,tf,tf_norm,idf,vocab,avg_doc_len)
        end =  time.time()- start
        totalTime += end
    ## converting moduleNScore to dataframe    
        retrievedDf = pd.DataFrame.from_dict(retrievedDocs,orient='index',columns = ['bm25Score'])        
    ## cleaned golden/vaildation set modules
        validModules = clean_elective_names(query_val['expectedElectivesInOrder'][queryCount])
    ## to compute the NDCG of a single query
        NDCGWithVariousKdf = NDCGWithVariousK(retrievedDf,validModules,0,queryCount)
        
    ## if this is 1st NDCG score been compute, make it's df to NDCG df else merge with the current overall NDGC df
        if queryCount == 0:
            NDCGDf = NDCGWithVariousKdf
        else:
            NDCGDf = pd.merge(NDCGDf, NDCGWithVariousKdf, on=["k"])
        
    ## print query and NDCG score at k =10
        displayNdcgAt10(queryCount,query_val,NDCGWithVariousKdf)
    ## update the query index by 1
        queryCount += 1
        
        ## print only when the last query is computed
        if queryCount == (len(query_val)):
            averageQueryTime = totalTime/queryCount
            print('Average Time for {} number of queries : {}'.format(queryCount,averageQueryTime))
    ## return a df with all the ndcg results
    
    return(NDCGDf)


# %%
## function to compute the NDCG for bm25basic for model 5 and 6
## model 5 BM25 Basic (query expansion, course information)
## model 6 BM25 Basic (query expansion, course information + survey (50%))
## model 7 Bm25 with Reformulation (query expansion, course information + survey (50%))

def get_NDCG_BM25Basic(query_val,tf,tf_norm,idf):
    """[summary]
    To obtain NDCG at 10 for BM25 using basic formale with query expansion
    Args:
        query_val ([df]): [description] queries for vaildation
        tf ([df]): [description] tf of corpus
        tf_norm ([df]): [description] tf_norm of corpus
        idf ([df]): [description] idf of corpus
    """
    import basic_bm25
    import utils.query_processing
    import time
    
    ## vairables required for basic_bm25 function
    vocab = tf.index.tolist()
    total_length = tf.to_numpy().sum()
    avg_doc_len = total_length / len(tf.columns) # average document length across all courses
    
    ## this section help to compute and obtain the NDCG for each query and store in df
    import pandas as pd
    queryCount = 0
    NDCGDf = 0
    totalTime = 0
    for index,row in query_val.iterrows():
        query = row['querySample']
        query = utils.query_processing.process_query(query)
        
    ## for query expansion
        glove_kv = '../pretrained_corpus/glove_6B_300d.kv'   # pretrained vectors for query expansion
        topn = 3
        query = utils.query_processing.expand_query(query,glove_kv,topn)
    ## compute basic_bm25 score and start the timer for querying
        start = time.time()
        retrievedDocs, rankedLs = basic_bm25.get_result(query,tf,tf_norm,idf,vocab,avg_doc_len)
        end =  time.time()- start
        totalTime += end
    ## converting moduleNScore to dataframe    
        retrievedDf = pd.DataFrame.from_dict(retrievedDocs,orient='index',columns = ['bm25Score'])        
    ## cleaned golden/vaildation set modules
        validModules = clean_elective_names(query_val['expectedElectivesInOrder'][queryCount])
    ## to compute the NDCG of a single query
        NDCGWithVariousKdf = NDCGWithVariousK(retrievedDf,validModules,0,queryCount)
        
    ## if this is 1st NDCG score been compute, make it's df to NDCG df else merge with the current overall NDGC df
        if queryCount == 0:
            NDCGDf = NDCGWithVariousKdf
        else:
            NDCGDf = pd.merge(NDCGDf, NDCGWithVariousKdf, on=["k"])
    ## print query and NDCG score at k =10
        displayNdcgAt10(queryCount,query_val,NDCGWithVariousKdf)
    ## update the query index by 1
        queryCount += 1
    
    ## print only when the last query is computed
        if queryCount == (len(query_val)):
            averageQueryTime = totalTime/queryCount
            print('Average Time for {} number of queries : {}'.format(queryCount,averageQueryTime))
    ## return a df with all the ndcg results
    
    return(NDCGDf)


# %%
## function to compute the NDCG for bm25basic for model 8
## model 8 Bm25 with Reformulation and Pseudo Relevance Feedback (query expansion, course information + survey (50%))
def get_NDCG_BM25WPseudo(query_val,tf,tf_norm,idf):
    """[summary]
    To obtain NDCG at 10 for BM25 with relevance feedback with query expansion
    Args:
        query_val ([df]): [description] queries for vaildation
        tf ([df]): [description] tf of corpus
        tf_norm ([df]): [description] tf_norm of corpus
        idf ([df]): [description] idf of corpus
    """
    import bm25_with_pseudo_relevance
    import utils.query_processing
    import time
    import pandas as pd
    ## vairables required for basic_bm25 function
    vocab = tf.index.tolist()
    total_length = tf.to_numpy().sum()
    avg_doc_len = total_length / len(tf.columns) # average document length across all courses
    norm_association_matrix = pd.read_csv('../data/trained_scores/norm_association_matrix_trained.csv', header = 0, index_col = 0)
    df = pd.read_csv('../data/trained_scores/course_info_with_survey_df_trained.csv', header=0, index_col=0)
    
    ## this section help to compute and obtain the NDCG for each query and store in df
    queryCount = 0
    NDCGDf = 0
    totalTime = 0
    for index,row in query_val.iterrows():
        query = row['querySample']
    ## do not run glove_kv, topn, expand_query for  get_NDCG_BM25WPseudo
        
    ## compute basic_bm25 score and start the timer for querying
        start = time.time()
        retrievedDocs, rankedLs = bm25_with_pseudo_relevance.bm25_pseudo_relevance_back(query, df, tf, tf_norm, idf, norm_association_matrix, vocab, avg_doc_len, k=10)
        end =  time.time()- start
        totalTime += end
    ## converting moduleNScore to dataframe    
        retrievedDf = pd.DataFrame.from_dict(retrievedDocs,orient='index',columns = ['bm25Score'])        
    ## cleaned golden/vaildation set modules
        validModules = clean_elective_names(query_val['expectedElectivesInOrder'][queryCount])
    ## to compute the NDCG of a single query
        NDCGWithVariousKdf = NDCGWithVariousK(retrievedDf,validModules,0,queryCount)
        
    ## if this is 1st NDCG score been compute, make it's df to NDCG df else merge with the current overall NDGC df
        if queryCount == 0:
            NDCGDf = NDCGWithVariousKdf
        else:
            NDCGDf = pd.merge(NDCGDf, NDCGWithVariousKdf, on=["k"])
    ## print query and NDCG score at k =10
        displayNdcgAt10(queryCount,query_val,NDCGWithVariousKdf)
    ## update the query index by 1
        queryCount += 1
        
    ## print only when the last query is computed
        if queryCount == (len(query_val)):
            averageQueryTime = totalTime/queryCount
            print('Average Time for {} number of queries : {}'.format(queryCount,averageQueryTime))
    ## return a df with all the ndcg results
    
    return(NDCGDf)
    


# %%
if __name__ == "__main__":
    # model 1 2 and 3
    import pandas as pd
    if True:
        ## model 1 Cosine Similarity (without and with query expansion, course information + (50% survey))
        print("#"*200)
        print('Calculating NDCCG for Model 1 Cosine Similarity (without query expansion, only with course information data)')
        tf = pd.read_csv('../data/course_info_scores/course_info_tf.csv', index_col = 0)
        query_val= pd.read_csv('../data/survey/vaildation_sample_query.csv',index_col = 0)
        model1NDCG = get_NDCG_cosine_no_expan(query_val,tf)
        model1NDCGAverage = model1NDCG.iloc[:, 1:].mean(axis=1)
        model1NDCGAverage.to_csv('../results/ndcg_score/ndcg_score_mdoel1.csv')
        print("Average NDCG@10 for model 1: {}".format(round(model1NDCGAverage.iloc[9],5)))
    if True:
        ## model 2 Cosine Similarity (with query expansion, only with course information data)
        print("#"*200)
        print('Calculating NDCCG for Model 2 Cosine Similarity (with query expansion, only with course information data)')
        tf = pd.read_csv('../data/course_info_scores/course_info_tf.csv', index_col = 0)
        query_val= pd.read_csv('../data/survey/vaildation_sample_query.csv',index_col = 0)
        model2NDCG = get_NDCG_cosine(query_val,tf)
        model2NDCGAverage = model2NDCG.iloc[:, 1:].mean(axis=1)
        model2NDCGAverage.to_csv('../results/ndcg_score/ndcg_score_mdoel2.csv')
        print("Average NDCG@10 for model 2: {}".format(round(model2NDCGAverage.iloc[9],5)))
        
        ## model 3 Cosine Similarity (with query expansion, with course information data and 50% of survey data)
        print("#"*200)
        print('Calculating NDCCG for Model 3 Cosine Similarity (with query expansion, with course information data and 50% of survey data)')
        tf = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_tf.csv', index_col = 0)
        query_val= pd.read_csv('../data/survey/vaildation_sample_query.csv',index_col = 0)
        model3NDCG = get_NDCG_cosine(query_val,tf)
        model3NDCGAverage = model3NDCG.iloc[:, 1:].mean(axis=1)
        model3NDCGAverage.to_csv('../results/ndcg_score/ndcg_score_mdoel3.csv')
        print("Average NDCG@10 for model 3: {}".format(round(model3NDCGAverage.iloc[9],5)))


# %%
# model 4 5 and 6
if __name__ == "__main__":
    import pandas as pd
    if False:
        ## model 4 BM25 Basic (without query expansion, course information)
        print("#"*200)
        print('Calculating NDCCG for Model 4 BM25 Basic (without query expansion, course information)')
        query_val= pd.read_csv('../data/survey/vaildation_sample_query.csv',index_col = 0)
        tf = pd.read_csv('../data/course_info_scores/course_info_tf.csv', index_col = 0)
        tf_norm = pd.read_csv('../data/course_info_scores/course_info_tf_norm.csv', index_col = 0)
        idf = pd.read_csv('../data/course_info_scores/course_info_idf.csv', header=0, index_col=0)
        model4NDCG = get_NDCG_BM25BasicNoExpan(query_val,tf,tf_norm,idf)
        model4NDCGAverage = model4NDCG.iloc[:, 1:].mean(axis=1)
        # model4NDCGAverage.to_csv('../results/ndcg_score/ndcg_score_mdoel4.csv')
        print("Average NDCG@10 for model 4: {}".format(round(model4NDCGAverage.iloc[9],5)))
        print('\n')
        
        ## model 5 BM25 Basic (query expansion, course information)
        print("#"*200)
        print('Calculating NDCCG for Model 5 BM25 Basic (query expansion, course information)')
        query_val= pd.read_csv('../data/survey/vaildation_sample_query.csv',index_col = 0)
        tf = pd.read_csv('../data/course_info_scores/course_info_tf.csv', index_col = 0)
        tf_norm = pd.read_csv('../data/course_info_scores/course_info_tf_norm.csv', index_col = 0)
        idf = pd.read_csv('../data/course_info_scores/course_info_idf.csv', header=0, index_col=0)
        model5NDCG = get_NDCG_BM25Basic(query_val,tf,tf_norm,idf)
        model5NDCGAverage = model5NDCG.iloc[:, 1:].mean(axis=1)
        # model5NDCGAverage.to_csv('../results/ndcg_score/ndcg_score_mdoel5.csv')
        print("Average NDCG@10 for model 5: {}".format(round(model5NDCGAverage.iloc[9],5)))
        print('\n')
        
        ## model 6 BM25 Basic (query expansion, course information + survey (50%))
        print("#"*200)
        print('Calculating NDCCG for Model 6 BM25 Basic (query expansion, course information + survey (50%))')
        query_val= pd.read_csv('../data/survey/vaildation_sample_query.csv',index_col = 0)
        tf = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_tf.csv', index_col = 0)
        tf_norm = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_tf_norm.csv', index_col = 0)
        idf = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_idf.csv', header=0, index_col=0)
        model6NDCG = get_NDCG_BM25Basic(query_val,tf,tf_norm,idf)
        model6NDCGAverage = model6NDCG.iloc[:, 1:].mean(axis=1)
        # model6NDCGAverage.to_csv('../results/ndcg_score/ndcg_score_mdoel6.csv')
        print("Average NDCG@10 for model 6: {}".format(round(model6NDCGAverage.iloc[9],5)))
        print('\n')


# %%
if __name__ == "__main__":
    import pandas as pd
    if False:
        ## to create the function to obtain the NDCG for model 7 and 8
        ## model 7 Bm25 with Reformulation (query expansion, course information + survey (50%))
        print("#"*200)
        print('Calculating NDCCG for Model 7 Bm25 with Reformulation (query expansion, course information + survey (50%))')
        query_val= pd.read_csv('../data/survey/vaildation_sample_query.csv',index_col = 0)
        tf = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_trained.csv', index_col = 0)
        tf_norm = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_norm_trained.csv', index_col = 0)
        idf = pd.read_csv('../data/trained_scores/course_info_with_survey_idf_trained.csv', header=0, index_col=0)
        model7NDCG = get_NDCG_BM25Basic(query_val,tf,tf_norm,idf)
        model7NDCGAverage = model7NDCG.iloc[:, 1:].mean(axis=1)
        # model7NDCGAverage.to_csv('../results/ndcg_score/ndcg_score_mdoel7.csv')
        print("Average NDCG@10 for model 7: {}".format(round(model7NDCGAverage.iloc[9],5)))
        print('\n')
        
        ## model 8 Bm25 with Reformulation and Pseudo Relevance Feedback (query expansion, course information + survey (50%))
        print("#"*200)
        print('Calculating NDCCG for Model 8 Bm25 with Reformulation and Pseudo Relevance Feedback (query expansion, course information + survey (50%))')
        query_val= pd.read_csv('../data/survey/vaildation_sample_query.csv',index_col = 0)
        tf = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_trained.csv', index_col = 0)
        tf_norm = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_norm_trained.csv', index_col = 0)
        idf = pd.read_csv('../data/trained_scores/course_info_with_survey_idf_trained.csv', header=0, index_col=0)
        model8NDCG = get_NDCG_BM25WPseudo(query_val,tf,tf_norm,idf)    
        model8NDCGAverage = model8NDCG.iloc[:, 1:].mean(axis=1)
        # model8NDCGAverage.to_csv('../results/ndcg_score/ndcg_score_mdoel8.csv')
        print("Average NDCG@10 for model 8: {}".format(round(model8NDCGAverage.iloc[9],5)))
    print('\n')


