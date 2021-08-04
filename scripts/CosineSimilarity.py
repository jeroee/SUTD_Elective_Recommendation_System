# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
"""[summary]
The script is to compute the cosine simliarities between queries and the document
The steps are as follows:
1. read the tf file
2. obtaint the input query
3. compute the cosine simliarities between query and each document
4.sort by the one with highest simliarity with the highest rank

"""


# %%
def readFile(_filePath):
    """[summary]

    Args:
        _filePath ([type]): [description]
    """
    import pandas as pd
    readFile = pd.read_csv(_filePath,index_col=0)
    return(readFile)


# %%
def queryVector(query,corpus):
    """[summary]
    vectorize query based on corpus
    
    Args:
        query ([list]): [list containing query terms]
    """
    corpusList = list(corpus.index)
    ## vector to track the corpusList
    vector = [0]*len(corpusList)
    ## format query\r\n",
    ## import scripts to process queries\r\n"
    import utils.query_processing
    ## convert to string if not string\r\n"
    if type(query) != str:
        query = ' '.join(query)
    ## clean the query
    
    query = utils.query_processing.process_query(query)

    "still lacking the query expansion"
    "Need  glove_kv and topn to be saved in the repo"
    glove_kv = '../pretrained_corpus/glove_6B_300d.kv'
    query = utils.query_processing.expand_query(query,glove_kv,3)    
    
    ## iterate over query terms
    for wordOfQuery in query:
    ## variable to tract the corpus index
        vectorIndex = 0
    ## iterate over the all corupus
        for wordOfCorpus in corpusList:
            if wordOfQuery ==  wordOfCorpus:
                vector[vectorIndex] += 1 
    ## update courpus index
            vectorIndex += 1               
    return(vector)


# %%
def cosineSimilarities(queryVector,docVector):
    "    [summary]"
    "    To compute the cosine similarity of the query with a single column/module in corpus"
    "    Args:"
    "        queryVector ([list]): [query]"
    "        docVector ([list]): [description]"
    from scipy import spatial
    import warnings
    warnings.filterwarnings("ignore")
    distance = 1 - spatial.distance.cosine(queryVector, docVector)
    return(round(distance,5))


# %%
def clean_elective_names(relevant_results):
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
        relevant_results.append('40.305 Advanced Topics in Stochastic Modelling#')
    return relevant_results


# %%
def topModulesForEachQuery(_df):
    "[summary]\r\n",
    "    To obtain the top modules for each of the query\r\n"
    "    Args:\r\n"
    "        _df ([dataframe]): [description] data with rows of words and columns contain the modules, the cell of the dataframe is the similarity\r\n"
    df = _df.copy(deep = False)
    overallTopModules = {}
    for query in _df.columns:
        singleModule = {}
        df.sort_values(by = query,ascending = False, inplace = True)
        ## obtain top 10 modules
        topModules = df[query][:10]
        singleModule['topModules'] = list(topModules.index)
        topModules = clean_elective_names(list(topModules))
        singleModule['topModulesScore'] = list(topModules)
        overallTopModules[str(query)] = (singleModule)
        
    import pandas as pd
    overallTopModulesDf = pd.DataFrame(overallTopModules)
    return(overallTopModulesDf)


# %%
def rankedModuleOfCosineSim(queries,corpus):
    """[summary]
    function to compute cosine simliar for each query and obtaining the relvant module
    Args:
        queries ([list of list]): [description] it contain a list of string or list of list of queries
        corupus ([dataframe]): [description] it is a tf document with row index as words and columns headers as modules
    """
    import time
    start=time.time()
    import pandas as pd 
    ## iterate over the vaildationQueries
    queriesScore = {}
    for row,query in queries.T.iteritems():
        print('\nCurrent computing Query: {}'.format(query['querySample']))
        singleQueryVector = queryVector(query['querySample'],corpus)
        print("Number of terms in corpus: {}".format(sum(singleQueryVector)))
        
        CosineSimilaritiesScore = []
        for modules in list(corpus):
            SimilaritiesScore = cosineSimilarities(singleQueryVector,corpus[modules])
            CosineSimilaritiesScore.append(SimilaritiesScore)
            # print("Score for Query on {} : {}".format(modules,SimilaritiesScore))
        queriesScore[str(query['querySample'])] = CosineSimilaritiesScore
        
    ## Convert to dict to df
    modules = list(corpus)
    df  = pd.DataFrame.from_dict(queriesScore,orient='index',columns=modules)
    df = df.T
    
    ##obtaining top modules for each query
    rankedModule = topModulesForEachQuery(df)
    end = time.time()- start
    averageQueringTime = end/ len(queries)
    print("################################")
    print("average Querying Time: {}".format(averageQueringTime))
    return(rankedModule)


# %%
if __name__ == "__main__":
    import pandas as pd 
    corpus = readFile('../data/course_info_scores/course_info_tf.csv') ## <-- assign the merge corupus here
    # corpus = readFile('../data/course_info_with_survey_scores/course_info_with_survey_tf.csv') ## <-- assign the merge corupus here
    
    ## vaildation set
    """[summary]
    U can control which vaildation set to choose to test the cosine similarity model
    True : would be for the testingQuery that is self create which ensure a value when computing cosineSimilarities
    False : obtain the query based on 50% of vaildation dataset which might have all cosine similarities of zero
    """
    if False:
        testingQuery1 = ['ability','able','abstract']
        testingQuery2 = ['abstraction','accommodate','accompanies']
        testingQuery3 = ['account','accounting','achieve']
        testingQueryList = [[testingQuery1],[testingQuery2],[testingQuery3]]
        testingQueryDf = pd.DataFrame.from_records(testingQueryList,columns = ['query Sample'])
        queries = testingQueryDf
        
    else:
        queries = readFile('../data/survey/vaildation_sample_query.csv')    
    
    ## called function to obtain df for top modules and their score
    topModulesDf = rankedModuleOfCosineSim(queries,corpus)
    topModulesDf
    