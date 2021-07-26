# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## This script is to obtain query samples types

# %%
import pandas as pd
from os import path
## inorder to obtain the relative file path
## https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python 
## ../ refer to looking into the folder above
relativePath = path.relpath("../../data/survey/CustomerFeedback(Responses).xlsx")
## require the engine = 'openpyxl' to read the xlsx
df = pd.read_excel(relativePath, index_col = 0,engine="openpyxl")
df.head()


# %%
## data cleaning
## drop from column
df.dropna(axis= 1, how = 'all', inplace= True)
## drop from row
df.dropna(axis= 0, how = 'all', inplace= True)
## replace column value to 1 if 'Check if taken' or ' Check if taken or taking during Term 6 to 8'
df.replace('Check if taken', 1, inplace = True)
df.replace('Check if taken or taking during Term 6 to 8', 1, inplace = True)
df.describe(include ='all')


# %%
import random
random.seed(123)
suveryMerge = df.sample(frac=0.5,random_state=200) #random state is a seed value
suveryFeedbackNValid= df.drop(suveryMerge.index)
suveryFeedback= suveryFeedbackNValid.sample(frac=0.5,random_state=200)
suveryVaild= suveryFeedbackNValid.drop(suveryFeedback.index)


# %%
## Saving the splitted suvery into a csv for training phase and one for vaildation phase
suveryMerge.to_csv('../../data/survey/survey_for_merging.csv')
suveryFeedback.to_csv('../../data/survey/Survey_for_feedback.csv')
suveryVaild.to_csv('../../data/survey/survey_for_vaild.csv')


# %%
suveryMerge.head()

# %% [markdown]
# # Formatting the dataset

# %%
def obtainingElectives(_df):
    print('Starting to obtain dict of electives')
    electiveDict = {}
    for columnName in _df.columns:
        if 'Elective Modules taken or taking' in columnName:
            openSquareBracket = columnName.find('[') + 1
            closeSquareBracket = columnName.find(']')
            electiveDict[columnName] = columnName[openSquareBracket:closeSquareBracket]
    return(electiveDict)


# %%
def updatingdict(_df,electiveWordsDict,key):
    key = str(key)
    ## if there is no word in columnname
    if key not in electiveWordsDict.keys():
        electiveWordsDict[key] = len(_df)
    else:
        electiveWordsDict[key] = electiveWordsDict[key] + len(_df)
    return(electiveWordsDict)


# %%
def obtainingDetailsFromPillar(_df,electiveWordsDict,columnOfInterest):
    ## filtering the _df
    pillarDf = _df.copy(deep = False)
    print('Obtaining from "{}" column'.format(columnOfInterest))
    ## number of times the word/skills appears
    for pillar in pillarDf[columnOfInterest].unique():
        singlePillarDf = pillarDf[pillarDf[columnOfInterest] == pillar]
        electiveWordsDict =  updatingdict(singlePillarDf,electiveWordsDict,pillar)
    return(electiveWordsDict)


# %%
def obtainingDetailsFromTrack(_df,electiveWordsDict,columnOfInterest):
    print('Obtaining from "{}" column'.format(columnOfInterest))
    for columnName in _df.columns:
    ## ensure a part of the columns name contains the columnOfInterest
        if columnOfInterest in columnName:
    ## filtered df
            singleTrackDf = _df[_df[columnName] == 1]
    ## finding the index of []
            openSquareBracket = columnName.find('[') + 1
            closeSquareBracket = columnName.find(']')
    ## if the columnName does not have bracket then do not shorten the columnName
            if (openSquareBracket > 0) & (closeSquareBracket > 0):
                columnName = columnName[openSquareBracket:closeSquareBracket]
    ## number of times the word/skills appears
            electiveWordsDict =  updatingdict(singleTrackDf,electiveWordsDict,columnName)
    return(electiveWordsDict)


# %%
def textCleaning(_string):
    ## spacy tutorial https://www.analyticsvidhya.com/blog/2020/03/spacy-tutorial-learn-natural-language-processing/
    import spacy
    import re
    import string
    nlp = spacy.load("en_core_web_trf") 
    #create nlp object
    _string = re.sub('[,!@#$-]', '', _string)
    doc = nlp(_string.lower())
    doc = [token for token in doc if not token.is_stop]
    return(doc)


# %%
def obtainingDetailsFromNonNumericalColumnOfInterest(_df,electiveWordsDict,columnOfInterest):
    print('Obtaining from "{}" column'.format(columnOfInterest))
    ## filter by those elective or columns which shows skills or interest
    df = _df.copy(deep = False)
    df = df[columnOfInterest]
    df.dropna(axis= 0, how = 'all', inplace= True)
    for skillOrInterest in df:
        skillOrInterest = textCleaning(skillOrInterest)
        print(skillOrInterest)
    ## number of times the word/skills appears
    ## just use a df of length 1 as we are checking one row of entry at a time
        dfOfLength1 = df.head(1)
        for word in skillOrInterest:
            electiveWordsDict = updatingdict(dfOfLength1,electiveWordsDict,word)
    return(electiveWordsDict) 


# %%
def formattingSurveyData(_df):
    df = _df.copy(deep = False)
    ## obtainig electives names from suvery
    electiveDict = obtainingElectives(df)
    ## dict to store the words from suvery about all elective
    allElectiveDict = {}
    ## iterate through the electives
    for k,v in electiveDict.items():
    ## filter by electives
        electiveDf = df[df[k] == 1]
    ## dict to store the words from suvery about all elective
        electiveWordsDict = {}
        print('\n')
    ## check the pillar, focus track, skills, interest and knowledge gain that are remaining after filtering
    ## count the occurances of them happening
    ## convert the those into words to be index of the rows (including elective name)
        pillar = obtainingDetailsFromPillar(electiveDf,electiveWordsDict,'Pillar')
        focusTrack = obtainingDetailsFromTrack(electiveDf,electiveWordsDict,'Focus Track')
        skillsFromEachElective = obtainingDetailsFromNonNumericalColumnOfInterest(electiveDf,electiveWordsDict,v)
    ## add to the overall
        allElectiveDict[v] = electiveWordsDict
    return(allElectiveDict)


# %%
def formattingTable(_dict):
    import pandas as pd
    wordDf = pd.DataFrame()
    for k,v in _dict.items():
        for k2,v2 in v.items():
            wordDf.at[ k2 ,k ] = v2
    wordDf.fillna(0, inplace = True)

    return(wordDf)

# %% [markdown]
# # Cleaning the the index words

# %%
def dataCleaningV2(s):
    import json
    import string
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    from gensim.parsing.preprocessing import remove_stopwords
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    import numpy as np

    lemmatizer = WordNetLemmatizer()
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    s = str(np.char.lower(s))
    s = ''.join([i for i in s if not i.isdigit()])
    s = remove_stopwords(s)
    s = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(s)]
    s = ' '.join([str(elem) for elem in s])
    s = s.translate(str.maketrans('','',string.punctuation))
    s = s.split()

    return(s)


# %%
def cleaningIndexedWord(_formattingTableDf):
    print('Starting to CleanIndexWords in to a single word for each row')
    print('There are currently: {}\n'.format(len(_formattingTableDf)))
    import pandas as pd
    newDf = pd.DataFrame(columns=_formattingTableDf.columns)
    ## counter to track which row it is at
    counter = 0

    for index, singleWordMertics in _formattingTableDf.iterrows():
        # print(index.lower())
        # print(singleWordMertics)
        cleanedDoc = dataCleaningV2(index)
        for word in cleanedDoc:
            for elective, count in singleWordMertics.items():
                newDf.at[ word ,elective ] = count
    ## tracker
        counter += 1
        if counter % 10 == 0:
            print('Currently Done {}%'.format(round(counter*100/len(_formattingTableDf),2))) 
    return(newDf)

# %% [markdown]
# # Sample Type

# %%
def sampleTypeQuery(_formattingTableDf, numOfSamples = 1, portionFromWordFromSurvey = 1 ,maxWords = 5, setRandom = 1):
    ## output example
    ## I am planning to go 'ESD' interested in Finances and wanting to learn python and R
    ## have synonyms using the parrot library in python
    ## this function is to obtain the same query example where half of the interest used would be based on half of the survey based on trging data
    import random
    import math
    from random_word import RandomWords
    r = RandomWords()
    if setRandom == 0:
        random.seed(123)

    sampleDict = {}
    
    for sample in range(numOfSamples):
        counter = 0
        lsOfWords = []
        queryAndResult =[]
        indexWord = list(set(_formattingTableDf.index))

        while counter < maxWords*portionFromWordFromSurvey:
            counter += 1
            wordIndex = random.randint(0, len(_formattingTableDf)-1)
            lsOfWords.append(indexWord[wordIndex])

        if portionFromWordFromSurvey < 1:
            rubbishWordCounter = 0
            while rubbishWordCounter < math.ceil(maxWords *(1-portionFromWordFromSurvey)):
                rubbishWordCounter += 1
                rubbishWord = r.get_random_word()
                while rubbishWord is None:
                    rubbishWord = r.get_random_word()
                lsOfWords.append(rubbishWord)

        ## reform lsofWords to strOfWords
        separator = ', '
        strOfWords = separator.join(lsOfWords)
        
        ## obtaining top 10 
        OnlyDfFromWordDf = _formattingTableDf.filter(items=lsOfWords, axis = 0)
        top10Sample = OnlyDfFromWordDf.sum().sort_values(ascending = False)[:10]
        
        
        sampleDict[sample] = [strOfWords,list(top10Sample.keys()), list(top10Sample.values)]

    ## convert dict to df
    SampleDf = pd.DataFrame.from_dict(sampleDict, orient='index', columns=['querySample','expectedElectivesInOrder','expectedElectivesInOrderSumWordCount'])
    SampleDf['queryType'] = portionFromWordFromSurvey
    return(SampleDf)


# %%
## processing the training Phase sampleQuery for feedback
try:
    suveryFeedbackDict = formattingSurveyData(suveryFeedback)
    suveryFeedbackDf = formattingTable(suveryFeedbackDict)
    suveryFeedbackDf = cleaningIndexedWord(suveryFeedbackDf)
    suveryFeedbackDf.to_csv('../../data/survey/survey_for_feedback_converted.csv')
except:
    import pandas as pd
    suveryFeedbackDf = pd.read_csv('../../data/survey/survey_for_feedback_converted.csv', index_col= 0)
    suveryFeedbackDf.fillna(0,inplace = True)


# %%
suveryFeedbackDf.to_csv('../../data/survey/training_sample_query.csv')


# %%
## training sample
trainingSampleDf = sampleTypeQuery(suveryFeedbackDf,30,1,5,0)
trainingSampleDf.head()


# %%
# export the the trainingSample
# trainingSampleDf.to_csv('querySampleForTrainingPhase/TrainingSampleQuery.csv')


# %%
## processing the vaildation phase sampleQuery
try:
    suveryVaildDict = formattingSurveyData(suveryVaild)
    suveryVaildDf = formattingTable(suveryVaildDict)
    suveryVaildDf = cleaningIndexedWord(suveryVaildDf)
    suveryVaildDf.to_csv('../../data/survey/survey_for_vaild_converted.csv')
except:
    import pandas as pd
    suveryVaildDf = pd.read_csv('../../data/survey/survey_for_vaild_converted.csv', index_col= 0)
    suveryVaildDf.fillna(0,inplace = True)


# %%
## export the the trainingSample
SampleType1Df = sampleTypeQuery(suveryVaildDf,30,1,5,0)
# SampleType2Df = sampleTypeQuery(suveryVaildDf,5,0.6,5,0)
# SampleType3Df = sampleTypeQuery(suveryVaildDf,5,0,5,0)


# %%
SampleType1Df.head()

# %% [markdown]
# # combining all the querySampleType

# %%
## combine the different types of queries if required
frames = [SampleType1Df]
combinedSamples = pd.concat(frames)
combinedSamples.reset_index(drop=True, inplace= True)
combinedSamples.head()


# %%
# combinedSamples.to_csv('../../data/survey/vaildation_sample_query.csv')


