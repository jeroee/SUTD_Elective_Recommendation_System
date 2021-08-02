# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## The purpose of the script
# This script is to convert the text from the suvery into word from suvery to modules 
# <br>
# This is to merge 50% of the suvery to training dataset

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
def textCleaning(s):
    ## spacy tutorial https://www.analyticsvidhya.com/blog/2020/03/spacy-tutorial-learn-natural-language-processing/

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
formattingSurveyDataDict = formattingSurveyData(suveryMerge)
formattingSurveyDataDict


# %%
def formattingTable(_dict):
    import pandas as pd
    wordDf = pd.DataFrame()
    for k,v in _dict.items():
        for k2,v2 in v.items():
            wordDf.at[ k2 ,k ] = v2
    wordDf.fillna(0, inplace = True)

    return(wordDf)


# %%
formattingTableDf = formattingTable(formattingSurveyDataDict)
formattingTableDf


# %%
formattingTableDf.to_csv('../../data/survey/survey_for_merging_converted.csv')

# %% [markdown]
# # Cleaning the the index words
# <br> don't use the function below, there are some faults with the code

# %%
from gensim.parsing.preprocessing import remove_stopwords
import re
import numpy as np
dict={}
for column in formattingTableDf.columns[1:]:
    # print(column)
    dict[column]=[]
    for index,row in formattingTableDf.iterrows():
        text = index
        text = str(np.char.lower(text))
        text = re.sub(r'[^\w]', ' ', text)
        text = re.sub(' +', ' ', text)
        text = remove_stopwords(text)
        text = text.split(' ')
        for i in range(int(row[column])):
            for word in text:
                dict[column].append(word)

# adding more weight by doubling the relevant word counts for each course
for i,j in dict.items():
    dict[i] = j*2

with open('../../data/survey/merged_survey.json','w') as file:
    json.dump(dict, file)


