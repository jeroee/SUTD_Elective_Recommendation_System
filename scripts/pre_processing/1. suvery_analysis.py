# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import pandas as pd
from os import path
## inorder to obtain the relative file path
## https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python 
## ../ refer to looking into the folder above
relativePath = path.relpath("../../data/survey/CustomerFeedback(Responses).xlsx")
## require the engine = 'openpyxl' to read the xlsx
suvery = pd.read_excel(relativePath, index_col = 0,engine="openpyxl")
suvery.head()


# %%
## data cleaning
## drop from column
suvery.dropna(axis= 1, how = 'all', inplace= True)
## drop from row
suvery.dropna(axis= 0, how = 'all', inplace= True)
## replace column value to 1 if 'Check if taken' or ' Check if taken or taking during Term 6 to 8'
suvery.replace('Check if taken', 1, inplace = True)
suvery.replace('Check if taken or taking during Term 6 to 8', 1, inplace = True)
suvery.describe(include ='all')


# %%
suvery.describe(include ='all')['40.240 Investment Science']

# %% [markdown]
# # Data Anlaysis of Suvery

# %%
## Analysis of What are your academic/career interest?
columnsOfInterest = suvery.columns[-2]
suvery[columnsOfInterest].value_counts(normalize = True)


# %%
## What are your skills of interest?
columnsOfInterest = suvery.columns[-1]
suvery[columnsOfInterest].value_counts(normalize = True)


# %%
suvery.columns
'suvery columns'


# %%
groupByPillar = suvery.groupby(by = ['Pillar']).count()


# %%
import matplotlib.pyplot as plt
FocusTrack = []
for columnName in suvery.columns:
    if 'Focus Track' in columnName:
        FocusTrack.append(columnName)

groupByPillar = suvery.groupby(by = ['Pillar']).count()
groupByPillar.plot.bar(y = FocusTrack,figsize = (8,4.5),rot=0, title = 'FocusTrack', grid = True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True, ncol=1)


# %%
ESDElective = []
for columnName in suvery.columns:
    if ('ESD Elective' in columnName) | ('ESD TAE Modules' in columnName):
        ESDElective.append(columnName)

groupByPillar = suvery.groupby(by = ['Pillar']).count()
groupByPillar.plot.bar(y = ESDElective,figsize = (16,9),rot=0, grid = True, title = 'Num Students based on sorted by pillar taking ESD Electives')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True, ncol=1)


# %%
ISTDElective = []
for columnName in suvery.columns:
    if ('ISTD Elective' in columnName) | ('ISTD TAE Modules' in columnName):
        ISTDElective.append(columnName)
groupByPillar.plot.bar(y = ISTDElective,figsize = (16,9),rot=0, grid = True, title = 'Num Students based on sorted by pillar taking ISTD Electives')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True, ncol=1)

# %% [markdown]
# 

# %%
groupByPillar

# %% [markdown]
# # Queries Sampling
# This Section is to obtain the different types of queries based on the suvery

# %%
def textCleaning(_string):
    ## spacy tutorial https://www.analyticsvidhya.com/blog/2020/03/spacy-tutorial-learn-natural-language-processing/
    import spacy
    import re
    nlp = spacy.load('en_core_web_sm') 
    #create nlp object
    _string = re.sub('[,!@#$]', '', _string)
    doc = nlp(_string)
    doc = [token for token in doc if not token.is_stop]
    return(doc)


# %%
def extractFocusTrackInElectives(_singleElectiveDf,_singleElectivesLs):
    ## modules this track students took
        singleElectiveDf = _singleElectiveDf.dropna(axis= 1, how = 'all').copy()
    ## this have not factor what the electives mentioned in the Other electives
        trackLs = []
        for focusTrack in singleElectiveDf.columns:
            if ('Focus Track' in focusTrack): 
                trackLs.append(focusTrack)
        _singleElectivesLs.append(trackLs)


# %%
def extractingOpenEndedInput(_df,_ls,columnName):
    ## check the different inputs from What are your academic/career interest?
        interestOrSkills = list(set(_df[columnName]))
        interestOrSkills = [x.lower() for x in interestOrSkills] 
    ## tokenization and textCleaning to add value of the elective
        tokenization =[]
        for text in interestOrSkills:
            tokenized = textCleaning(text)
            tokenization.extend(tokenized)        
        tokenization = list(set(tokenization))
        _ls.append(tokenization)


# %%
def extractingRelevantElectiveIntertestSkills(_df):
    ## can add the count of the terms that appearing for each combination

    import pandas as pd
    ## check what focus track relates to the inputs in What are your academic/career interest?
    electivesLs = []
    for elective in _df.columns:
        if ('Elective' in elective) | ('TAE Modules' in elective):
            electivesLs.append(elective)

    ## dict to save all the focus track elective, interest and skills
    allElectivesDict = {}
    ## check What are your academic/career interest? input selected the focus track
    ## iterating from different columns to check what modules relate to them
    for electives in electivesLs:
    ## dict to store the electives, interest and skills for one track
        singleElectivesLs = []
    ## students taking this track students
    ## if electives == 1 
        electivesDf = _df[_df[electives] == 1]
        extractFocusTrackInElectives(electivesDf,singleElectivesLs)
    ## check the different inputs from 'What are your academic/career interest?'
        extractingOpenEndedInput(electivesDf,singleElectivesLs,'What are your academic/career interest?')
    ## check the different inputs from 'What are your skills of interest?'
        extractingOpenEndedInput(electivesDf,singleElectivesLs,'What are your skills of interest?')
    ## appending to allElectivesDict
        allElectivesDict[electives] = singleElectivesLs

    ## convert dict to df
    allElectivesDf = pd.DataFrame.from_dict(allElectivesDict, orient = 'index', columns =['FocusTracks','interest','skills'])

    return(allElectivesDf)


# %%
# extractingRelevantElectiveIntertestSkillsDf = extractingRelevantElectiveIntertestSkills(suvery)
# extractingRelevantElectiveIntertestSkillsDf


# %%
# ## export csv
# relativeSaveCSVPath = path.relpath("../../data/suveryData/suveryDataSorted.xlsx")
# extractingRelevantElectiveIntertestSkillsDf.to_excel(relativeSaveCSVPath)


