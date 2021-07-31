import json
import string
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import re
import math
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()

# declare input file to load and output file to save
course_info_path = '../data/web_scrap/course_info.json'
survey_info_path = '../data/survey/merged_survey.json'
tf_path = '../data/course_info_with_survey_scores/course_info_with_survey_tf.csv'
tf_norm_path = '../data/course_info_with_survey_scores/course_info_with_survey_tf_norm.csv'
df_path = '../data/course_info_with_survey_scores/course_info_with_survey_df.csv'
idf_path = '../data/course_info_with_survey_scores/course_info_with_survey_idf.csv'
tfidf_path = '../data/course_info_with_survey_scores/course_info_with_survey_tfidf.csv'

# open json file
with open(course_info_path) as f:
  course_info = json.load(f)

# defining speech tagging to words to provide better lemmatization
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# get unique words on a istd course
def get_unique_words(course):
    '''
    generate a list of unqique words for each course
    '''
    course_name = course.pop('name', None)    # remove course name
    pre_requisite = course.pop('pre_requisite', None)   # remove pre-req
    topics = course.pop('topics_covered', None) # remove topics covered
    for key,value in course.items():
        # print(key)
        if value == None:
            s = ''
        if type(value) == str:
            s = value.replace('-',' ')
            s = s.translate(str.maketrans('','',string.punctuation))
        if type(value) == list:
            if len(value)!=0:
                for item in value[0]:
                    item = item.replace('-',' ')
                    item = item.translate(str.maketrans('','',string.punctuation))
                    s += (' '+item)
    s = str(np.char.lower(s))
    s = ''.join([i for i in s if not i.isdigit()])
    s = remove_stopwords(s)
    # https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    s = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(s)]
    s = ' '.join([str(elem) for elem in s])
    s = s.translate(str.maketrans('','',string.punctuation))
    return course_name,s 

with open(survey_info_path) as f:
  survey_results = json.load(f)


# for cleaning, rename the survey title to the module name
survey_results['50.035 Computer Vision'] = survey_results.pop(" 50.035 Computer Vision")
survey_results['50.043 Database Systems'] = survey_results.pop("50.043 Database Systems / Database and Big Data Systems (for class 2021)")
opti_stochastic = survey_results.pop("40.302 Advanced Optim/ 40.305 Advanced Stochastic")
survey_results['40.302 Advanced Topics in Optimisation#'] = opti_stochastic
survey_results['40.305 Advanced Topics in Stochastic Modelling#'] = opti_stochastic


''''
To geneerate tf,df,idf,tf_norm,tf_idf from course info data merged with the survey scores
'''
list_of_courses = []    # course names
list_of_course_words = []   # words about course
for idx,course in enumerate(course_info):
    course_name, s = get_unique_words(course)
    list_of_courses.append(course_name)
    # merging data from survey into course info
    if course_name in survey_results:
        for survey_word in survey_results[course_name]:
            s += ' ' + survey_word
    list_of_course_words.append(s)

tfidfvectorizer = TfidfVectorizer(analyzer='word')  # import tfidf vectorizer from sklearn
tfidf_wm = tfidfvectorizer.fit_transform(list_of_course_words)  # get tf-idf score matrix
tfidf_tokens = tfidfvectorizer.get_feature_names()  # get all the unique words
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = list_of_courses,columns = tfidf_tokens).T
df_tfidfvect.to_csv(tfidf_path)

from IPython.display import display
corpus_size = len(list_of_courses)
count_vectorizer = CountVectorizer(analyzer='word')
tf_wm = count_vectorizer.fit_transform(list_of_course_words)
tf_tokens = count_vectorizer.get_feature_names()
tf_arr = tf_wm.toarray()
tf_vect = pd.DataFrame(data={list_of_courses[i]:tf_arr[i] for i in range(corpus_size)}, index = tf_tokens)
tf_vect_norm = tf_vect.apply(lambda x: x/x.max(), axis=0) # normalise each column by dividing the max frequency

tf_vect = tf_vect.replace(np.nan, 0) # replace nan with 0, coz 50.047 Mobile Robotics have no scriped data. Hence all column is nan
tf_vect_norm = tf_vect_norm.replace(np.nan, 0)

tf_vect.to_csv(tf_path)
tf_vect_norm.to_csv(tf_norm_path)

# https://stackoverflow.com/questions/26053849/counting-non-zero-values-in-each-column-of-a-dataframe-in-python
df_arr = np.count_nonzero(tf_vect, axis=1) # compute df: count the nunber of non zero columns in each role 
df_vect = pd.DataFrame(data={'df':df_arr}, index = tf_tokens)
df_vect.to_csv(df_path)
df_vect['df'] = df_vect['df'].apply(lambda freq: math.log10((corpus_size) / (freq))) # calc idf
idf_vect = df_vect.rename({'df': 'idf'}, axis=1, inplace=False)

idf_vect = idf_vect.replace(np.nan, 0)
idf_vect.to_csv(idf_path)