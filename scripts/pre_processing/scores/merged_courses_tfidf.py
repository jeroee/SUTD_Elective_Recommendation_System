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
input_file = '../../../data/scrap_processing/merged_courses.json'
output_file = '../../../data/bm25/bm25_no_survey/merged_courses_tfidf.csv'

# open json file
with open(input_file) as f:
  istd_courses = json.load(f)

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



list_of_courses = []    # course names
list_of_course_words = []   # words about course
for idx,course in enumerate(istd_courses):
    course_name, s = get_unique_words(course)
    list_of_courses.append(course_name)
    list_of_course_words.append(s)

tfidfvectorizer = TfidfVectorizer(analyzer='word')  # import tfidf vectorizer from sklearn
tfidf_wm = tfidfvectorizer.fit_transform(list_of_course_words)  # get tf-idf score matrix
tfidf_tokens = tfidfvectorizer.get_feature_names()  # get all the unique words
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = list_of_courses,columns = tfidf_tokens).T
df_tfidfvect.to_csv(output_file)



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

tf_vect.to_csv('../../../data/bm25/bm25_no_survey/merged_courses_tf.csv')
tf_vect_norm.to_csv('../../../data/bm25/bm25_no_survey/merged_courses_tf_norm.csv')


# https://stackoverflow.com/questions/26053849/counting-non-zero-values-in-each-column-of-a-dataframe-in-python
df_arr = np.count_nonzero(tf_vect, axis=1) # compute df: count the nunber of non zero columns in each role 
df_vect = pd.DataFrame(data={'df':df_arr}, index = tf_tokens)
df_vect['df'] = df_vect['df'].apply(lambda freq: math.log10((corpus_size) / (freq))) # calc idf
idf_vect = df_vect.rename({'df': 'idf'}, axis=1, inplace=False)

idf_vect = idf_vect.replace(np.nan, 0)
idf_vect.to_csv('../../../data/bm25/bm25_no_survey/merged_courses_idf.csv')



