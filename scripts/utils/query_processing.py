import numpy as np
import string
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    '''
    Identify the type of word in the sentence to generate the appropriate lemitsation 
    Map POS tag to first character lemmatize() accepts
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def process_query(query):
    '''
    Takes in input query. (string)
    Performs punctuation removal, lower casing, stopword removal, lemitistion. 
    Converts into a list of words. (list)
    '''
    query = query.translate(str.maketrans('', '', string.punctuation))
    query = str(np.char.lower(query))
    query = ''.join([i for i in query if not i.isdigit()])
    query = remove_stopwords(query)
    query = [lemmatizer.lemmatize(w, get_wordnet_pos(w))
            for w in nltk.word_tokenize(query)]
    query = ' '.join([str(elem) for elem in query])
    query = query.translate(str.maketrans('', '', string.punctuation))
    return query.split()

def expand_query(query,glove_kv,topn):
    '''
    Expands query (list) with a pretrained glove 6B 300d corpus. 
    Takes each word and expand the word by top n most similar word.
    '''
    model = KeyedVectors.load(glove_kv)
    expanded_query = []
    for word in query:
        expanded_query.append(word)
        if word in model:
            for close_word in model.most_similar(word, topn = topn):
                expanded_query.append(close_word[0])
    return expanded_query

