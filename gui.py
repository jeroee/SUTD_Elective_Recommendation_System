from flask import Flask, request, jsonify, render_template
import pandas as pd
from scripts.basic_bm25 import get_result
from scripts.utils.query_processing import process_query, expand_query
import itertools
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

tf_with_survey = pd.read_csv(
    'data/course_info_with_survey_scores/course_info_with_survey_tf.csv', header=0, index_col=0)
tf_norm_with_survey = pd.read_csv(
    'data/course_info_with_survey_scores/course_info_with_survey_tf_norm.csv', header=0, index_col=0)
idf_with_survey = pd.read_csv(
    'data/course_info_with_survey_scores/course_info_with_survey_idf.csv', header=0, index_col=0)

vocab = tf_with_survey.index.tolist()  # unique words
total_length = tf_with_survey.to_numpy().sum()
# average document length across all courses
avg_doc_len = total_length / len(tf_with_survey.columns)


glove_kv = 'pretrained_corpus/glove_6B_300d.kv'


@app.route('/')  # Homepage
def home():
    return render_template('index.html', )


@app.route('/get_relevant_courses', methods=['POST'])
def get_relevant_courses():

    user_input = [x for x in request.form.values()]

    query = user_input[0]
    query = process_query(query)
    query = expand_query(query, glove_kv, topn=3)
    result_dic, result_ls = get_result(query=query, tf=tf_with_survey, tf_norm=tf_norm_with_survey,
                                       idf=idf_with_survey, vocab=vocab, avg_doc_len=avg_doc_len)

    top_k = 10
    result_dic = dict(itertools.islice(result_dic.items(), top_k))
    display_info = []
    rank = 0
    for result, score in result_dic.items():
        if score != 0:
            rank += 1
            d = {'rank': rank, 'result': result,
                 'score': '%s' % float('%.1g' % score)}
            # display_info.append(f"{result}:  {score}")
            display_info.append(d)

    return render_template('index.html', retrieved_courses=display_info)


if __name__ == '__main__':
    app.run(debug=True)
# I am planning to go 'ESD' interested in Finances and wanting to learn python and R
