from MAP import get_map, get_map_pseudo_relevance
import pandas as pd
def get_map_scores():

    '''
        Print out the Mean Average Precision for scenario
    '''
    print("#"*200)
    print('Calculating Mean Average Precision for bm25 without survey data added')
    query_val = pd.read_csv('../data/survey/vaildation_sample_query.csv', header = 0 , index_col = 0)

    # scores without survey
    tf = pd.read_csv('../data/course_info_scores/course_info_tf.csv', header=0, index_col=0)
    tf_norm = pd.read_csv('../data/course_info_scores/course_info_tf_norm.csv', header=0, index_col=0)
    idf = pd.read_csv('../data/course_info_scores/course_info_idf.csv', header=0, index_col=0)
    map = get_map(query_val, tf=tf, tf_norm=tf_norm, idf=idf)
    print("Mean Average Precision on validation query (bm25 with no survey): ", map)

    print("#"*200)
    print('Calculating Mean Average Precision for bm25 with survey data added')
    tf_with_survey = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_tf.csv', header=0, index_col=0)
    tf_norm_with_survey = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_tf_norm.csv', header=0, index_col=0)
    idf_with_survey = pd.read_csv('../data/course_info_with_survey_scores/course_info_with_survey_idf.csv', header=0, index_col=0)
    map_with_survey = get_map(query_val, tf=tf_with_survey, tf_norm=tf_norm_with_survey, idf=idf_with_survey)
    print("Mean Average Precision on validation query (bm25 after relevance feedback training): ", map_with_survey)


    print("#"*200)
    print('Calculating Mean Average Precision for bm25 after training with relevance feedback')
    tf_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_trained.csv', header=0, index_col=0)
    tf_norm_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_tf_norm_trained.csv', header=0, index_col=0)
    idf_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_idf_trained.csv', header=0, index_col=0)
    map_relevance_feedback = get_map(query_val, tf=tf_relevance_feedback, tf_norm=tf_norm_relevance_feedback, idf=idf_relevance_feedback)
    print("Mean Average Precision on validation query (bm25 after relevance feedback training): ", map_relevance_feedback)

    df_relevance_feedback = pd.read_csv('../data/trained_scores/course_info_with_survey_df_trained.csv', header=0, index_col=0)
    norm_association_matrix = pd.read_csv('../data/trained_scores/norm_association_matrix_trained.csv', header = 0, index_col = 0)
    print("#"*200)
    print('Calculating Mean Average Precision for bm25 with pseudo relevance feedback')
    map_pseudo_relevance_feedback = get_map_pseudo_relevance(query_val, df=df_relevance_feedback, tf=tf_relevance_feedback, tf_norm=tf_norm_relevance_feedback, idf=idf_relevance_feedback, norm_association_matrix=norm_association_matrix)
    print("Mean Average Precision on validation query (bm25 with pseudo relevance feedback): ", map_pseudo_relevance_feedback)


if __name__ == '__main__':
    get_map_scores()