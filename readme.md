# 50.045 Information Retrieval Project 
## SUTD Elective Course Recommendation System :books:
This project entails utilizing information retrieval techniques to build a course elective recommendation system catered for SUTD students. The idea is to find out more about the students by understanding their interests and skills and recommending the electives to them based on the information given. As a proof of concept, the system is currently catered towards ESD and ISTD pillars and will only be able to recommend elective courses only. (Term 6 - Term 8). 

### Usage
To use our recommendation system:
```
update when we have completed our UI
```
### Data Collection
Our recommedation system is built in accordance to two types of data. 
1. Web scraping course information from [ISTD courses](https://istd.sutd.edu.sg/education/undergraduate/course-catalogue/) and [ESD courses](https://esd.sutd.edu.sg/academics/undergraduate-programme/courses/)
2. Surveying our peers (seniors) to know more about themselves. This included their interests, the courses they have taken and their skills learnt from those courses.

### Models Used
We experimented with several models to construct our recommendation system:
1. Cosine Similarity  (without query expansion, only with course information data)
2. Cosine Similarity  (with query expansion, only with course information data)
3. Cosine Similarity  (with query expansion, with course information data and 50% of survey data)
4. BM25 Basic  (without query expansion, only with course information data)
5. BM25 Basic  (with query expansion, only with course information data)
6. BM25 Basic  (with query expansion, with course information data and 50% of survey data)
7. BM25 with Reformulation  (with query expansion, with course information data and 50% of survey data. Reformulation training with 25% of survey data)
8. Bm25 with Reformulation and Pseudo Relevance Feedback (with query expansion, with course information data and 50% of survey data. Reformulation training with 25% of survey data)

### Folder Struture
```
- scripts                                       
    - logs
        - bm25_with_relevance_feedback_training.log   # bm25 with relevance feedback training logs
        - MAP_log.txt                                 # MAP scores for the methods experimented 
        - retrieval_speed.txt                         # query speeds for the methods 
    - pre_processing
        - survey
            - query_sample_types.py                   #
            - survey_analysis.py                      #
            - survey_results_converted.py             #
        - web_scrap
            - esd_scrap.py                            # scrap esd course information from esd course website
            - istd_scrap.py                           # scrap istd course information from istd course website
            - merge_course_info.py                    # merge esd and istd course information together 
        - tf_idf.py                                   # convert merged course information into tf,df,idf scores
        - tf_idf_with_survey.py                       # merge course info with survey data, then convert into tf,df,idf scores
    - utils
        - association_matrix.py                       # generate association matrix with tf scores
        - query_processing.py                         # process input query before feeding into model
    - basic_bm25.py                                   # bm25 basic formula
    - bm25_with_relevance_feedback.py                 # bm25 reformulated formula and training procedure
    - bm25_with_pseudo_relevance.py                   # bm25 reformulated with pseudo-relevance formula
    - cos_sim.py                                      # cosine-similarity formula
    - MAP.py                                          # MAP scores for all methods
    - NDCG.py                                         # NDCG scores for all methods
    - retrieval_speed.py                              # query speeds for all methods
- data
    - course_info_scores                              # tf,df,idf,association matrix scores on course information
    - course_info_with_survey                         # tf,df,idf,association matrix scores on course information with survey data
    - survey
        -
        -                                            
    - trained_scores                                  # tf,df,idf,association matrix scores after training (bm25)
    - web_scrap                                       # course information from web scrap
- pretrained_corpus
    - pretrained_corpus_readme.md                     # pretrained_corpus files for query expansion                                                                  
```
