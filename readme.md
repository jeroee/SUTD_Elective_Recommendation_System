# 50.045 Information Retrieval Project 
## SUTD Elective Course Recommendation System :books:
This project entails building a course elective recommendation system catered for SUTD students. The idea is to find out more about the students by understanding their interests and skills and recommending the electives to them based on the information given. As a proof of concept, the system is currently catered towards ESD and ISTD pillars and will only be able to recommend elective courses only. (Term 6 - Term 8). 

### Usage
To use our recommendation system:
```
update when we have completed our UI
```
### Data Collection
Our recommedation system is built in accordance to two types of data. 
1. Web scraping course information from [ISTD courses](https://istd.sutd.edu.sg/education/undergraduate/course-catalogue/) and [ESD courses](https://esd.sutd.edu.sg/academics/undergraduate-programme/courses/)
2. Surveying our peers (seniors) to know more about themselves. This included their interests, the courses they have taken and their skills learnt from those courses.

### Methods Used
We experimented with several methods to construct our recommendation system:
1. Cosine Similarity  (without query expansion, only with course information data)
2. Cosine Similarity  (with query expansion, only with course information data)
3. Cosine Similarity  (with query expansion, with course information data and 50% of survey data)
4. BM25 Basic  (without query expansion, only with course information data)
5. BM25 Basic  (with query expansion, only with course information data)
6. BM25 Basic  (with query expansion, with course information data and 50% of survey data)
7. BM25 with Reformulation  (with query expansion, with course information data and 50% of survey data. Reformulation training with 25% of survey data)
8. Bm25 with Reformulation and Pseudo Relevance Feedback (with query expansion, with course information data and 50% of survey data. Reformulation training with 25% of survey data)

### Folder Struture
- scripts
    - archived
    - logs
    - pre_processing
    - utils
    - basic_bm25.py
    - bm25_with_relevance_feedback.py
    - bm25_with_pseudo_relevance.py
    - cos_sim.py
    - MAP.py
    - NDCG.py
    - retrieval_speed.py
- data
    - course_info_scores
    - course_info_with_survey
    - survey
    - trained_scores
    - web_scrap
- pretrained_corpus