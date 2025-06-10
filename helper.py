import pickle
import numpy as np
from xgboost import XGBClassifier

tfidf = pickle.load(open('tfidf.pkl','rb'))

def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return len(w1 & w2)

def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return (len(w1) + len(w2))

def query_point_creator(q1,q2):
    
    input_query = []
    
    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))
    
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    
    input_query.append(test_common_words(q1,q2))
    input_query.append(test_total_words(q1,q2))
    input_query.append(round(test_common_words(q1,q2)/test_total_words(q1,q2),2))

    # tf-idf feature for q1
    q1_tfidf = tfidf.transform([q1]).toarray()
    
    # tf-idf feature for q2
    q2_tfidf = tfidf.transform([q2]).toarray()
    
    return np.hstack((np.array(input_query).reshape(1,7),q1_tfidf,q2_tfidf))