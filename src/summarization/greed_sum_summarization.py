import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def greed_sum(text, num_sent, min_df=1, max_df=1.0):
        
    #fit a TFIDF vectorizer
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    vectorizer.fit(text)
    
    #get the matrix
    X = vectorizer.transform(text).toarray()
    
    #get the sentence indices
    idx = []
    while sum(sum(X)) != 0:
        ind = np.argmax(X.sum(axis=1))
        idx.append(ind)

        #update the matrix deleting the columns corresponding to the words found in previous step
        cols = X[ind]
        col_idx = [i for i in range(len(cols)) if cols[i] > 0]
        X = np.delete(X, col_idx, 1)
        
           
    if num_sent != 0:
        idx = idx[:num_sent]
    
        
    idx.sort()
    
    summary = ' '.join([text[i] for i in idx])
    
    return summary
    
