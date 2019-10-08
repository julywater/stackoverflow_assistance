


import sys
sys.path.append("..")
from utils import *
import numpy as np
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_features(X_train, X_test, vectorizer_path):
    """Performs TF-IDF transformation and dumps the model."""
    tfidf_vectorizer=TfidfVectorizer(use_idf=True,ngram_range=(1,2),min_df=0.00005,max_df=0.9,token_pattern='(\S+)')
    fitted_vectorizer=tfidf_vectorizer.fit(X_train)
    file = open(vectorizer_path, 'wb')
    pickle.dump(fitted_vectorizer, file)
    file.close()
    X_train=fitted_vectorizer.transform(X_train) 
    X_test=fitted_vectorizer.transform(X_test)
    return X_train, X_test
sample_size = 200000
dialogue_df = pd.read_csv('data/dialogues.tsv', sep='\t').sample(sample_size, random_state=0)
stackoverflow_df = pd.read_csv('data/tagged_posts.tsv', sep='\t').sample(sample_size, random_state=0)
dialogue_df.head()
stackoverflow_df.head()
from utils import text_prepare
dialogue_df['text'] = [text_prepare(t) for t in dialogue_df['text']] 
stackoverflow_df['title'] = [text_prepare(t) for t in stackoverflow_df['title']]
from sklearn.model_selection import train_test_split
X = np.concatenate([dialogue_df['text'].values, stackoverflow_df['title'].values])
y = ['dialogue'] * dialogue_df.shape[0] + ['stackoverflow'] * stackoverflow_df.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

X_train_tfidf, X_test_tfidf = tfidf_features(X_train,X_test,'tfidf_vectorizer.pkl') 
# Train the **intent recognizer** using LogisticRegression on the train set 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
intent_recognizer=LogisticRegression(penalty='l2',C=10,random_state=0)
intent_recognizer.fit(X_train_tfidf,y_train)
y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Tes accuracy = {}'.format(test_accuracy))
pickle.dump(intent_recognizer, open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))

X = stackoverflow_df['title'].values
y = stackoverflow_df['tag'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

vectorizer = pickle.load(open(RESOURCE_PATH['TFIDF_VECTORIZER'], 'rb'))

X_train_tfidf, X_test_tfidf = vectorizer.transform(X_train), vectorizer.transform(X_test)


# Train the **tag classifier** using OneVsRestClassifier wrapper over LogisticRegression.


from sklearn.multiclass import OneVsRestClassifier
tag_classifier=OneVsRestClassifier(LogisticRegression(penalty='l2',C=5,random_state=0))
tag_classifier.fit(X_train_tfidf,y_train)



# Check test accuracy.
y_test_pred = tag_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))


# Dump the classifier to use it in the running bot.



pickle.dump(tag_classifier, open(RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))


# ## Part II. Ranking  questions with embeddings 
starspace_embeddings, embeddings_dim = load_embeddings('data/starspace_emb.tsv')
# Since we want to precompute representations for all possible answers, we need to load the whole posts dataset, unlike we did for the intent classifier:

posts_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')
# Look at the distribution of posts for programming languages (tags) and find the most common ones. 
counts_by_tag = posts_df.groupby(['tag'], as_index=False).agg({'title':['count']})
counts_by_tag.columns=['tag','count']
counts_by_tag.set_index('tag',inplace=True)
# * `tag_post_ids` — a list of post_ids with shape `(counts_by_tag[tag],)`. It will be needed to show the title and link to the thread;
# * `tag_vectors` — a matrix with shape `(counts_by_tag[tag], embeddings_dim)` where embeddings for each answer are stored.
# 
# Implement the code which will calculate the mentioned structures and dump it to files. It should take several minutes to compute it.
import os
os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)
for tag, count in counts_by_tag.itertuples():
    tag_posts = posts_df[posts_df['tag'] == tag]
    tag_post_ids = tag_posts.post_id.values
    tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)
    for i, title in enumerate(tag_posts['title']):
        tag_vectors[i, :] =question_to_vec(text_prepare(title), starspace_embeddings,embeddings_dim) ######### YOUR CODE HERE #############
    # Dump post ids and vectors to a file.
    filename = os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag))
    pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))

