
# coding: utf-8

# # Final project: StackOverflow assistant bot
# 
# Congratulations on coming this far and solving the programming assignments! In this final project, we will combine everything we have learned about Natural Language Processing to construct a *dialogue chat bot*, which will be able to:
# * answer programming-related questions (using StackOverflow dataset);
# * chit-chat and simulate dialogue on all non programming-related questions.
# 
# For a chit-chat mode we will use a pre-trained neural network engine available from [ChatterBot](https://github.com/gunthercox/ChatterBot).
# Those who aim at honor certificates for our course or are just curious, will train their own models for chit-chat.
# ![](https://imgs.xkcd.com/comics/twitter_bot.png)
# ©[xkcd](https://xkcd.com)

# ### Data description
# 
# To detect *intent* of users questions we will need two text collections:
# - `tagged_posts.tsv` — StackOverflow posts, tagged with one programming language (*positive samples*).
# - `dialogues.tsv` — dialogue phrases from movie subtitles (*negative samples*).
# 

# In[2]:


import sys
sys.path.append("..")
from common.download_utils import download_project_resources

download_project_resources()


# For those questions, that have programming-related intent, we will proceed as follow predict programming language (only one tag per question allowed here) and rank candidates within the tag using embeddings.
# For the ranking part, you will need:
# - `word_embeddings.tsv` — word embeddings, that you  trained with StarSpace in the 3rd assignment. It's not a problem if you didn't do it, because we can offer an alternative solution for you.

# As a result of this notebook, you should obtain the following new objects that you will then use in the running bot:
# 
# - `intent_recognizer.pkl` — intent recognition model;
# - `tag_classifier.pkl` — programming language classification model;
# - `tfidf_vectorizer.pkl` — vectorizer used during training;
# - `thread_embeddings_by_tags` — folder with thread embeddings, arranged by tags.
#     

# Some functions will be reused by this notebook and the scripts, so we put them into *utils.py* file. Don't forget to open it and fill in the gaps!

# In[3]:


from utils import *


# ## Part I. Intent and language recognition

# We want to write a bot, which will not only **answer programming-related questions**, but also will be able to **maintain a dialogue**. We would also like to detect the *intent* of the user from the question (we could have had a 'Question answering mode' check-box in the bot, but it wouldn't fun at all, would it?). So the first thing we need to do is to **distinguish programming-related questions from general ones**.
# 
# It would also be good to predict which programming language a particular question referees to. By doing so, we will speed up question search by a factor of the number of languages (10 here), and exercise our *text classification* skill a bit. :)

# In[4]:


import numpy as np
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer


# ### Data preparation

# In the first assignment (Predict tags on StackOverflow with linear models), you have already learnt how to preprocess texts and do TF-IDF tranformations. Reuse your code here. In addition, you will also need to [dump](https://docs.python.org/3/library/pickle.html#pickle.dump) the TF-IDF vectorizer with pickle to use it later in the running bot.

# In[4]:


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


# Now, load examples of two classes. Use a subsample of stackoverflow data to balance the classes. You will need the full data later.

# In[5]:


sample_size = 200000

dialogue_df = pd.read_csv('data/dialogues.tsv', sep='\t').sample(sample_size, random_state=0)
stackoverflow_df = pd.read_csv('data/tagged_posts.tsv', sep='\t').sample(sample_size, random_state=0)


# Check how the data look like:

# In[6]:


dialogue_df.head()


# In[7]:


stackoverflow_df.head()


# Apply *text_prepare* function to preprocess the data:

# In[5]:


from utils import text_prepare


# In[10]:


dialogue_df['text'] = [text_prepare(t) for t in dialogue_df['text']] ######### YOUR CODE HERE #############
stackoverflow_df['title'] = [text_prepare(t) for t in stackoverflow_df['title']]######### YOUR CODE HERE #############


# ### Intent recognition

# We will do a binary classification on TF-IDF representations of texts. Labels will be either `dialogue` for general questions or `stackoverflow` for programming-related questions. First, prepare the data for this task:
# - concatenate `dialogue` and `stackoverflow` examples into one sample
# - split it into train and test in proportion 9:1, use *random_state=0* for reproducibility
# - transform it into TF-IDF features

# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X = np.concatenate([dialogue_df['text'].values, stackoverflow_df['title'].values])
y = ['dialogue'] * dialogue_df.shape[0] + ['stackoverflow'] * stackoverflow_df.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)######### YOUR CODE HERE ##########
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

X_train_tfidf, X_test_tfidf = tfidf_features(X_train,X_test,'tfidf_vectorizer.pkl') ######### YOUR CODE HERE ###########


# Train the **intent recognizer** using LogisticRegression on the train set with the following parameters: *penalty='l2'*, *C=10*, *random_state=0*. Print out the accuracy on the test set to check whether everything looks good.

# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[20]:



intent_recognizer=LogisticRegression(penalty='l2',C=10,random_state=0)
intent_recognizer.fit(X_train_tfidf,y_train)


# In[21]:


# Check test accuracy.
y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Tes accuracy = {}'.format(test_accuracy))


# Dump the classifier to use it in the running bot.

# In[22]:


pickle.dump(intent_recognizer, open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))


# ### Programming language classification 

# We will train one more classifier for the programming-related questions. It will predict exactly one tag (=programming language) and will be also based on Logistic Regression with TF-IDF features. 
# 
# First, let us prepare the data for this task.

# In[23]:


X = stackoverflow_df['title'].values
y = stackoverflow_df['tag'].values


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))


# Let us reuse the TF-IDF vectorizer that we have already created above. It should not make a huge difference which data was used to train it.

# In[25]:


vectorizer = pickle.load(open(RESOURCE_PATH['TFIDF_VECTORIZER'], 'rb'))

X_train_tfidf, X_test_tfidf = vectorizer.transform(X_train), vectorizer.transform(X_test)


# Train the **tag classifier** using OneVsRestClassifier wrapper over LogisticRegression. Use the following parameters: *penalty='l2'*, *C=5*, *random_state=0*.

# In[26]:


from sklearn.multiclass import OneVsRestClassifier


# In[28]:


tag_classifier=OneVsRestClassifier(LogisticRegression(penalty='l2',C=5,random_state=0))
tag_classifier.fit(X_train_tfidf,y_train)


# In[29]:


# Check test accuracy.
y_test_pred = tag_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))


# Dump the classifier to use it in the running bot.

# In[30]:


pickle.dump(tag_classifier, open(RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))


# ## Part II. Ranking  questions with embeddings

# To find a relevant answer (a thread from StackOverflow) on a question you will use vector representations to calculate similarity between the question and existing threads. We already had `question_to_vec` function from the assignment 3, which can create such a representation based on word vectors. 
# 
# However, it would be costly to compute such a representation for all possible answers in *online mode* of the bot (e.g. when bot is running and answering questions from many users). This is the reason why you will create a *database* with pre-computed representations. These representations will be arranged by non-overlaping tags (programming languages), so that the search of the answer can be performed only within one tag each time. This will make our bot even more efficient and allow not to store all the database in RAM. 

# Load StarSpace embeddings which were trained on Stack Overflow posts. These embeddings were trained in *supervised mode* for duplicates detection on the same corpus that is used in search. We can account on that these representations will allow us to find closely related answers for a question. 
# 
# If for some reasons you didn't train StarSpace embeddings in the assignment 3, you can use [pre-trained word vectors](https://code.google.com/archive/p/word2vec/) from Google. All instructions about how to work with these vectors were provided in the same assignment. However, we highly recommend to use StartSpace's embeddings, because it contains more appropriate embeddings. If you chose to use Google's embeddings, delete the words, which is not in Stackoverflow data.

# In[6]:


starspace_embeddings, embeddings_dim = load_embeddings('data/starspace_emb.tsv')


# Since we want to precompute representations for all possible answers, we need to load the whole posts dataset, unlike we did for the intent classifier:

# In[7]:


posts_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')


# Look at the distribution of posts for programming languages (tags) and find the most common ones. 
# You might want to use pandas [groupby](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html) and [count](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.count.html) methods:

# In[8]:


counts_by_tag = posts_df.groupby(['tag'], as_index=False).agg({'title':['count']})
counts_by_tag.columns=['tag','count']
counts_by_tag.set_index('tag',inplace=True)


# Now for each `tag` you need to create two data structures, which will serve as online search index:
# * `tag_post_ids` — a list of post_ids with shape `(counts_by_tag[tag],)`. It will be needed to show the title and link to the thread;
# * `tag_vectors` — a matrix with shape `(counts_by_tag[tag], embeddings_dim)` where embeddings for each answer are stored.
# 
# Implement the code which will calculate the mentioned structures and dump it to files. It should take several minutes to compute it.

# In[10]:


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

