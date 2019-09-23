import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
	embeddings_file=open(embeddings_path,"r")
	embeddings={}
	embeddings_dim=100
	for line in embeddings_file.readlines():
		line=line.split()
		if(len(line)==101):
			embeddings[line[0]]=np.array([float(ar) for ar in line[1:]])	
	return(embeddings,embeddings_dim)
   
   

    


def question_to_vec(question, embeddings, dim):
	question=text_prepare(question)
	question_vec=np.zeros(dim)
	words=question.strip().split()
	cnt=0
	for word in words:
		if word in embeddings:
			question_vec+=embeddings[word]
			cnt+=1
	if cnt!=0:        
		question_vec/=cnt
	return question_vec


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

