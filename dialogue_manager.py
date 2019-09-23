
import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.spatial.distance import cosine
from chatterbot import ChatBot
from utils import *


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        print('TAG NAME IS')
        print(tag_name)
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):

        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        
        
        question_vec = question_to_vec(question,self.word_embeddings,self.embeddings_dim).reshape(1,-1)


        min_distance=999999.0
        argmin=-1
        for i in range(len(thread_embeddings)):
                distance = cosine(question_vec,thread_embeddings[i].reshape(1,-1))
                if distance<=min_distance:
                        min_distance=distance
                        argmin=i
        print(argmin)       
        return thread_ids.values[argmin]
def create_chitchat_bot():
        """Initializes self.chitchat_bot with some conversational model."""

        chatbot=ChatBot('Norman',trainer="chatterbot.trainers.ChatterBotCorpusTrainer")
        chatbot.train("chatterbot.corpus.english")
        return chatbot

class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = None

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
#        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.tag_classifier = None
        self.thread_ranker = ThreadRanker(paths)
        self.chitchatbot=None
        self.paths=paths
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        self.tfidf_vectorizer = unpickle_file(self.paths['TFIDF_VECTORIZER']) 
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        self.tfidf_vectorizer = None
        intent = self.intent_recognizer.predict(features)

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            if self.chitchatbot==None:
                self.chitchatbot=create_chitchat_bot()
            response = self.chitchatbot.get_response(prepared_question)
            return response
        
        # Goal-oriented part:
        else:

            self.chitchatbot=None        
            self.tag_classifier = unpickle_file(self.paths['TAG_CLASSIFIER'])
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]#### YOUR CODE HERE ####
            self.tag_classifier = None 
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question,tag)#### YOUR CODE HERE ####
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)


