#!/usr/bin/python
#--- misc operations ---
import os
import sys
import math
from collections import defaultdict


#--- nltk ---
import nltk

# Class: CorpusManager
# -----------------------
# takes care of all things tf.idf related, including classification
class CorpusManager:

	#--- texts ---
	documents = []

	#--- word counts ---
	word_counts = defaultdict (lambda: 0)				#global number of word counts
	word_counts_by_document = {}						#word counts per document



	########################################################################################################################
	###############################[--- CONSTRUCTOR/INITIALIZATION ---]#####################################################
	########################################################################################################################


	# Function: fill_word_counts
	# --------------------------
	# will iterate through all of the texts and fill in word_counts and word_counts_by_document
	# appropriately
	def fill_word_counts (self):

		### Step 1: set up the defaultdicts ###
		for document in self.documents:
			self.word_counts_by_document[document['name']] = defaultdict(lambda: 0)

		### Step 2: get word counts ###
		for document in self.documents:
			for word in document['content']:
				self.word_counts[word] += 1
				self.word_counts_by_document[document['name']][word] += 1



	# Function: constructor
	# ---------------------
	# takes in a list of 'texts,' which are dicts with two fields:
	# - 'name' -> name of the text
	# - 'content' -> tokenized list of words occuring in the document
	def __init__ (self, documents_list):
		self.documents = documents_list
		self.fill_word_counts ()



	########################################################################################################################
	###############################[--- TF.IDF RELATED FUNCTIONS ---]#######################################################
	########################################################################################################################

	# Function: tf
	# ------------
	# returns the number of times query_word appears in doc_name
	def tf (self, doc_name, query_word):
		tf = self.word_counts_by_document[doc_name][query_word]
		if tf > 0.0:
			return 1.0 + math.log10 (float(tf))
		else:
			return 0.0

	# Function: df
	# ------------
	# returns the number of documents that contain query_word
	def df (self, query_word):
		containing_documents = [doc for doc in self.documents if self.word_counts_by_document[doc['name']][query_word] > 0]
		return len(containing_documents)

	# Function: idf
	# -------------
	# returns log_10( (num_of_documents) / (query_word)
	def idf (self, query_word):

		return math.log10( float(len(self.documents)) / float(self.df(query_word)) )

    # Function: tf_idf
    # ----------------
    # given the name of a text and a query word, this will return the tf.idf for it.
	def tf_idf (self, doc_name, query_word):
	
		return self.tf(doc_name, query_word) * self.idf (query_word)








if __name__ == '__main__':

	doc1 = {'name':'doc1', 'content':['hello', 'world', 'this']}
	doc2 = {'name':'doc2', 'content':['hello', 'shanghai']}
	doc3 = {'name':'doc3', 'content':['hello', 'world', 'thiss']}

	corpus_manger = CorpusManager ([doc1, doc2, doc3])
	print corpus_manger.tf_idf ('doc1', 'world')
	print corpus_manger.tf_idf ('doc2', 'shanghai')


	# #make a 'Text'
	# filename_futurama = 'data/all_comments/futurama.txt'
	# filename_anime = 'data/all_comments/anime.txt'
	# filename_comics = 'data/all_comments/comics.txt'

	# futurama = open (filename_futurama, 'r')
	# anime = open (filename_anime, 'r')
	# comics = open (filename_comics, 'r')

	# futurama_text = Text(futurama.read().lower())
	# anime_text = Text (anime.read().lower())
	# comics_text = Text(comics.read().lower())


	# test_text_1 = Text(['hello', ',', 'world', '!', 'raptor'])
	# test_text_2 = Text(['hello', ',', 'world', 'raptor', 'raptor'])

	# all_texts = [futurama_text, anime_text, comics_text]
	# text_reader = TextCollection ([test_text_1, test_text_2])

	# #works, ostensibly, up to here
	# print text_reader.tf_idf ('raptor', test_text_1)
	# print text_reader.tf_idf ('raptor', test_text_2)
