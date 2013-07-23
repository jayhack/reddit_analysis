#!/usr/bin/python
#--- misc operations ---
import os
import sys
import math
import pickle
import operator
from collections import defaultdict


#--- nltk ---
from nltk import wordpunct_tokenize



# Class: CorpusManager
# -----------------------
# takes care of all things tf.idf related, including classification
class CorpusManager:

	#--- texts ---
	documents = []
	documents_filepath = 'saved_data/documents.obj'


	#--- word counts ---
	word_counts_filepath = 'saved_data/word_counts.obj'
	word_counts_by_document_filepath = 'saved_data/word_counts_by_document.obj'
	
	word_counts 				= defaultdict (lambda: 0)			#global number of word counts
	word_counts_nondefault 		= {}

	word_counts_by_document 			= {}						#word counts per document
	word_counts_by_document_nondefault 	= {}

	min_number_of_occurences = 5 									#the minimumb number of occurences for it to be recorded
	min_number_of_occurences_by_document = 2



	########################################################################################################################
	###############################[--- DATA MANAGEMENT (LOADING, SAVING) ---]##############################################
	########################################################################################################################

	# Function: convert_to_nondefault_dict
	# ---------------------------------
	# converts default_dict to a non-default_dict and returns it
	def convert_to_nondefault_dict (self, default_dict):
		non_default_dict = {}
		for element in default_dict.keys ():
			non_default_dict[element] = default_dict[element]
		return non_default_dict

	# Function: convert_to_default_dict
	# ---------------------------------
	# converts nondefault_dict to a defaultdict and returns it
	def convert_to_default_dict (self, nondefault_dict):
		default_dict = defaultdict(lambda: 0)
		for element in nondefault_dict.keys ():
			default_dict[element] = nondefault_dict[element]
		return default_dict	



	# Function: get_non_defaultdict_versions
	# --------------------------------------
	# will fill in word_counts_nondefault and word_counts_by_document_nondefault with non-defaultdict versions of their
	# respective counterparts
	def get_non_defaultdict_versions (self):
		self.word_counts_nondefault = self.convert_to_nondefault_dict (self.word_counts)
		for doc_name in self.word_counts_by_document.keys ():
			self.word_counts_by_document_nondefault[doc_name] = self.convert_to_nondefault_dict (self.word_counts_by_document[doc_name])

	# Function: get_defaultdict_versions
	# ----------------------------------
	# will convert word_counts_nondefault and word_counts_by_document_nondefault to defaultdicts
	def get_defaultdict_versions (self):
		self.word_counts = self.convert_to_default_dict(self.word_counts_nondefault)
		
		self.word_counts_by_document = {}
		for doc_name in self.word_counts_by_document_nondefault:
			self.word_counts_by_document[doc_name] = self.convert_to_default_dict (self.word_counts_by_document_nondefault[doc_name])



	# Function: save_word_counts
	# --------------------------
	# packs word_counts and word_counts_by_documents into pickled files
	def save_word_counts (self):
		self.get_non_defaultdict_versions ()

		pickle.dump (self.word_counts_nondefault, open(self.word_counts_filepath, 'wb'))
		pickle.dump (self.word_counts_by_document_nondefault, open(self.word_counts_by_document_filepath, 'wb'))

	# Function: load_word_counts
	# --------------------------
	# will unload word_counts and word_counts_by_doc from a pickled file
	def load_word_counts (self):
		self.word_counts_nondefault = pickle.load(open(self.word_counts_filepath, 'rb'))
		self.word_counts_by_document_nondefault = pickle.load (open(self.word_counts_by_document_filepath, 'rb'))
		self.get_defaultdict_versions ()



	# Function: save_documents
	# ------------------------
	# pickles documents into a file
	def save_documents (self):
		
		pickle.dump (self.documents, open(self.documents_filepath, 'wb'))

	# Function: load_documents
	# ------------------------
	# load documents
	def load_documents (self):

		self.documents = pickle.load (open(self.documents_filepath, 'rb'))







	########################################################################################################################
	###############################[--- CONSTRUCTOR/INITIALIZATION, DESTRUCTOR ---]#########################################
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
			print "	- processing " + document['name']
			for word in document['content']:
				self.word_counts[word] += 1
				self.word_counts_by_document[document['name']][word] += 1

		### Step 3: eliminate words that rarely occur ###
		for word in self.word_counts.keys():
			if self.word_counts[word] < self.min_number_of_occurences:
				print "		- NOTICE: removing the following word: " + word
				del self.word_counts[word]

		remove_pairs = []
		for doc_name in self.word_counts_by_document.keys ():
			for word in self.word_counts_by_document[doc_name]:
				if self.word_counts_by_document[doc_name][word] < self.min_number_of_occurences_by_document:
					print "		- NOTICE: removing the following pair: ", (doc_name, word)
					remove_pairs.append ((doc_name, word))
		for pair in remove_pairs:
			del self.word_counts_by_document[pair[0]][pair[1]]




	# Function: constructor
	# ---------------------
	# takes in a list of 'texts,' which are dicts with two fields:
	# - 'name' -> name of the text
	# - 'content' -> tokenized list of words occuring in the document
	# by default, it will find the word counts itself (i.e. it wont load them) and it will save them.
	def __init__ (self, documents_list=None, load=False, save=False):

		self.load = load 
		self.save = save

		### Step 1: get documents ###
		self.documents = documents_list
	
		### Step 2: get word counts ###
		if not load:
			print "	### Getting documents and word counts manually ###"
			self.documents = documents_list
			self.fill_word_counts ()
		else:
			print "	### loading documents and word counts from a pickled file ###"
			self.load_documents ()
			self.load_word_counts ()


	# Function: destructor
	# --------------------
	# will save data where appropriately 
	def __del__ (self):
		if save:
			print "---> Status: saving documents"
			self.save_documents ()
			print "---> Status: saving word counts"
			self.save_word_counts ()






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

    # Function: tfidf
    # ----------------
    # given the name of a text and a query word, this will return the tf.idf for it.
	def tfidf (self, doc_name, query_word):
	
		return self.tf(doc_name, query_word) * self.idf (query_word)

	# Function: compute_tfidf_vectors
	# -------------------------------
	# will get the tfidf_vectors for each document in document_list. 
	# Note: (as of now, only contains top 2000 elements.)
	def compute_tfidf_vectors (self):

		for document in self.documents:

			tfidf_vector = defaultdict (lambda: 0)
			for word in self.word_counts_by_document[document['name']].keys ():
				tfidf_vector[word] = self.tfidf (document['name'], word)

			document['tfidf_vector'] = dict(sorted(tfidf_vector.iteritems(), key=operator.itemgetter(1), reverse=True)[0:2000])



	# Function: normalize_vector
	# --------------------------
	# given a vec mapping keys to numbers, this will return a normalized version
	# (i.e. the length of the vector will be 1)
	def normalize_vector (self, vec):
		length = 0.0
		for element in vec.keys ():
			length += vec[element]*vec[element]
		length = math.sqrt(length)

		normalized_vec = {}
		for element in vec.keys ():
			normalized_vec[element] = vec[element] / length
		return normalized_vec


	# Function: tfidf_cosine_sim
	# --------------------------
	# given two documents, this will calculate the cosine similarity between them
	def tfidf_cosine_sim (self, doc1, doc2):

		doc1_normalized_vec = self.normalize_vector (doc1['tfidf_vector'])
		doc2_normalized_vec = self.normalize_vector (doc2['tfidf_vector'])

		dot_product = 0.0
		common_words = set(doc1_normalized_vec).intersection(set(doc2_normalized_vec))
		for word in common_words:
			dot_product += doc1_normalized_vec[word] * doc2_normalized_vec[word]
		return dot_product







	########################################################################################################################
	###############################[--- INTERFACE/DESCRIPTIONS ---]#########################################################
	########################################################################################################################

	# Function: print_tfidf_profiles
	# ---------------------------
	# for each document, this function will print out the top tf.idf words in it
	def print_tfidf_profiles (self):

		for document in self.documents:
			print "	--- profile for " + document['name'] + '---'

			tfidf_profile = sorted(document['tfidf_vector'].iteritems(), key=operator.itemgetter(1), reverse=True)[0:50]
			for word in tfidf_profile:
				print "		", word
			print "\n"




















# Function: make_document
# -----------------------
# given the filepath to a textfile, this will create and return an appropriate 
# 'document' representation. This consists of a dict with the following fields:
# - name: name of the document (should be unique)
# - content: a list of words that are in the document
def make_document (filepath):
	text_string = open(filepath, 'r').read ()
	content = wordpunct_tokenize (text_string.lower())
	name = filepath.split('/')[-1].split('.')[0]
	return {'name':name, 'content':content}




if __name__ == '__main__':

	save = False

	if len(sys.argv == 1):
		save 

	### Step 1: get a list of all documents as strings ###
	# print "---> Status: loading/creating the document representations"
	# documents = []
	# all_comments_dir = os.path.join (os.getcwd(), 'data_dev/all_comments')
	# for f in os.listdir (all_comments_dir):
		# comment_filepath = os.path.join(all_comments_dir, f)
		# new_document = make_document (comment_filepath)
		# documents.append (new_document)
		# print "	- added document: ", new_document['name']


	### Step 2: create the CorpusManager ###
	print "---> Status: creating corpus manager"
	corpus_manager = CorpusManager (load=True, save=True)


	### Step 3: compute tfidf_vectors ###
	print "---> Status: computing tfidf vectors for each document"
	corpus_manager.compute_tfidf_vectors ()
	corpus_manager.print_tfidf_profiles ()





