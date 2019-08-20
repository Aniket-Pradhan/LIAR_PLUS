# Class imports
from paths import Path

# Library imports
import pickle
import progressbar
from math import log
from argparse import ArgumentParser

class tfidf:

	def save_tf(self):
		save_file_path = self.paths.logs
		with open(save_file_path + self.filename_tf, 'wb') as file:
			pickle.dump(self.term_freq, file)
	
	def get_freq(self, doc):
		normalized_doc = doc.lower().strip().split(" ")
		total_terms = len(normalized_doc)
		tf = []
		for i in normalized_doc:
			tf.append(self.get_term_freq(i, normalized_doc, total_terms))
		self.term_freq.append(tf)
		self.total_terms_in_docs.append(total_terms)

	def get_term_freq(self, term, normalized_doc, total_terms):
		num_terms = normalized_doc.count(term)
		return round(num_terms/total_terms, 4)

	def get_tf(self):
		print("GETTING TF...")
		for i in progressbar.progressbar(self.docs):
			self.get_freq(i)
	
	def save_idf(self):
		save_file_path = self.paths.logs
		with open(save_file_path + self.filename_idf, 'wb') as file:
			pickle.dump(self.idf, file)
	
	def save_tfidf(self):
		save_file_path = self.paths.logs
		with open(save_file_path + self.filename_tfidf, 'wb') as file:
			pickle.dump(self.tf_idf, file)

	def optimize_idf(self):
		self.optimized_idf = {}
		for i in range(len(self.idf)):
			self.optimized_idf[self.terms[i]] = self.idf[i]
	
	def load_idf(self):
		save_file_path = self.paths.logs
		with open(save_file_path + self.filename_idf, 'rb') as file:
			self.idf = pickle.load(file)
	
	def load_tf(self):
		save_file_path = self.paths.logs
		with open(save_file_path + self.filename_tf, 'rb') as file:
			self.term_freq = pickle.load(file)
	
	def get_idf(self):
		counter = 0
		log_file_name = "/terms_index.log"
		print("CALCULATING IDF...")
		for i in progressbar.progressbar(range(len(self.terms))):
			term = self.terms[i]
			counter += 1
			num_of_docs = 0
			for doc in self.docs:
				if term.lower() in doc.lower().split(" "):
					num_of_docs += 1
			
			if num_of_docs != 0:
				idf = log(self.num_docs/num_of_docs)
				self.idf.append(idf)
			else:
				self.idf.append(0)
			if counter%100 == 0:
				self.save_idf()
				with open(self.paths.logs + log_file_name, "a") as file:
					line_to_write = str(i) + "\t" + term + "\n"
					file.write(line_to_write)

	def calc_tfidf(self):
		for doc_index in range(len(self.term_freq)):
			doc = self.docs[doc_index]
			normalized_doc = doc.lower().strip().split(" ")
			for term_index in range(len(normalized_doc)):
				term = normalized_doc[term_index]
				tf = self.term_freq[doc_index][term_index]
				idf = self.optimized_idf[term]
				self.tf_idf.append(tf*idf)


	def get_docs(self, filename):
		docs = []
		puncts = ['\'', '\"', '.', ',', '(', ')', '),', ').', ';', ':']
		with open(filename, 'r') as file:
			for line in file:
				line = line.strip().split("\t")
				if line[0].isdigit():
					line.pop(0)
				doc = ""
				try:
					doc = line[14]
				except IndexError:
					doc = line[len(line)-1]
				for i in puncts:
					doc.replace(i, "")
				docs.append(doc)
				terms = doc.lower().strip().split(" ")
				self.terms.extend(terms)
			self.docs.extend(docs)
		self.num_docs = len(docs)
		self.terms = list(set(self.terms))
		# print(len(self.terms))

	def __init__(self):
		self.paths = Path()

		parser = ArgumentParser()
		parser.add_argument("-f", "--file", required=True, help="Specify the location of the dataset to analyze", metavar="FILE")
		parser.add_argument("-r", default=False, help="If you want to re-iterate and generate tf-idf again.", metavar="BOOLEAN")
		args = parser.parse_args()

		self.file = args.file
		self.reiterate = True if args.r == "true" else False

		self.filename_idf = "/idf.log"
		self.filename_tf = "/tf.log"
		self.filename_tfidf = "/tf_idf.log"

		self.docs = []
		self.terms = []
		self.term_freq = []
		self.total_terms_in_docs = []
		self.idf = []
		self.tf_idf = []

		self.get_docs(self.file)

		if self.reiterate:
			self.get_tf()
			self.save_tf()
			self.get_idf()
			self.save_idf()
		else:
			self.load_tf()
			self.load_idf()

		self.optimize_idf()

		self.calc_tfidf()
		self.save_tfidf()

if __name__ == "__main__":
	tfidf()