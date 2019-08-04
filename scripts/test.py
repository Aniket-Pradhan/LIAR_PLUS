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
		filename = "/tf.log"
		with open(save_file_path + filename, 'wb') as file:
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
		filename = "/idf.log"
		with open(save_file_path + filename, 'wb') as file:
			pickle.dump(self.idf, file)
	
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

	def get_docs(self, filename):
		docs = []
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
				docs.append(doc)
				terms = doc.strip().split(" ")
				self.terms.extend(terms)
			self.docs.extend(docs)
		self.num_docs = len(docs)
		self.terms = list(set(self.terms))

	def __init__(self):
		self.paths = Path()

		parser = ArgumentParser()
		parser.add_argument("-f", "--file", required=True, help="Specify the location of the dataset to analyze", metavar="FILE")
		args = parser.parse_args()

		self.file = args.file
		self.docs = []
		self.terms = []
		self.get_docs(self.file)
		self.term_freq = []
		self.total_terms_in_docs = []
		self.idf = []

		self.get_tf()
		self.save_tf()
		self.get_idf()

if __name__ == "__main__":
	tfidf()