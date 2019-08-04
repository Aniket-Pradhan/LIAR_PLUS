# Class imports
from paths import Path

# Library imports
import pickle
import numpy as np
from argparse import ArgumentParser
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import time

class naive_bayes_model:
	def read_data(self, filename):
		with open(filename, 'r') as file:
			for line in file:
				line = line.strip().split("\t")
				if line[0].isdigit():
					line.pop(0)

				tempList = []
				for i in range(8, 13):
					tempList.append(int(float(line[i])))

				tempList.append(min(tempList[3:5]))
				self.data.append(tempList)
				self.finalVerdicts.append(line[1])
		if self.binary:
			self.make_verdicts_binary()
			for i in range(len(self.data)):
				data_arr = self.data[i]
				new_data_arr = [0] * 2
				new_data_arr[0] = sum(data_arr[:3])
				new_data_arr[1] = sum(data_arr[3:])
				self.data[i] = new_data_arr

	def final_verdicts_to_indices(self):
		self.finalVerdictsIndices = []
		for i in range(len(self.finalVerdicts)):
			ind = self.classToIndex.get(self.finalVerdicts[i])
			if ind == None:
				print("INDEX NOT FOUND!!")
				exit(1)
			self.finalVerdictsIndices.append(ind)
		self.finalVerdictsIndices = np.array(self.finalVerdictsIndices)

	def process_data(self):
		self.processed_data = np.zeros((len(self.classes), len(self.data)))
		for i in range(len(self.processed_data)):
			for j in range(len(self.data)):
				self.processed_data[i][j] = self.data[j][i]

	def naive_bayes_train(self):
		self.model = self.model = LogisticRegression(random_state=0, solver='lbfgs')

		X = np.array(self.data)
		y = self.finalVerdictsIndices

		self.model.fit(X, y)

	def save_data(self):
		save_file_path = self.paths.models
		filename = "/trained_data_logistic.pickle"
		with open(save_file_path + filename, 'wb') as file:
			pickle.dump(self.model, file)

	def load_data(self):
		save_file_path = self.modelfile
		with open(save_file_path, 'rb') as file:
			model = pickle.load(file)
		self.model = model

	def make_verdicts_binary(self):
		false_items = ["pants-fire", "barely-true", "false"]
		true_items = ["half-true", "mostly-true", "true"]
		for i in range(len(self.finalVerdicts)):
			if self.finalVerdicts[i] in false_items:
				self.finalVerdicts[i] = "false"
			else:
				self.finalVerdicts[i] = "true"

	def train_data(self):
		self.data = []
		self.finalVerdicts = []
		self.read_data(self.trainfile)
		self.final_verdicts_to_indices()
		self.process_data()
		self.naive_bayes_train()
		self.save_data()

	def test_data(self):
		self.data = []
		self.finalVerdicts = []
		self.load_data()
		self.read_data(self.testfile)
		self.process_data()
		self.final_verdicts_to_indices()

		test_data = []
		for i in self.data:
			temp_data = np.array(i)
			temp_data.reshape(1, -1)
			test_data.append(temp_data)
		test_data = np.array(test_data)
		
		y_pred = self.model.predict(test_data)
		print("Accuracy:",metrics.accuracy_score(self.finalVerdictsIndices, y_pred))

	def __init__(self):
		self.paths = Path()

		parser = ArgumentParser()
		parser.add_argument("--binary", default=False, help="If you want to do a binary classification, or the default six-way classification.", metavar="BOOLEAN")
		parser.add_argument("--train", default=False, help="Want to train the model? Set as True, if you want to...", metavar="BOOLEAN")
		parser.add_argument("--test", default=False, help="Want to test the model? Set as True, if you want to...", metavar="BOOLEAN")
		parser.add_argument("--trainfile", help="Specify the location of the training dataset", metavar="FILE")
		parser.add_argument("--testfile", help="Specify the location of the testing dataset", metavar="FILE")
		parser.add_argument("--model", help="Specify the location of the trained model", metavar="FILE")
		args = parser.parse_args()

		if (args.train and args.trainfile is None) or ((not args.train) and args.trainfile is not None):
			parser.error("--train requires --trainfile.")
		if (args.test and (args.testfile is None or args.model is None)) or ((not args.test) and (args.testfile is not None or args.model is not None)):
			parser.error("--test requires --testfile and --model.")

		self.train = True if args.train == "true" else False 
		self.test = True if args.test == "true" else False 
		self.trainfile = args.trainfile
		self.testfile = args.testfile
		self.modelfile = args.model
		self.binary = True if args.binary == "true" else False
		
		if self.train == self.test:
			parser.error("--train and --test cannot be true or false together.")

		if self.binary:
			self.classes = ["false", "true"]
		else:
			self.classes = ["pants-fire", "barely-true", "false", "half-true", "mostly-true", "true"]
		if self.binary:
			self.classToIndex = {
				"true": 0,
				"false": 1
			}
		else:
			self.classToIndex = {
				"pants-fire": 0,
				"barely-true": 1,
				"false": 2,
				"half-true": 3,
				"mostly-true": 4,
				"true": 5
			}

		if self.train:
			print("TRAINING...")
			self.train_data()

		elif self.test:
			print("TESTING...")
			self.test_data()
			
if __name__ == "__main__":
	naive_bayes_model()