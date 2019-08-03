from sklearn.naive_bayes import MultinomialNB
from paths import Path
from argparse import ArgumentParser
import numpy as np
import time
import pickle

class main:

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

	def updateFalseFalse(self, verdict):
		if verdict == "pants-fire":
			self.pants_fire[1][1] += 1
		elif verdict == "true":
			self.true[1][1] += 1
		elif verdict == "mostly-true":
			self.mostly_true[1][1] += 1
		elif verdict == "false":
			self.false[1][1] += 1
		elif verdict == "barely-true":
			self.barely_true[1][1] += 1
		elif verdict == "half-true":
			self.half_true[1][1] += 1

	def updateFalseTrue(self, verdict):
		if verdict == "pants-fire":
			self.pants_fire[1][0] += 1
		elif verdict == "true":
			self.true[1][0] += 1
		elif verdict == "mostly-true":
			self.mostly_true[1][0] += 1
		elif verdict == "false":
			self.false[1][0] += 1
		elif verdict == "barely-true":
			self.barely_true[1][0] += 1
		elif verdict == "half-true":
			self.half_true[1][0] += 1

	def updateTrueTrue(self, verdict):
		if verdict == "pants-fire":
			self.pants_fire[0][0] += 1
		elif verdict == "true":
			self.true[0][0] += 1
		elif verdict == "mostly-true":
			self.mostly_true[0][0] += 1
		elif verdict == "false":
			self.false[0][0] += 1
		elif verdict == "barely-true":
			self.barely_true[0][0] += 1
		elif verdict == "half-true":
			self.half_true[0][0] += 1

	def updateTrueFalse(self, verdict):
		if verdict == "pants-fire":
			self.pants_fire[0][1] += 1
		elif verdict == "true":
			self.true[0][1] += 1
		elif verdict == "mostly-true":
			self.mostly_true[0][1] += 1
		elif verdict == "false":
			self.false[0][1] += 1
		elif verdict == "barely-true":
			self.barely_true[0][1] += 1
		elif verdict == "half-true":
			self.half_true[0][1] += 1

	def final_verdicts_to_indices(self):
		self.finalVerdictsIndices = []
		for i in range(len(self.finalVerdicts)):
			ind = self.classToIndex.get(self.finalVerdicts[i])
			if ind == None:
				print("INDEX NOT FOUND!!")
				exit(1)
			self.finalVerdictsIndices.append(ind)
		self.finalVerdictsIndices = np.array(self.finalVerdictsIndices)

	def save_data(self):
		save_file_path = self.paths.models
		filename = "/trained_data.pickle"
		with open(save_file_path + filename, 'wb') as file:
			pickle.dump(self.model, file)

	def get_model_data(self):
		save_file_path = self.modelfile
		with open(save_file_path, 'rb') as file:
			model = pickle.load(file)
		self.model = model

	def process_data(self):
		self.processed_data = np.zeros((len(self.classes), len(self.data)))
		for i in range(len(self.processed_data)):
			for j in range(len(self.data)):
				self.processed_data[i][j] = self.data[j][i]

		# for cl in self.classes:
		# 	classIndex = self.classToIndex.get(cl)

		# 	for i in range(len(self.data)):
		# 		finalVerdict = self.finalVerdicts[i]
			
		# 		if self.data[i][classIndex] == 0:
		# 			if finalVerdict == cl:
		# 				self.updateFalseFalse(cl)
		# 			else:
		# 				self.updateFalseTrue(cl)
		# 		else:
		# 			if finalVerdict == cl:
		# 				self.updateTrueTrue(cl)
		# 			else:
		# 				self.updateTrueFalse(cl)

	def naive_bayes_train(self):
		self.model = MultinomialNB()
		X = self.processed_data
		y = [1, 2, 3, 4, 5, 6]
		y = np.array(y)
		self.model.fit(X, y)

	def __init__(self):
		self.paths = Path()

		parser = ArgumentParser()
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

		if self.train == self.test:
			parser.error("--train and --test cannot be true or false together.")

		self.classToIndex = {
			"pants-fire": 0,
			"barely-true": 1,
			"false": 2,
			"half-true": 3,
			"mostly-true": 4,
			"true": 5
		}
		self.indexToClass = {v: k for k, v in self.classToIndex.items()}

		if self.train:
			self.data = []
			self.finalVerdicts = []
			self.classes = ["pants-fire", "barely-true", "false", "half-true", "mostly-true", "true"]

			self.read_data(self.trainfile)
			self.final_verdicts_to_indices()
			self.process_data()
			self.naive_bayes_train()
			self.save_data()

		elif self.test:
			self.get_model_data()
			pass

if __name__ == "__main__":
	main()