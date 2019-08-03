import os
import sys

class Path:
	def path_exists(self, path):
		if not os.path.isdir(path):
			self.create_path(path)

	def create_path(self, path):
		os.makedirs(path)

	def __init__(self):
		self.base = sys.path[0] + "/.."
		
		self.dataset = self.base + "/dataset"
		self.path_exists(self.dataset)
		
		self.temp_data = self.base + "temp_data"
		
		self.models = self.base + "/models"
		self.path_exists(self.models)
