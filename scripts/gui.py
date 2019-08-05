from PyQt5 import QtWidgets, QtGui
from mydesign import Ui_MainWindow  # importing our generated file
import sys
import subprocess

class mywindow(QtWidgets.QMainWindow):

	def gaussian_nb(self):
		command = "python naive_bayes.py --train true --trainfile ../dataset/train2.tsv --nb_type GaussianNB"
		command = command.strip().split()
		test = subprocess.Popen(command, stdout=subprocess.PIPE)
		output = test.communicate()[0]
		self.ui.label.setText(output.decode("utf-8").strip())
	
	def bernoulli_nb(self):
		command = "python naive_bayes.py --train true --trainfile ../dataset/train2.tsv --nb_type BernoulliNB"
		command = command.strip().split()
		test = subprocess.Popen(command, stdout=subprocess.PIPE)
		output = test.communicate()[0]
		self.ui.label.setText(output.decode("utf-8").strip())
	
	def mutinomial_nb(self):
		command = "python naive_bayes.py --train true --trainfile ../dataset/train2.tsv --nb_type MultinomialNB"
		command = command.strip().split()
		test = subprocess.Popen(command, stdout=subprocess.PIPE)
		output = test.communicate()[0]
		self.ui.label.setText(output.decode("utf-8").strip())
	
	def nb_test(self):
		command = "python naive_bayes.py --test true --testfile ../dataset/test2.tsv --model ../models/trained_data_nb.pickle"
		command = command.strip().split()
		test = subprocess.Popen(command, stdout=subprocess.PIPE)
		output = test.communicate()[0]
		self.ui.label.setText(output.decode("utf-8").strip())
	
	def logistic(self):
		command = "python logistic.py --train true --trainfile ../dataset/train2.tsv"
		command = command.strip().split()
		test = subprocess.Popen(command, stdout=subprocess.PIPE)
		output = test.communicate()[0]
		self.ui.label.setText(output.decode("utf-8").strip())
	
	def logistic_test(self):
		command = "python logistic.py --test true --testfile ../dataset/test2.tsv --model ../models/trained_data_logistic.pickle"
		command = command.strip().split()
		test = subprocess.Popen(command, stdout=subprocess.PIPE)
		output = test.communicate()[0]
		self.ui.label.setText(output.decode("utf-8").strip())

	def __init__(self):
		super(mywindow, self).__init__()
		self.ui = Ui_MainWindow()	
		self.ui.setupUi(self)
		self.ui.label.setFont(QtGui.QFont('SansSerif', 30))

		self.ui.pushButton.clicked.connect(self.gaussian_nb)
		self.ui.pushButton_2.clicked.connect(self.mutinomial_nb)
		self.ui.pushButton_3.clicked.connect(self.bernoulli_nb)
		self.ui.pushButton_4.clicked.connect(self.nb_test)
		self.ui.pushButton_5.clicked.connect(self.logistic)
		self.ui.pushButton_6.clicked.connect(self.logistic_test)
 
app = QtWidgets.QApplication([])
application = mywindow() 
application.show()
 
sys.exit(app.exec())