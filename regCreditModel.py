import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class LRCreditRiskModel():

	def __init__(self):
		self.data = []    # Loan data 
		self.data_x, self.data_y = [], []
		self.x_train, self.x_test = [], []
		self.y_train, self.y_test = [], []
		
		self.model = None

	def load_data(self, path):
		self.loan_data = pd.read_csv(path)
		

	def set_regression_variables(self, x_variables : list, y_variables : list):
		
		self.data_x = self.loan_data[x_variables]
		self.data_y = self.loan_data[y_variables]
		

	def train_test_split(self, test_size=0.4, random_state=123):
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_x, self.data_y,
																test_size=test_size,
																random_state=random_state)

	
	def train_model(self, solver='lbfgs'):
		self.model = LogisticRegression(solver=solver)
		self.model.fit(self.x_train, np.ravel(self.y_train))



