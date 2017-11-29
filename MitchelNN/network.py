##network.py

import numpy as np
import pandas as pd

class ANN(object):
	
	def __init__ (self, num_features, num_labels, hidden_units):
		''' Define layer sizes, init weights '''
		self.inputLayerSize  = num_features
		self.outputLayerSize = num_labels
		self.hiddenLayerSize = hidden_units

		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
		self.b1 = np.random.randn(self.hiddenLayerSize)
		self.b2 = np.random.randn(self.outputLayerSize)
	
	def forward(self, inputs):
		'''Propagate inputs though network'''
		#print ("\nDoing forward prop...")
		self.z1 = np.dot(inputs, self.W1)
		self.a1 = self.sigmoid(self.z1)
		print ("\nFirst layer activations: ", self.a1)
		self.z2 = np.dot(self.a1, self.W2)
		output = self.sigmoid(self.z2) 
		return output
	
	def sigmoid (self,z):
		#signal = np.clip( z, -500, 500 )
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self,z):
		#Gradient of sigmoid
		return np.exp(-z)/((1+np.exp(-z))**2)

	def cost(self, output, true):
		#Compute cost for given X,y, use weights already stored in class.
		#print ("\nCalculating cost...")
		C = output-true
		C = C ** 2
		
		C_hat = []

		for row in C:
			C_hat.append(sum(row))

		return C_hat

	def costFunctionPrime(self, output, true, training_features):
		#Compute derivative with respect to W and W2 for a given X and y:

		delta2 = np.multiply(-(output-true), self.sigmoidPrime(self.z2))
		dJdW2 = np.dot(self.a1.T, delta2)

		delta1 = np.dot(delta2, self.W2.T)*self.sigmoidPrime(self.z1)
		dJdW1 = np.dot(training_features.T, delta1) 

		return dJdW1, dJdW2

	def w_update(self, dw1, dw2, lr):
		#print ("Updating weights...")
		self.W1 = self.W1 + (lr * dw1)
		self.W2 = self.W2 + (lr * dw2)










	

