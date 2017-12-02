import numpy as np

class ANN(object):
	
	def __init__ (self, num_features, num_labels, hidden_units):
		''' Define layer sizes, initializes weights and biases '''
		self.inputLayerSize = num_features
		self.outputLayerSize = num_labels
		self.hiddenLayerSize = hidden_units

		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
		
		self.b1 = np.full(shape=self.hiddenLayerSize, fill_value = 0)
		self.b2 = np.full(shape=self.outputLayerSize, fill_value = 0)

	def forward(self, X):
		'''Propagate inputs though network'''
		self.z1 = np.dot(X, self.W1) + self.b1
		self.a1 = self.sigmoid(self.z1)
		self.z2 = np.dot(self.a1, self.W2) + self.b2
		output = self.sigmoid(self.z2) 
		return output

	def sigmoid (self,z):
		'''Applies sigmoid function'''
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self,z):
		'''Derivative of the sigmoid'''
		return np.exp(-z)/((1+np.exp(-z))**2)
	
	def cost(self, output, true):
		'''Compute cost for given X,y, use weights already stored in class.'''
		C = (output-true)**2
		C_hat = []
		for row in C:
			C_hat.append(sum(row))
		return C_hat

	def backprop(self, output, true, training_features):
		'''Compute the derivatives of W1 and W2'''

		delta2 = np.multiply(-(output-true), self.sigmoidPrime(self.z2))
		dJdW2 = np.dot(self.a1.T, delta2)

		delta1 = np.dot(delta2, self.W2.T)*self.sigmoidPrime(self.z1)
		dJdW1 = np.dot(training_features.T, delta1)

		dJdB1 = np.sum(delta1, axis=0, keepdims=True)
		dJdB2 = np.sum(delta2, axis=0) 

		return dJdW1, dJdW2, dJdB1, dJdB2

	def w_update(self, dw1, dw2, db1, db2, lr):
		'''Update the weights and biases'''
		self.W1 = self.W1 + (lr * dw1)
		self.W2 = self.W2 + (lr * dw2)

		self.b1 = self.b1 + (lr * db1)
		self.b2 = self.b2 + (lr * db2)



		