import numpy as np

class ANN(object):
	
	def __init__ (self, num_features, num_labels, hidden_units):
		''' Define layer sizes, init weights '''
		self.inputLayerSize = num_features
		self.outputLayerSize = num_labels
		self.hiddenLayerSize = hidden_units

		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
		self.b1 = np.random.randn(self.hiddenLayerSize)
		self.b2 = np.random.randn(self.outputLayerSize)

	def forwardProp(self, X):
		'''Propagate inputs though network'''
		print "\nDoing forward prop..."
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3) 
		return yHat

	def sigmoid (self,z):
		#signal = np.clip( z, -500, 500 )
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self,z):
		#Gradient of sigmoid
		signal = np.clip( z, -500, 500 )
		return np.exp(-signal)/((1+np.exp(-signal))**2)
	
	def costFunction(self, yhat, y):
		#Compute cost for given X,y, use weights already stored in class.
		print "\nCalculating cost..."
		J = 0.5*sum((y-yhat)**2)
		return J
	
	def costFunctionPrime(self, yhat, y, X):

		print "\nDoing backprop..."
		#Compute derivative with respect to W and W2 for a given X and y:
		self.yHat = yhat

		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2) 

		#print "\ndJdW1: ", dJdW1 
		#print "\ndJdW2: ", dJdW2 

		return dJdW1, dJdW2
		