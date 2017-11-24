import numpy as np

class ANN(object):
	
	def __init__ (self, num_features, num_labels, hidden_units):
		self.inputLayerSize = num_features
		self.outputLayerSize = num_labels
		self.hiddenLayerSize = hidden_units

		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
		self.b1 = np.random.randn(hidden_units)

	def forwardProp(self, X):
		'''Propagate inputs though network'''

		self.z2 = np.dot(X, self.W1) # calculate 1st layer weighted sum
		self.z2 = self.z2 + self.b1 # add bias
		self.a2 = self.sigmoid(self.z2) # apply sigmoid
		self.z3 = np.dot(self.a2, self.W2) # calculate 2nd layer weighted sum
		yHat = self.sigmoid(self.z3) # apply sigmoid
		return yHat

	def sigmoid (self,z):
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self,z):
		'''Gradient of sigmoid'''
		return np.exp(-z)/((1+np.exp(-z))**2)

	def costFunction(self, X, y):
		'''Compute cost for given X,y, use weights already stored in class'''
		self.yHat = self.forwardProp(X)
		J = 0.5*np.sum((y-self.yHat)**2, axis=1)
		#print "\nJ: \n", J
		#print "J shape: ", J.shape
		return J

	def costFunctionPrime(self, X, y):
	    '''Compute derivative with respect to W and W2 for a given X and y:'''
	    # NOT WORKING
	    self.yHat = self.forwardProp(X)
	    
	    delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
	    dJdW2 = np.dot(self.a2.T, delta3)
	    
	    delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
	    dJdW1 = np.dot(X.T, delta2)  
	    
	    return dJdW1, dJdW2
		