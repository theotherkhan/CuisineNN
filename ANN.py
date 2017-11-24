class ANN (object):
	def __init__ (self):
		self.inputLayerSize = 0
		self.outputLayerSize = 0 
		self.hiddenLayerSize = 0

		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
	
	def forwardProp(self, X):
		#Propagate inputs though network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		return z3

	def softmax(self,output_array):
	    logits_exp = np.exp(output_array)
	    return logits_exp / np.sum(logits_exp, axis = 1, keepdims = True)

	def sigmoid (self,z):
		return 1/(1+np.exp(-z))