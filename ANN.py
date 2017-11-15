
class ANN (self):

	def __init__ (self):
		self.inputLayerSize = len(testing_features)
		self.outputLayerSize = 20 
		self.hiddenLayerSize = 100

		#Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

	
	def forwardProp:


def softmax(output_array):
    logits_exp = np.exp(output_array)
    return logits_exp / np.sum(logits_exp, axis = 1, keepdims = True)

def sigmoid (z):
	return 1/(1+np.exp(-z)
