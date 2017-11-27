
	def sigmoidPrime(self,z):
		'''Gradient of sigmoid'''
		return np.exp(-z)/((1+np.exp(-z))**2)

	def softmax(self, A):
		expA = np.exp(A)
		return expA / expA.sum(axis=1, keepdims=True)

	def costFunction(self, y, yhat):
		'''Compute cost for given X,y, use weights already stored in class'''
		J = 0.5*np.sum((y-yhat)**2, axis=1)
		return J
	
	def back_pass(self, yhat, y, X):
		# backward pass
		alpha = 10e-5

		delta2 = yhat - y
		delta1 = (delta2).dot(self.W2.T) * self.a2 * (1 - self.a2)

		#print "\nUpdating weights..."
		self.W2 -= alpha * self.a2.T.dot(delta2)
		self.b2 -= alpha * (delta2).sum(axis=0)

		self.W1 -= alpha * X.T.dot(delta1)
		self.b1 -= alpha * (delta1).sum(axis=0)

		print "\n2nd layer weights head, update: ", self.W2[0]

	def costFunctionPrime(self, X, y):
	    '''Compute derivative with respect to W and W2 for a given X and y:'''
	    # NOT WORKING
	    self.yHat = self.forwardProp(X)
	    
	    delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
	    dJdW2 = np.dot(self.a2.T, delta3)
	    
	    delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
	    dJdW1 = np.dot(X.T, delta2)  
	    
	    return dJdW1, dJdW2

	def train(self, y, training_features):
		print "\nTraining..."
		
		for epoch in range(100):
		    # forward pass
			#A = sigmoid(X.dot(W1) + b1) # A = sigma(Z)
			#Y = softmax(A.dot(W2) + b2) # Y = softmax(Z2)

			yhat = self.forwardProp(training_features)
			#cost1 = nn.costFunction(y, yhat) p
			self.back_pass(yhat, y, training_features) #backprop
			
			yhatt = self.forwardProp(training_features)  # 2nd forward prop
			#cost2 = nn.costFunction(y, yhatt)
			# save loss function values across training iterations
			
			if epoch % 99 == 0:
			    cost = self.costFunction(y, yhat) 
			    print "Loss function value: ", cost
			    #costs.append(loss)
