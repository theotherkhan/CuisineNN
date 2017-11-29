## Mitchels neural net

import numpy as np
import pandas as pd
from bitarray import bitarray
from network import ANN


num_features = 8
num_labels = 8
num_hidden = 3 
l_rate = 1
epochs = 5000
epoch = 1


training = np.array([ ['10000000', '10000000'],
['01000000', '01000000'],
['00100000', '00100000'],
['00010000', '00010000'],
['00001000', '00001000'],
['00000100', '00000100'],
['00000010', '00000010'],
['00000001', '00000001']])

#training = np.array([ ['10000000', '10000000']])


nn = ANN(num_features, num_labels, num_hidden)

# Build training features
training_features = []
for i in range (len(training)):
	training_features.append( np.asarray( list(training[i][0]), dtype = int )) 
training_features = np.asarray(training_features)



# Build training labels
training_labels = []
for i in range (len(training)):
	training_labels.append( list(np.asarray( list(training[i][1]), dtype = int ))) 
training_labels = np.asarray(training_labels)


print ("\nWeight W2:", nn.W2)
#print ("\nTraining features: \n", pd.DataFrame(training_features))
#print ("\nTrue: \n", pd.DataFrame(training_labels))

output = nn.forward(training_features) #probs of last layer
print ("\nOutput: ", output)

sqes = nn.cost(output, training_labels) #array with all squared errors for all examples 
print ("\nCost: ", sqes)
avg_sqe = sum(sqes)/len(sqes) #average mean squared error across all examples
print ("\nCost Average on epoch", epoch, ": ", avg_sqe)

dJdW1, dJdW2 = nn.costFunctionPrime(output, training_labels, training_features)
#dJdW1, dJdW2 = nn.costFunctionPrime(output, training_labels, training_features)


print ("\ndJdW2: ", dJdW2)
#print ("\ndJdW2 w/ learning rate: ", dJdW2*l_rate)

nn.w_update(dJdW1, dJdW2, l_rate)

print ("\nUpdated Weight W2:", nn.W2)










