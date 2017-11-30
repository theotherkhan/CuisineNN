
from setup import get_ingredients, get_recipes
import numpy as np
from NN import ANN
import pandas as pd

#global ingr
#global recps
global label_maps

ingr = get_ingredients("ingredients.json")
recps = get_recipes("training.json", ingr[1])

label_maps = {"brazilian":0, "british":1, 
 "cajun_creole":2,"chinese":3,
 "filipino": 4,"french": 5,
 "greek": 6,"indian": 7,
 "irish": 8,"italian": 9,
 "jamaican": 10,
 "japanese": 11,
 "korean": 12,
 "mexican": 13,
 "moroccan": 14,
 "russian": 15,
 "southern_us": 16,
 "spanish": 17,
 "thai": 18,
 "vietnamese": 19}

#####################################################################################

def train_test():
    ''' Generates nicely dispersed training and testing data '''

    split_factor = 2
    
    tr_f = np.empty([len(recps)/split_factor, len(ingr[0])])
    tr_l = np.empty([len(recps)/split_factor, 1], dtype = int) 

    te_f = np.empty([len(recps) - len(recps)/split_factor, len(ingr[0])])
    te_l = np.empty([len(recps) - len(recps)/split_factor, 1], dtype = int)

    for i in range (0, len(recps)): 
        if i % 2 == 0:  
            tr_f[i/2] = (np.asarray(list(recps[i].ingredients), dtype = int)) 
            c_recipe = recps[i].uid.encode("ascii")
            tr_l[i/2] = label_maps[c_recipe]

        else:
            te_f[(i-1)/2] = (np.asarray(list(recps[i].ingredients), dtype = int))
            c_recipe = recps[i].uid.encode("ascii")
            te_l[(i-1)/2] = label_maps[c_recipe]

    #print ("\nT&T sizes: \n")
    #print "Training feature shape: ", tr_f.shape
    #print ("Training label: ", tr_l.shape)
    #print ("Testing feature", te_f.shape)
    #print ("Testing label", te_l.shape)

    tr_l = label_vectors(tr_l)
    te_l = label_vectors(te_l)  

    return tr_f, tr_l, te_f, te_l

def label_vectors(training_labels):
    ''' generate correct train label arrays '''
    #y = np.zeros([897, 20])
    y = np.full([897, 20], 0.005, dtype=float)

    for i in range (0, len(training_labels)):
        #print "set at coord: ", i, training_labels[i][0]
        y[i][training_labels[i][0]] = 0.9
    return y

def some_predictions(output, training_labels):
    '''prints NN Cuisine predictions alongside true labels''' 
    print ("\nPredictions:\n")
    #true = ''

    #for i in range (0, 400, 20):  #len(yhat) (shorten predictions) 
    for i in range(len(output)):
        true_label = np.max(training_labels[i])
        output_label = np.argmax(output[i]) 

        #print "Output: ", output_label, "True: ", true_label
        
        for cuisine, idd in label_maps.items():  
            if idd == output_label:
                output = cuisine
        
        for cuisine, idd in label_maps.items():  
            if idd == true_label:
                true = cuisine

        print ("Output, true: ", output, true)

def predictions(output, training_labels):
    '''prints cuisine ID predictions alongside true labels, as tuples''' 
    predictions = [] 
    for i in range (0, len(output)):      
        true_label = np.argmax(testing_labels[i])
        output_label = np.argmax(output[i]) 
        predictions.append((output_label, true_label))
    print "Predictions: \n", predictions

def cleanOutputDisplay(output):
    # Clean the outputs
    o = np.zeros_like(output)    
    for r in range (len(output)):
        o[r][np.argmax(output[r])] = 1
    
    o = np.asarray(o, dtype = int)
    return o

def cuisine_labels(testing_labels):
    cuisine_labels = np.empty([len(testing_labels)], dtype = str)

    for example in range(len(testing_labels)):
        output_label_id = np.argmax(testing_labels[example])

        for cuisine, idd in label_maps.items():  
            if idd == output_label_id:
                cuisine_labels[example] = cuisine

    return cuisine_labels

def acc(output, testing_labels):
    match = 0.0
    for i in range (len(output)):
        #print output[i], testing_labels[i]
        if output[i] == testing_labels[i]:
            match+=1
    return float(match)/float(len(output))





#####################################################################################
#####################################################################################

training_features, training_labels, testing_features, testing_labels = train_test()


num_features = len(ingr[1]) 
num_labels = 20
num_hidden = 10 
l_rate = 0.01
epochs = 5000

#print "\nTraining features: ", training_features
#print "\nTesting labels: ", testing_labels

#print "\nTesting labels (r): ", cuisine_labels(testing_labels)

#print "\nTesting features: ", testing_features
#print "\nTesting labels: ", testing_labels



## RUNNING THE NETWORK #############################


nn = ANN(num_features, num_labels, num_hidden)

for e in range (epochs):

    output = nn.forward(training_features)
    #print ("\nOutput: ", output)

    cost = nn.cost(output, training_labels)
    avg_cost = sum(cost)/len(cost) 
    #print "Average cost on epoch", e, ": ", avg_cost

    dJdW1, dJdW2, dJdB1, dJdB2 = nn.costFunctionPrime(output, training_labels, training_features)

    '''
    print ("\nb1:", nn.b1)
    print ("\ndJdB1: ", dJdB1)
    
    print ("\nb2:", nn.b2)
    print ("\ndJdB2: ", dJdB2) 
    
    print ("\nW1", nn.W1)
    print ("\ndJdW1: ", dJdW1)
    
    print ("\nW2", nn.W2)
    print ("\ndJdW2: ", dJdW2)
    '''
    
    nn.w_update(dJdW1, dJdW2, dJdB1, dJdB2, l_rate)
    
    '''
    print ("\nUpdated B1: ", nn.b1)
    print ("\nUpdated B2: ", nn.b2)
    print ("\nUpdated Weight W1:", nn.W2)
    print ("\nUpdated Weight W2:", nn.W2)
    '''

output = nn.forward(testing_features)

cleanO = cleanOutputDisplay(output)
cleanT = cleanOutputDisplay(testing_labels)

print "\nEnd cost: ", avg_cost
#print "\nOutputs", output
#print "\nClean outputs:\n ", pd.DataFrame(cleanO).head()
#print "\nTesting labels:\n", pd.DataFrame(cleanT).head()

print "\nOutputs (r):\n ", cuisine_labels(output)
print "\nTesting labels (r): ", cuisine_labels(testing_labels)
print "\nAccuracy: ", acc( cuisine_labels(output), cuisine_labels(testing_labels))

#some_predictions(output, testing_labels)















