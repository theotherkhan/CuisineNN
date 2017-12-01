
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


def label_vectors(training_labels, k_group_size, num_labels, def_label_init):
    ''' generate correct train label arrays '''

    y = np.full([k_group_size, num_labels], def_label_init, dtype=float)
    
    for i in range (0, len(training_labels)):
        y[i][training_labels[i][0]] = 0.9
    
    return y

def predictions(output, training_labels):
    '''prints cuisine ID predictions alongside true labels, as tuples''' 
    predictions = [] 
    for i in range (0, len(output)):      
        true_label = np.argmax(testing_labels[i])
        output_label = np.argmax(output[i]) 
        predictions.append((output_label, true_label))
    print "Predictions: \n", predictions

def cleanOutputDisplay(x):
    '''Cleans labels into a readable, interpretable format'''
    o = np.zeros_like(x)  
    for r in range (len(x)):
        o[r][np.argmax(x[r])] = 1
    
    o = np.asarray(o, dtype = int)
    return o

def cuisine_labels(testing_labels):
    '''Converts testing labels arrays into readable labels''' 
    cuisine_labels = np.empty([len(testing_labels)], dtype = str)

    for example in range(len(testing_labels)):
        output_label_id = np.argmax(testing_labels[example])

        for cuisine, idd in label_maps.items():  
            if idd == output_label_id:
                cuisine_labels[example] = cuisine

    return cuisine_labels

def acc(output, testing_labels):
    ''' Given output predictions and true labels, returns network accuracy'''
    match = 0.0
    for i in range (len(output)):
        #print output[i], testing_labels[i]
        if output[i] == testing_labels[i]:
            match+=1
    return float(match)/float(len(output))

def tt(feature_clusters, label_clusters, k):
    ''' Returns a list of all possible train/test sets in accordance with given k-fold cross validation'''
    allPossible = []
    
    for i in range (1):
        
        testing_features = np.asarray(feature_clusters[i].tolist())
        testing_labels = np.asarray(label_clusters[i])

        if (i==0):
            #print "tuple: ", feature_clusters[i+1:]
            training_features = np.concatenate(feature_clusters[i+1:].tolist())
            training_labels = np.concatenate(label_clusters[i+1:].tolist()) 
        elif(i==k):
            training_features = np.concatenate(feature_clusters[:i].tolist()) 
            training_labels = np.concatenate(label_clusters[:i].tolist())
        else: 
            training_features = np.concatenate( np.concatenate(feature_clusters[i+1:].tolist()), np.concatenate(feature_clusters[:i].tolist()))  
            training_labels = np.concatenate(np.concatenate(label_clusters[i+1:].tolist()), np.concatenate(label_clusters[:i].tolist()))  

        allPossible.append([training_features, training_labels, testing_features, testing_labels])
    
    return allPossible

def cf_validation(k, recps, num_datapoints):

    feature_clusters = np.empty([k, num_datapoints/k], dtype = list)
    label_clusters = np.empty([k, num_datapoints/k]) 

    for group in range(0, k):
        cluster_i = 0
        for i in range (0, len(recps)-1, k): 
            #while i < len(recps) 
            #print "group, cluster_i:", group, cluster_i
            feature_clusters[group][cluster_i] = (np.asarray(list(recps[i].ingredients), dtype = int)) 

            c_recipe = recps[i].uid.encode("ascii")
            label_clusters[group][cluster_i] = label_maps[c_recipe]

            cluster_i +=1

    return feature_clusters, label_clusters

# This function is for holdover cross validation; debugging purpose only:
def train_test():
    ''' Generates nicely dispersed training and testing data, holdover cv '''

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

    tr_l = label_vectors(tr_l, 897, 20, 0.05)
    te_l = label_vectors(te_l, 897, 20, 0.05)  

    return tr_f, tr_l, te_f, te_l


#####################################################################################
#####################################################################################

# Hyperparamters
num_features = len(ingr[1])
num_datapoints = 1794
num_labels = 20
num_hidden = 10 
l_rate = 0.01
epochs = 5000
k = 6
k_group_size = num_datapoints / k
def_label_init = 0.005

# Non cross validation train/test
training_featuresB, training_labelsB, testing_featuresB, testing_labelsB = train_test()

# Build cross validation train/test
feature_clusters, label_clusters = cf_validation(k, recps, num_datapoints)
training_features, training_labels, testing_features, testing_labels = tt(feature_clusters, label_clusters, k)[0]

testing_labels = np.reshape(testing_labels, (k_group_size, 1))
testing_labels = label_vectors(testing_labels, k_group_size, num_labels, def_label_init)

training_labels = np.reshape(training_labels, (k_group_size*(k-1), 1))
training_labels = label_vectors(training_labels, k_group_size*(k-1), num_labels, def_label_init)

#training_features.shape = (1495, 2398)

print "TF types: ", type(training_features), type(training_features[0]), type(training_features[0][0])
print "TFB types: ", type(training_featuresB), type(training_featuresB[0]), type(training_featuresB[0][0])

print "Training feature shapes: ", training_features.shape, training_features[0].shape, training_features[0][0].shape
print "Training feature B shape, type: ", training_featuresB.shape, training_featuresB[0].shape, training_featuresB[0][0].shape

print "Testing feature shapes: ", testing_features.shape, testing_features[0].shape, testing_features[0][0].shape
print "Testing feature B shape, type: ", testing_featuresB.shape, testing_featuresB[0].shape, testing_featuresB[0][0].shape

#print "Training labels shape: ", training_labels.shape
print "\nTraining label shapes: ", training_labels.shape
print "Training label B shape, type: ", training_labelsB.shape

print "Testing label shapes: ", testing_labels.shape
print "Testing label B shape, type: ", testing_labelsB.shape


#print "Testing labels shape: ", testing_labels.shape

print "\nTesting labels: ", testing_labels
print "\nTesting labels B: ", testing_labelsB


## RUNNING THE NETWORK #############################


nn = ANN(num_features, num_labels, num_hidden)

for e in range (epochs):

    output = nn.forward(training_features)
    print "otuput shape:", output.shape
    #print ("\nOutput: ", output)

    cost = nn.cost(output, training_labels)
    avg_cost = sum(cost)/len(cost) 
    print "Average cost on epoch", e, ": ", avg_cost

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














