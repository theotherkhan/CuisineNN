
from setup import get_ingredients, get_recipes
import numpy as np
from NN import ANN

ingr = get_ingredients("ingredients.json")
recps = get_recipes("training.json", ingr[1])

global label_maps

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

    tr_lv = np.empty([len(recps)/split_factor, 1]) 
    te_lv = np.empty([len(recps)/split_factor, 1]) 

    for i in range (0, len(recps)): 
        if i % 2 == 0:  
            tr_f[i/2] = (np.asarray(list(recps[i].ingredients), dtype = int)+1) 
            c_recipe = recps[i].uid.encode("ascii")
            tr_l[i/2] = label_maps[c_recipe]

        else:
            te_f[(i-1)/2] = (np.asarray(list(recps[i].ingredients), dtype = int)+1)
            c_recipe = recps[i].uid.encode("ascii")
            te_l[(i-1)/2] = label_maps[c_recipe]

    print "\nT&T sizes: \n"
    print "Training feature: ", tr_f.shape
    print "Training label: ", tr_l.shape
    print "Testing feature", te_f.shape
    print "Testing label", te_l.shape
    

    return tr_f, tr_l, te_f, te_l

def label_vectors():
    ''' generate correct train label arrays '''
    #y = np.zeros([897, 20])
    y = np.full([897, 20], 0.1, dtype=float)

    for i in range (0, len(training_labels)):
        #print "set at coord: ", i, training_labels[i][0]
        y[i][training_labels[i][0]] = 0.9
    return y

def some_predictions(yhat, training_labels):
    '''prints NN Cuisine predictions alongside true labels''' 
    print "\nPredictions:\n"

    for i in range (0, 400, 20):  #len(yhat) (shorten predictions)   
        true_label = np.max(training_labels[i])
        output_label = np.argmax(yhat[i]) 

        #print "Output: ", output_label, "True: ", true_label
        
        for cuisine, idd in label_maps.items():  
            if idd == output_label:
                output = cuisine
        
        for cuisine, idd in label_maps.items():  
            if idd == true_label:
                true = cuisine

        print "Output, true: ", output, true

def predictions(yhat, training_labels):
    '''prints cuisine ID predictions alongside true labels, as tuples''' 
    predictions = [] 
    for i in range (0, len(yhat)):      
        true_label = np.max(training_labels[i])
        output_label = np.argmax(yhat[i]) 
        predictions.append((output_label, true_label))
    print "Predictions: \n", predictions

#####################################################################################
#####################################################################################

training_features, training_labels, testing_features, testing_labels = train_test()


num_features = len(ingr[1]) 
num_labels = 20
num_hidden = 4000 # lower this

#print training_features
#print training_labels
#print testing_features
#print testing_labels

## FORWARD PROPOGATION #############################

nn = ANN(num_features, num_labels, num_hidden)
y = label_vectors() 

print "\nWeight W1:", nn.W1
#print "\nWeight W2:", nn.W2

yhat = nn.forwardProp(training_features)

print "\nyhat: ", yhat

print "\ny: ", y

some_predictions(yhat, training_labels)

cost1 = nn.costFunction(yhat,y)

print "Cost1: ", cost1

dJdW1, dJdW2  = nn.costFunctionPrime(yhat, y, training_features)

print "\ndJdW1:", dJdW1
print "\nUpdating weights..."

scalar = 100
nn.W1 = nn.W1 - scalar*dJdW1
nn.W2 = nn.W2 - scalar*dJdW2

#print "\nWeight W1:", nn.W1

yhatt = nn.forwardProp(training_features)

cost2 = nn.costFunction(yhatt,y)

print "\nCost1 Avg: ", np.sum(cost1)/len(cost1)
print "\nCost2 Avg: ", np.sum(cost2)/len(cost2)

#print "\nCost2: ", cost2

print "********************************************************"


#predictions(yhat, training_labels)
some_predictions(yhatt, training_labels)

'''
cost1 = nn.costFunction(y, yhat)

nn.back_pass(yhat, y, training_features)
yhatt  = nn.forwardProp(training_features)

cost2 = nn.costFunction(y, yhatt)

#print "\nCost1: ", cost1
#print "\nCost2: ", cost2

print "\n \n"

print_predictions(yhat, training_labels)

'''













