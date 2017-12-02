# CuisineConnoisseur

### How To Run:
1. path to the directory with driver.py, NN.py, setup.py, recipe.py, jsonReader.py, and run.sh
2. make sure to include in that directory ingredients.json and training.json
3. execute `./run.sh`
  * NOTE: the bash script is currently configured to operate on ingredients.json and training.json; if you wish to change this, simply change those input statements for another json of attributes and training data.

### Adjustable Parameters:
These can be found at the top of driver.py under the comment 'Adjustable Hyperparameters'
* epochs -> this is the number of back-propogations which will be performed per fold
* k -> this is the number of folds to divide the data into
* l_rate -> this is the learning rate that determines how far the delta-weights can pull the previous weights per epoch

### Method -- Neural Network:

We used a neural network for this project. Our project consists of a setup.py file, an NN.py file which defines the ANN() class containing the meat of the neural network architecture, and a driver.py file which conducts preprocessing tasks, builds training/testing features & labels, and runs the network using a 6-fold cross validation testing scheme.

Our setup file is a simple wrapper for readability. It is a series of related functions that takes in the names of json files to read in, and then makes use of our jsonReader class to setup up a list of attributes and a list of training data. For the training data, it will initialize an instance of the recipe class for each input and, for each, will assign its unique id number (uid), cuisine category string (label), and an ingredient bitarray (ingredients). This bitarray contains a bit for each attribute where non-present ingredients are a 0 and present ones are a 1.

Our ANN class defines a generalized artificial neural network with 3 layers: a input layer, a single hidden layer, and an output layer. Our hidden layer consists of 10 neurons. The forward() function of the ANN propogates all training-example features through our network, all at once. The backprop() function calculates the delta-weight values, including the weight on the biases. These weights are subsequently updated in the w_update() function. By using matrices that hold the information of all our training examples and running all the examples through the network in each epoch, we can pass forward and then back-propogate through our network in a fast and optimized fashion (i.e we are not using stochastic GD).

Let's get a little more into the intuitive explanation behind back-propogation. We know we want to change our weights based on the network cost (the mean squared error over all training examples, for a given epoch). More precisely, we want find the rate-of-change between our cost values with respect to our weights. Importantly, the costs result from forward-propogating through our network, getting the output, and subtracting with our true labels. We can thus say that our cost is a function of our output and true labels. Our output (output layer activation) is subsequently a function of our z2 term, which is a function our W2 term, B2 term, and a1 term (and so on for each hypotehtical layer). It follows then that indirectly, we can say that our cost is a function of our weights. In order to calculate the derivative of our cost function, we must multiply the chain of partial derivatives that allow us to relate the rate of change of weights to the cost.  

Our driver function defines all our network hyperparameters (number of epochs, number of k-folds, learning rate, etc). It builds 6 'clusters' of data using the cf_validation() and kfold() functions, from which we subsequntly build a set of all possible train/test pairings. We subsequently run the network on each of the six pairings using run(), producing the mean squared error (cost) and accuracies for each pairing. We then finally calculate the average overall accuracy and cost of our neural network.

### Accuracies:

Some sample accuracies from 6-fold cross-validation (_highest per is italicized_, **lowest is bolded**):

| FOLD 	| 50 epochs      	    | 500 epochs          | 1000 epochs         | 2000 epochs    	    |
|------	|--------------------	|--------------------	|--------------------	|--------------------	|
| 0    	| 0.31103678929 	    | 0.69899665551 	    | _0.83946488294_ 	  | 0.84615384615 	    |
| 1    	| 0.34782608695 	    | _0.79264214046_ 	  | 0.79598662207 	    | **0.82608695652** 	|
| 2    	| 0.35785953177 	    | 0.75585284280 	    | 0.77926421404 	    | _0.88628762541_ 	  |
| 3    	| **0.26755852842** 	| **0.64882943143** 	| 0.81270903010 	    | 0.85953177257 	    |
| 4    	| _0.37792642140_ 	  | 0.70568561872 	    | 0.83277591973 	    | 0.85284280936 	    |
| 5    	| 0.32775919732 	    | 0.67558528428 	    | **0.77591973244** 	| 0.84615384615 	    |

And averages:

|     	   | 50 epochs      	    | 500 epochs          | 1000 epochs         | 2000 epochs    	    |
|--------- |--------------------	|--------------------	|--------------------	|--------------------	|
| **AVRG** | 0.33166109253 	      | 0.71293199554 	    | 0.80602006689 	    | 0.85284280936 	    |

![Average Accuracies Over Epoch Number](images/avgacc.png)

### Costs / Errors:

Some sample costs / errors from 6-fold cross-validation (**highest per is bolded**, _lowest is italicized_):

| FOLD 	| 50             	    | 500            	    | 1000           	    | 2000           	    |
|------	|--------------------	|--------------------	|--------------------	|--------------------	|
| 0    	| 0.7018267821 	      | 0.29519142114 	    | _0.17096196781_ 	  | 0.14506582411 	    |
| 1    	| 0.6752300708 	      | _0.24202939300_ 	  | 0.21456706118 	    | **0.15949890919** 	|
| 2    	| 0.6986968620 	      | 0.25601260771 	    | 0.19918335260 	    | _0.11209239151_   	|
| 3    	| **0.7147797920** 	  | **0.36826245843** 	| 0.19385758772 	    | 0.12898899282 	    |
| 4    	| _0.6976042622_ 	    | 0.30609310475 	    | 0.18539573498 	    | 0.12598703630 	    |
| 5    	| 0.6989790947 	      | 0.30729006119 	    | **0.23355297715** 	| 0.14221348664 	    |

And averages:

|     	   | 50 epochs      	    | 500 epochs          | 1000 epochs         | 2000 epochs    	    |
|--------- |--------------------	|--------------------	|--------------------	|--------------------	|
| **AVRG** | 0.69785281067 	      | 0.29581317437 	    | 0.1995864469 	      | 0.13564110676 	    |

![Average Accuracies Over Epoch Number](images/avgcost.png)

### Useful Sources:
* [3blue1brown ANN Tutorial](https://www.youtube.com/watch?v=tIeHLnjs5U8)
* [Welsch Labs Neural Networks Demystified Tutorials](https://www.youtube.com/watch?v=bxe2T-V8XRs)

### Authors:
* Hasan Khan kh4cd
* Zachary Danz zsd4yr
