# CuisineConnoisseur

### How To Run:
1. path to the directory with driver.py, NN.py, setup.py, recipe.py, jsonReader.py, and run.sh
2. make sure to include in that directory ingredients.json and training.json
3. execute `./run.sh`
  * NOTE: the bash script is currently configured to operate on ingredients.json and training.json; if you wish to change this, simply change those input statements for another json of attributes and training data.

### Adjustable Parameters:
These can be found at the top of driver.py under the comment 'Adjustable Hyperparameters'
* epochs -> this is the number of back-propogations which will be performed per fold
* k -> this is the number of folds to divide training data into
* l_rate -> this is the learning rate that determines how far the delta-weights can pull the previous weights per epoch

### Accuracies:

Some sample accuracies from 6-fold cross-validation (_highest per is italicized_, **lowest is bolded**):

| FOLD 	| 50 epochs      	    | 500 epochs          | 1000 epochs         | 2000 epochs    	    |
|------	|--------------------	|--------------------	|--------------------	|--------------------	|
| 0    	| 0.311036789298 	    | 0.698996655518 	    | _0.839464882943_ 	  | 0.846153846154 	    |
| 1    	| 0.347826086957 	    | _0.792642140468_ 	  | 0.795986622074 	    | **0.826086956522** 	|
| 2    	| 0.357859531773 	    | 0.755852842809 	    | 0.779264214047 	    | _0.886287625418_ 	  |
| 3    	| **0.267558528428** 	| **0.648829431438** 	| 0.812709030100 	    | 0.859531772575 	    |
| 4    	| _0.377926421405_ 	  | 0.705685618729 	    | 0.832775919732 	    | 0.852842809365 	    |
| 5    	| 0.327759197324 	    | 0.675585284281 	    | **0.775919732441** 	| 0.846153846154 	    |

And averages:

|     	| 50 epochs      	    | 500 epochs          | 1000 epochs         | 2000 epochs    	    |
|------	|--------------------	|--------------------	|--------------------	|--------------------	|
| AVR 	| 0.331661092531 	    | 0.712931995541 	    | 0.806020066890 	    | 0.852842809365 	    |

### Costs / Errors

Some sample costs / errors from 6-fold cross-validation (**highest per is bolded**, _lowest is italicized_):

| FOLD 	| 50             	    | 500            	    | 1000           	    | 2000           	    |
|------	|--------------------	|--------------------	|--------------------	|--------------------	|
| 0    	| 0.701826782104 	    | 0.295191421140 	    | _0.170961967811_ 	  | 0.145065824116 	    |
| 1    	| 0.675230070862 	    | _0.242029393000_ 	  | 0.214567061180 	    | **0.159498909199** 	|
| 2    	| 0.698696862067 	    | 0.256012607714 	    | 0.199183352604 	    | _0.112092391518_   	|
| 3    	| **0.714779792030** 	| **0.368262458434** 	| 0.193857587727 	    | 0.128988992820 	    |
| 4    	| _0.697604262262_ 	  | 0.306093104757 	    | 0.185395734980 	    | 0.125987036302 	    |
| 5    	| 0.698979094749 	    | 0.307290061198 	    | **0.233552977155** 	| 0.142213486649 	    |

And averages:

|     	| 50 epochs      	    | 500 epochs          | 1000 epochs         | 2000 epochs    	    |
|------	|--------------------	|--------------------	|--------------------	|--------------------	|
| AVR 	| 0.697852810679 	    | 0.295813174374 	    | 0.19958644691 	    | 0.135641106767 	    |
