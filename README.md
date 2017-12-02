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

### Accuracies:

|------ | Epochs                                                                |
|------------------------------------------------------------------------------	|
| FOLD 	| 50             	| 500            	| 1000           	| 2000           	|
|------	|----------------	|----------------	|----------------	|----------------	|
| 0    	| 0.311036789298 	| 0.698996655518 	| 0.839464882943 	| 0.846153846154 	|
| 1    	| 0.347826086957 	| 0.792642140468 	| 0.795986622074 	| 0.826086956522 	|
| 2    	| 0.357859531773 	| 0.755852842809 	| 0.779264214047 	| 0.886287625418 	|
| 3    	| 0.267558528428 	| 0.648829431438 	| 0.8127090301   	| 0.859531772575 	|
| 4    	| 0.377926421405 	| 0.705685618729 	| 0.832775919732 	| 0.852842809365 	|
| 5    	| 0.327759197324 	| 0.675585284281 	| 0.775919732441 	| 0.846153846154 	|

50: 0.331661092531
500: 0.712931995541
1000: 0.80602006689
2000: 0.852842809365

### Costs / Errors
50:
500:
1000:
2000:
