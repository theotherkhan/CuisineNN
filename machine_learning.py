#external
from bitarray import bitarray
import sys as sys

#internal
from recipe import Recipe
from setup import get_ingredients, get_recipes

def run_program(name_of_atr_file, name_of_training_file):
    attribute_list, attribute_to_index_dict = get_ingredients(name_of_atr_file)
    training_set = get_recipes(name_of_training_file, attribute_to_index_dict)

    #print(training_set)
    print(str(training_set[0]) + '\n')
    print(str(training_set[0].ingredients[0]))
    print(str(training_set[0].ingredients[6]))
    #next code here




if __name__ == '__main__':
    if(len(sys.argv) > 2):
        #print(str(sys.argv))
        run_program(sys.argv[1], sys.argv[2])


#run_program("ingredients.json", "training.json")