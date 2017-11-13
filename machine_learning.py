#external
from bitarray import bitarray
import sys as sys

#internal
from jsonReader import decode
from recipe import Recipe
from setup import get_ingredients, get_recipes

def run_program(name_of_atr_file, name_of_training_file):
    attribute_list, attribute_to_index_dict = get_ingredients(name_of_atr_file)
    training_set = get_recipes(name_of_training_file, attribute_to_index_dict)

    print(training_set)
    print(training_set[0])
    #next code here




if __name__ == '__main__':
    if(len(sys.argv) > 2):
        #print(str(sys.argv))
        run_program(sys.argv[1], sys.argv[2])