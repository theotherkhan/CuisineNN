#external
from bitarray import bitarray

#internal
from jsonReader import decode
from recipe import Recipe
from setup import get_ingredients, get_recipes

attribute_list, attribute_to_index_dict = get_ingredients()
training_set = get_recipes(attribute_to_index_dict)

print(training_set)
print(training_set[0])