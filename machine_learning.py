#external
from bitarray import bitarray

#internal
from jsonReader import decode
from recipe import Recipe
from setup import get_ingredients, get_recipes

ingredients_list, ingredients_to_index_dict = get_ingredients()
list_of_recipes = get_recipes(ingredients_to_index_dict)

print(list_of_recipes)
print(list_of_recipes[0])