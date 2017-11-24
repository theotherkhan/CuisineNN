#external
from bitarray import bitarray
import math
import numpy as np


#internal
from jsonReader import decode
from recipe import Recipe


''' HEADER INFO '''
#none

''' UTILITY FUNCTION(S) '''

def zero_list_maker(n):
    listofzeros = '0' * n
    return bitarray(listofzeros) #bitarray(0, n)

''' SETTING UP INGREDIENTS '''
def get_ingredients(name_of_ingredients_file):
    ingredientsDictJSON = decode(name_of_ingredients_file)[0] #dict mapping 'ingredients' to a list of all ingredients
    ingredients = [] #list of all ingredients
    for key in ingredientsDictJSON.keys():
        for value in ingredientsDictJSON[key]:
            ingredients.append(value)

    ingredientsLength = len(ingredients) #ingredients still maps index to ingredient
    ingredientsIndexMapping = {} #dictionary mapping each ingredient to a unique index to be used for arrays
    for i in range(0,ingredientsLength):
        ingredientsIndexMapping[ingredients[i]] = i
    
    return ingredients, ingredientsIndexMapping

''' SETTING UP RECIPES '''
def get_recipes(name_of_recipes_file, ingredientsIndexMapping):
    cookbook = [] #a list of recipes

    trainingSet = decode(name_of_recipes_file)
    ingredientsLength = len(ingredientsIndexMapping.values())
    for dataPoint in trainingSet:
        uid, cuisine, makeUp = dataPoint.values()
        ohs = zero_list_maker(ingredientsLength)
        for ingr in makeUp:
            index = ingredientsIndexMapping[ingr]
            ohs[index] = 1
        r = Recipe(uid, cuisine, ohs)
        cookbook.append(r)

    '''for cook in cookbook:
        print(cook)'''
    
    return cookbook

