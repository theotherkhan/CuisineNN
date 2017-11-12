#external
from bitarray import bitarray

class Recipe:
    uid = -1 #unique id
    label = '' #label for type of cuisine
    ingredients = bitarray(0) #list of bits that represent which ingredients are present and which are not

    def __init__(self, uid, label, ingredients):
        self.uid = uid
        self.label = label
        self.ingredients = ingredients

    def is_ingredient_included(index):
        if (index > len(ingredients)): return false
        return ingredients[index] == 1

    def __str__(self):
        return str(self.uid) + ': ' + str(self.label) + ' -> ' + str(self.ingredients)
    

