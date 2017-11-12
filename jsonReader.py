import simplejson as json

def decode(file):
    with open(file, "r") as ins:
        array = []
        current = []
        for line in ins:
            array.append(line)
        for elem in array:
            current.append(json.loads(elem))
        return current

#ingredients = decode('ingredients.json')
#print(ingredients)

#print("###################################################################")

#training = decode('training.json')
#print(training)
