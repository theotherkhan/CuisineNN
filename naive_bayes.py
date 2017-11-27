# externals
from bitarray import bitarray
import math

# internals
from recipe import Recipe

# recall that P(c|x) = [(P(x|c) * P(c)) / P(x)]
# where P(c|x) is the posterior prob. of class(c, target)
#   given predictor(x, attributes);
# P(c) is the prior prob. of class;
# P(x|c) is the likelihood which is the prob. of predictor given class;
# and P(x) is the prior prob. of the predictor

def separate_by_label(training_set):
    labels = []
    separated_by_class = {}
    for i in range(len(training_set)):
        recipe = training_set[i]
        label = recipe.label
        if label not in separated_by_class:
            separated_by_class[label] = []
            labels.append(label)
        separated_by_class[label].append(recipe)
    return separated_by_class, labels

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def gaussian_prob(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def summary_for_attr(list_of_examples_for_label, value_j):
    list_of_atr_absent = [] #for the two values an atr can hold
    list_of_atr_present = []
    for k in range(len(list_of_examples_for_label)): #for each example
        example = list_of_examples_for_label[k]
        value = example.ingredients[value_j]
        if value == True:
            list_of_atr_present.append(1)
            list_of_atr_absent.append(0)
        else if value == False:
            list_of_atr_present.append(0)
            list_of_atr_absent.append(1)
    me_ab = mean(list_of_atr_absent)
    me_pr = mean(list_of_atr_present)
    st_ab = stdev(list_of_atr_absent)
    st_pr = stdev(list_of_atr_present)

    ab = (me_ab, st_ab)
    pr = (me_pr, st_pr)
    summar = (ab, pr)

    return summar


def summarize_by_class(attributes, training_set):
    separation_by_labels, labels = separate_by_label(training_set)
    summary = {}
    # there are many classes and two values for each attribute 
    # (absent 0 or present 1)
    # typography:
    #   Cl1: [summary_atr_0, summary_atr_1]
    #       where summary_atr_i = (absent_summary, present_summary)
    #           and absent_summary = (mean_absent, stdev_abset)
    # so each attribute summary is a tuple of tuples of the 
    # mean and stdev of its possible values
    # while the summary for a label is a list of these for each attribute
    for i in range(len(labels)): #for each label
        label = labels[i]
        list_of_examples_for_label = separation_by_labels[label]
        for j in range(len(attributes)): #for each attribute
            if label not in summary:
                summary[label] = []

            summar = summary_for_attr(list_of_examples_for_label, j)
            summary[label].append(summar)
    
    return summary, labels


            
            
           






