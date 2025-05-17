from random import random, randint, shuffle
import numpy as np
from math import exp, sqrt
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional, Union
from matplotlib import cm

class AbstractAnnealModel:

    # Get the "badness" function
    def evaluate(self):
        return 0

    # Make a change that can be reverted
    def revertable_change(self, args = None):
        return

    # Abort the last change made. Note it can work only once in a row.
    # Often it can be done more optimally than just making class copy
    def abort_last_change(self):
        return

    def get_params(self):
        return

class Term:

    def __init__(self, name:str, positive:bool = False):
        self.name = name
        self.positive = positive

    def evaluate(self, eval):
        return eval[self.name] ^ (not self.positive)

    def get_names(self):
        names = set()
        names.add(self.name)
        return names

    def to_string(self):
        if (not self.positive):
            return "!" + self.name
        return self.name

class Disjunction:

    def __init__(self, terms:[Term]):
        self.terms = terms

    def evaluate(self, eval):
        for term in self.terms:
            if term.evaluate(eval):
                return True
        return False

    def get_names(self):
        names = set()
        for term in self.terms:
            for name in term.get_names():
                names.add(name)
        result = list(names)
        return result

    def to_string(self):
        result = "("
        for term in self.terms:
            result += term.to_string()
            result += " " + "|" + " "
        result = result[0:len(result)-3]
        return result + ")"



class Conjunction:
    def __init__(self, disjunctions:[Disjunction]):
        self.disjunctions = disjunctions

    def evaluate(self, eval):
        for disjunction in self.disjunctions:
            if not disjunction.evaluate(eval):
                return False
        return True

    def get_names(self):
        names = set()
        for disjunction in self.disjunctions:
            for name in disjunction.get_names():
                names.add(name)
        result = list(names)
        return result


    def to_string(self):
        result = "("
        for term in self.disjunctions:
            result += term.to_string()
            result += " " + "&" + " "
        result = result[0:len(result)-3]
        return result + ")"

class SATAnnealModel(AbstractAnnealModel):

    def __init__(self, conjunction:Conjunction):
        self.conjunction = conjunction
        self.ev = {}
        self.names = conjunction.get_names()
        self.goodnesses = []
        self.pos_mapper = {}
        self.neg_mapper = {}
        self.badness = 0
        i = 0
        for name in self.names:
            self.ev[name] = True
        for disjunction in self.conjunction.disjunctions:
            self.goodnesses.append(0)
            for term in disjunction.terms:
                if term.evaluate(self.ev):
                    self.goodnesses[len(self.goodnesses) - 1] = self.goodnesses[len(self.goodnesses) - 1] + 1
                if term.name not in self.pos_mapper:
                    self.pos_mapper[term.name] = []
                if term.name not in self.neg_mapper:
                    self.neg_mapper[term.name] = []
                if term.positive:
                    self.pos_mapper[term.name].append(i)
                else:
                    self.neg_mapper[term.name].append(i)
            if self.goodnesses[len(self.goodnesses) - 1] == 0 :
                self.badness = self.badness + 1
            i = i + 1


    def evaluate(self):
        return self.badness

    def true_evaluate(self):
        return self.conjunction.evaluate(self.ev)

    def get_params(self):
        return self.ev


    def mutate(self, name):

        if self.ev[name]:
            for i in self.pos_mapper[name]:
                self.goodnesses[i] -= 1
                if self.goodnesses[i] == 0:
                    self.badness += 1
            for i in self.neg_mapper[name]:
                self.goodnesses[i] += 1
                if self.goodnesses[i] == 1:
                    self.badness -= 1
        else:
            for i in self.pos_mapper[name]:
                self.goodnesses[i] += 1
                if self.goodnesses[i] == 1:
                    self.badness -= 1
            for i in self.neg_mapper[name]:
                self.goodnesses[i] -= 1
                if self.goodnesses[i] == 0:
                    self.badness += 1

        self.ev[name] = not self.ev[name]



    def revertable_change(self, args = None):
        i = len(self.names)
        name = self.names[randint(0, i - 1)]
        self.last_name = name
        self.mutate(name)

    def abort_last_change(self):
        self.mutate(self.last_name)





def exponent_temp_constructor(a, b, skip = 0):
    return lambda i: b * exp(-(i + skip) * a)

def hyperbolic_temp_constructor(a, skip = 0):
    return lambda i: a * (1 / (i + skip))

def rand_n1_1():
    return 2 * random() - 1

def hyperb_change_attrs_constructor(a, skip = 0):
    return lambda i: [a * rand_n1_1() / (i + skip), a * rand_n1_1() / (i + skip)]

def half_hyp_attrs_constructor(a, skip = 0):
    return lambda i: [a * rand_n1_1() / sqrt((i + skip)), a * rand_n1_1() / sqrt((i + skip))]


def const_change_attrs_constructor(a, skip = 0):
    return lambda i: [a * rand_n1_1(), a * rand_n1_1()]

def exp_change_attrs_constructor(a, b, skip = 0):
    return lambda i: [a * rand_n1_1() * exp(-(i + skip) * b), a * rand_n1_1() * exp(-(i + skip) * b)]

def default_tolerance_func(change, temperature):
    return exp(-change / temperature)

def discrete_tolerance_func(change, temperature):
    return temperature



maps = {}
def simulate_annealing(annealing_model : AbstractAnnealModel, temperature_func, change_attrs_func,
                       iterations : int, tolerance_func = discrete_tolerance_func):

    states = []
    states.append(annealing_model.get_params().copy())
    for i in range(1, iterations + 1):

        if annealing_model.evaluate() == 0:
            print("Finished successfully, took iterations: " + str(i))
            break

        before_result = annealing_model.evaluate()

        annealing_model.revertable_change(change_attrs_func(i))

        after_result = annealing_model.evaluate()


        if before_result >= after_result:
            states.append(annealing_model.get_params().copy())
            # good change
        elif random() < tolerance_func(after_result - before_result, temperature_func(i)):
            states.append(annealing_model.get_params().copy())
            # bad change, but we're lucky this time
        else:
            annealing_model.abort_last_change()
            # revert bad changes

    return states


def generate_CNF(variables_count, disj_size, conj_size):
    variables = []
    counter = 0
    for first in range(ord('a'), ord('z') + 1):
        for second in range(ord('a'), ord('z') + 1):
            for third in range(ord('a'), ord('z') + 1):
                variables.append(chr(first) + chr(second) + chr(third))
                counter += 1
                if counter >= variables_count:
                    break
            if counter >= variables_count:
                break
        if counter >= variables_count:
            break

    eval_v = {}
    for variable in variables:
        if randint(0, 1) == 0:
            eval_v[variable] = False
        else:
            eval_v[variable] = True

    disjunctions = []
    for i in range(conj_size):
        terms = []

        main_variable = randint(0, variables_count - 1)
        term = Term(variables[main_variable], eval_v[variables[main_variable]])
        terms.append(term)
        for j in range(disj_size - 1):
            rand_variable = randint(0, variables_count - 1)
            while (rand_variable == main_variable):
                rand_variable = randint(0, variables_count - 1)
            if randint(0, 1) == 0:
                term = Term(variables[rand_variable], not eval_v[variables[rand_variable]])
                terms.append(term)
            else:
                term = Term(variables[rand_variable], eval_v[variables[rand_variable]])
                terms.append(term)

        shuffle(terms)
        disjunction = Disjunction(terms)
        disjunctions.append(disjunction)
    conjunction = Conjunction(disjunctions)
    return conjunction


offset = 0

def parser_terms(string):
    global offset
    while (string[offset] == ' ' or string[offset] == '|'):
        offset += 1

    truth = True
    if (string[offset] == '!'):
        truth = False
        offset += 1
    name = ''
    while(string[offset] != ' ' and string[offset] != '|' and string[offset] != '(' and string[offset] !=')'):
        name += string[offset]
        offset += 1

    return Term(name, truth)

def parser_disjunction(string):

    global offset
    while (string[offset] == ' ' or string[offset] == '&'):
        offset += 1
    terms = []
    if (string[offset] == '('):
        offset += 1

        while (not string[offset] == ')'):
            term = parser_terms(string)
            terms.append(term)
            while (string[offset] == ' ' or string[offset] == '|'):
                offset += 1
        offset += 1

    return Disjunction(terms)

def parser_CNF(string):
    global offset
    while string[offset] == ' ':
        offset += 1
    disjunctions = []
    if (string[offset] == '('):
        offset += 1
        while string[offset] == ' ':
            offset += 1
        while (not string[offset] == ')'):
            disjunction = parser_disjunction(string)
            disjunctions.append(disjunction)
            while (string[offset] == ' ' or string[offset] == '&'):
                offset += 1
        offset += 1

    return Conjunction(disjunctions)


def sat_experiment():
    conjunction = generate_CNF(2500, 3, 10000)
    print(conjunction.to_string())

    model = SATAnnealModel(conjunction)
    temperature_func = hyperbolic_temp_constructor(100, 100)
    change_attrs_func = const_change_attrs_constructor(1)
    iterations = 1000000
    states = simulate_annealing(model, temperature_func, change_attrs_func, iterations)

    final_eval = states[len(states)-1]

    for name in conjunction.get_names():
        print(name + ": " + str(final_eval[name]))

    print(conjunction.evaluate(final_eval))

def client():
    print("Введите строку в КНФ")
    s = input()
    conjunction = parser_CNF(s)

    model = SATAnnealModel(conjunction)
    temperature_func = hyperbolic_temp_constructor(100, 100)
    change_attrs_func = const_change_attrs_constructor(1)
    iterations = 1000000
    states = simulate_annealing(model, temperature_func, change_attrs_func, iterations)

    final_eval = states[len(states)-1]

    for name in conjunction.get_names():
        print(name + ": " + str(final_eval[name]))

    print(conjunction.evaluate(final_eval))

if __name__ == '__main__':
    client()