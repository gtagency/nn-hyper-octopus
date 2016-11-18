import tensorflow as tf
import numpy as np
from random import uniform, random, randint
import sys

class NN(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        with self.g.as_default():
            self.X = tf.placeholder(tf.float32, [178 , 13], name ='X')
            self.Y = tf.placeholder(tf.float32, [178, 3], name = 'Y')
            W = tf.Variable(tf.zeros([13, 3]), name='W')
            b =  tf.Variable(tf.zeros([3]), name='b')
            pred = tf.sigmoid(tf.matmul(self.X, W) + b)
            self.cost = tf.reduce_mean(tf.square(self.Y - pred))
            self.train_step = tf.train.GradientDescentOptimizer(uniform(0, self.learning_rate)).minimize(self.cost)
            self.init = tf.initialize_all_variables()

input_data = np.loadtxt("data/wine.data",float,"#",",")
extractedData = input_data[:,0]
input_data = np.delete(input_data, 0, 1)
extractedData = extractedData.tolist();
extractedData[:] = [x - 1 for x in extractedData]
extractedData = np.eye(3)[extractedData]

def gen_individual(learning_rate):
    return NN(uniform(0, learning_rate))

def population(count):
    return [gen_individual(0.001) for x in range(count)]

def fitness(individual):
    sess=tf.InteractiveSession(graph=individual.g)
    data = {individual.X: input_data.reshape(178, 13), individual.Y: extractedData.reshape(178, 3)}
    sess.run(individual.init)
    for i in range(25000):
        sess.run(individual.train_step, feed_dict=data)
    print('.',end="")
    sys.stdout.flush()
    return sess.run(individual.cost, feed_dict=data)

def evolve(pop):
    graded = [(fitness(x), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    parents = graded[:5]
    #randomly add other individuals
    for individual in graded[5:]:
        if 0.05 > random():
            parents.append(individual)
    #mutate random individuals
    for individual in parents:
        if 0.05 > random():
            individual.learning_rate *= 0.5
    #breed
    children = []
    while len(children) < (len(pop) - len(parents)):
        male = randint(0, len(parents) - 1)
        female = randint(0, len(parents) - 1)
        if male != female:
            child_learning_avg = (parents[male].learning_rate/2) + (parents[female].learning_rate/2)
            children.append(NN(child_learning_avg))
    parents.extend(children)
    return parents 

#print(fitness(gen_individual(0.0004)))
pop = population(20)
for i in range(25):
    pop = evolve(pop)
    print(" ")

print(pop[0].learning_rate)
print(fitness(pop[0]))
