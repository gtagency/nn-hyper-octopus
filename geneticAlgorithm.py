import random

def runGen(gen, mutRate, maxGen, maxRange):
	nextGen = []
	fitness = []
	for par1 in gen:
		for par2 in gen:
			child = []
			for i in range(len(par1)):
				if random.random() < mutRate:
					child.append(maxRange * random.random())
				else:
					child.append((par1[i] + par2[i]) / 2)
			nextGen.append(child)
			fitness.append(child[0])  # Run NN fitness = accuracy
	average = sum(fitness) / len(fitness)
	if len(nextGen) > 50:
		nextGen = [child for (fit, child) in sorted(zip(fitness, nextGen))]
		return nextGen[len(nextGen) - maxGen:], average
	return nextGen, average

gen = []
mutRate = float(input("Enter mutation rate: "))
numGen = 50
trials = 1000
maxRange = 100
for i in range(numGen):
	gen.append([maxRange * random.random()])
print(gen)
for i in range(trials):
	gen, average = runGen(gen, mutRate, numGen, maxRange)
	print(average)
