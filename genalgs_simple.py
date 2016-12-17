from random import uniform,randint,random

point1 = [0, 1]
point2 = [1, 3]
point3 = [2, 7] 

#generates an equation 
def gen_eq(min1, max1, min2, max2, min3, max3):
    return [uniform(min1, max1), uniform(min2, max2), uniform(min3, max3)]

def population(count, min1, max1, min2, max2, min3, max3):
    return [gen_eq(min1, max1, min2, max2, min3, max3) for x in xrange(count)]

def fitness(individual, point1, point2, point3):
    point1_fitted = (individual[0] * point1[0]**2) + (individual[1] * point1[0]) + individual[2]     
    point2_fitted = (individual[0] * point2[0]**2) + (individual[1] * point2[0]) + individual[2]
    point3_fitted = (individual[0] * point3[0]**2) + (individual[1] * point3[0]) + individual[2]
    MSE = (point1[1] - point1_fitted) ** 2 + (point2[1] - point2_fitted) ** 2 + (point3[1] - point3_fitted) ** 2
    return MSE

def grade_pop(pop, point1, point2, point3):
    net_err = 0
    for individual in pop:
        net_err += fitness(individual, point1, point2, point3)
    return (net_err/len(pop))

def evolve(pop, point1, point2, point3):
    graded = [(fitness(x, point1, point2, point3), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    parents = graded[:20]
    #randomly add other individuals
    for individual in graded[20:]:
        if 0.05 > random():
            parents.append(individual)
    #mutate random individuals
    for individual in parents:
        if 0.05 > random():
            pos_to_mutate = randint(0, len(individual) - 1)
            individual[pos_to_mutate] = uniform(min(individual), max(individual))
    #breed
    children = []
    while len(children) < (len(pop) - len(parents)):
        male = randint(0,len(parents)-1)
        female = randint(0,len(parents)-1)
        if male != female:
            child = parents[male][:1] + parents[female][1:]
            children.append(child)
    parents.extend(children)
    return parents


p =  population(100, -10, 10, -10, 10, -10, 10)
for i in xrange(1000):    
    p = evolve(p, point1, point2, point3)
    print grade_pop(p, point1, point2,point3)
print p[0]
