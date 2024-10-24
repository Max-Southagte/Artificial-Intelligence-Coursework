import random
import numpy

###--FUNCTIONS--###

### generates a list of pops according to the input variables
def initial_population(pop_size, chromosome_count): 
    population = []
    for i in range(pop_size):
        population.append(random.choices(range(2), k=chromosome_count))
    return population

### adds numbers 1 through 8, 24 through 32, and the inverse of 8 through 24 to find the total fitness. a solution will have a fitness of 32
def fitness_func(individual): 
    #print(individual)
    ones_total = sum(individual[0:8] + individual[24:32])
    zeros_total = sum(individual[8:24])
    return ones_total + (16 - zeros_total)

def roulette_wheel(fitness, population):
    parents = []
    fitness_total = sum(fitness)
    normalised = [x/fitness_total for x in fitness]

    ### creates the "wheel" values by cumulativly adding all of the normalised values together, up to 1.0
    f_cumulative = []
    index = 0
    for n in normalised:
        index += n
        f_cumulative.append(index) 
    
    ### takes the population and randomly selects parents for the next generation. Uses normalised fitness totals to weight the "wheel" and increase the odds of fitter pops being selected to breed
    pop_size = len(population)
    for index2 in range(pop_size):
        rand_num = random.uniform(0,1)
        individual_num = 0
        for fitness_value in f_cumulative:
            if(rand_num <= fitness_value):
                parents.append(population[individual_num])
                break
            individual_num += 1
    return parents

### takes two parents and cuts them into a random length. Takes two opposing halves and combines them into a single child
def crossover(parents):
    offspring = []

    for i in range(0, len(parents), 2):
        cut_point = random.randint(1, len(parents[i]) - 1)
        offspring.append(parents[i][:cut_point] + parents[i+1][cut_point:])
        offspring.append(parents[i+1][:cut_point] + parents[i][cut_point:])
    return offspring

### takes a chromosome and randomly updates it using the mutation rate to determine the frequency of a bit being flipped
def mutate(chromosome, mutation_rate):
    for i in range (len(chromosome)):
        if random.random() < mutation_rate:
            chromosome = chromosome[:i] + [1-chromosome[i]] + chromosome[i + 1:]
    return chromosome

def final_fitness(population):
    fitness = []
    for i in range(pop_size):
        fitness.append(fitness_func(population[i]))
    x = numpy.argmax(fitness)
    return population[x]

###--MAIN--###

pop_size = 100
chromosome_count = 32
mutation_rate = 0.001
generation_count = 500

population = initial_population(pop_size, chromosome_count) 

### loop
k = 1
while k <= generation_count:
    fitness = []
    for i in range(pop_size):
        fitness.append(fitness_func(population[i]))
    
    parents = []
    parents = roulette_wheel(fitness, population)
    
    population = crossover(parents)
    
    for j in range(len(population)):
        population[j] = mutate(population[j], mutation_rate)

    k += 1

print("best:")
print(final_fitness(population))