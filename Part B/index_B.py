import random
import numpy

###--DATA--###
items = [
    {'weight': 3, 'profit': 126},
    {'weight': 8, 'profit': 154},
    {'weight': 2, 'profit': 256},
    {'weight': 9, 'profit': 526},
    {'weight': 7, 'profit': 388},
    {'weight': 1, 'profit': 245},
    {'weight': 8, 'profit': 210},
    {'weight': 13, 'profit': 442},
    {'weight': 10, 'profit': 671},
    {'weight': 9, 'profit': 348},
]

max_weight = 35
item_count = len(items)
pop_size = 50
chromosome_count = item_count  # The number of items (10)
mutation_rate = 0.05
generation_count = 200

###--FUNCTIONS--###

### generates a list of populations according to the input variables
def initial_population(pop_size, chromosome_count): 
    population = []
    for i in range(pop_size):
        population.append(random.choices(range(2), k=chromosome_count))
    return population

### fitness function to calculate profit, respecting the weight limit
def fitness_func(individual): 
    total_weight = 0
    total_profit = 0
    
    for i in range(item_count):
        if individual[i] == 1:  # If item is selected
            total_weight += items[i]['weight']
            total_profit += items[i]['profit']
    
    # Penalize if weight exceeds the limit
    if total_weight > max_weight:
        return 0  # Or you could apply a penalty, e.g., return total_profit - some penalty value
    
    return total_profit

def roulette_wheel(fitness, population):
    parents = []
    fitness_total = sum(fitness)
    normalised = [x / fitness_total for x in fitness]

    ### creates the "wheel" values by cumulatively adding all of the normalised values together, up to 1.0
    f_cumulative = []
    index = 0
    for n in normalised:
        index += n
        f_cumulative.append(index) 
    
    ### takes the population and randomly selects parents for the next generation
    pop_size = len(population)
    for index2 in range(pop_size):
        rand_num = random.uniform(0, 1)
        individual_num = 0
        for fitness_value in f_cumulative:
            if rand_num <= fitness_value:
                parents.append(population[individual_num])
                break
            individual_num += 1
    return parents

### takes two parents and cuts them into a random length
def crossover(parents):
    offspring = []
    
    for i in range(0, len(parents), 2):
        cut_point = random.randint(1, len(parents[i]) - 1)
        offspring.append(parents[i][:cut_point] + parents[i + 1][cut_point:])
        offspring.append(parents[i + 1][:cut_point] + parents[i][cut_point:])
    return offspring

### takes a chromosome and randomly updates it using the mutation rate to determine the frequency of a bit being flipped
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Flip the bit
    return chromosome

def final_fitness(population):
    fitness = []
    for i in range(pop_size):
        fitness.append(fitness_func(population[i]))
    x = numpy.argmax(fitness)
    return population[x], fitness[x]

def genetic_algorithm():
    population = initial_population(pop_size, chromosome_count)

    ### loop
    for generation in range(1, generation_count + 1):
        fitness = []
        for i in range(pop_size):
            fitness.append(fitness_func(population[i]))
        
        parents = roulette_wheel(fitness, population)
        
        population = crossover(parents)
        
        for j in range(len(population)):
            population[j] = mutate(population[j], mutation_rate)

        best_solution, best_fitness = final_fitness(population)
        print(f"Generation {generation}: Best solution (items selected) = {best_solution}, Best profit = {best_fitness}")

    return best_solution, best_fitness

###--MAIN--###

best_solution, best_fitness = genetic_algorithm()
print("Final best solution (items selected):", best_solution)
print("Final best profit:", best_fitness)
