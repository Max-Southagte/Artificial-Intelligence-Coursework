import random
import numpy as np
import pandas as pd

###--DATA--###
# Read the CSV file containing diamond data
data = pd.read_csv('diamonds.csv')

# Define the features and target (price)
features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
target = 'price'

# Preprocess categorical features using one-hot encoding
data = pd.get_dummies(data, columns=['cut', 'color', 'clarity'])

# Normalize numerical features (carat, depth, table, x, y, z)
data[['carat', 'depth', 'table', 'x', 'y', 'z']] = data[['carat', 'depth', 'table', 'x', 'y', 'z']].apply(lambda x: (x - x.mean()) / x.std())

# Extract feature values and target prices
X = data.drop(columns=[target])
y = data[target]

# Number of features in the dataset
feature_count = X.shape[1]

# Genetic algorithm parameters
pop_size = 50
mutation_rate = 0.3
generation_count = 5 #changed from 200 - too slow
chromosome_count = feature_count  # One weight per feature

###--FUNCTIONS--###

# Generate the initial population (weights for the linear model)
def initial_population(pop_size, chromosome_count):
    population = []
    for _ in range(pop_size):
        population.append([random.uniform(-1, 1) for _ in range(chromosome_count)])  # Random weights between -1 and 1
    return population

# Fitness function to evaluate how well a set of weights predicts the price
def fitness_func(individual):
    predicted_prices = X.dot(individual)  # Linear model: predicted_price = X * weights
    error = np.mean((predicted_prices - y) ** 2)  # Mean squared error (MSE)
    return -error  # We minimize error, so we return the negative of the error for maximization

# Roulette wheel selection to pick parents based on fitness
def roulette_wheel(fitness, population):
    parents = []
    fitness_total = sum(fitness)
    normalized = [x / fitness_total for x in fitness]
    
    # Cumulative fitness
    cumulative_fitness = np.cumsum(normalized)
    
    # Select parents based on fitness
    for _ in range(len(population)):
        rand_num = random.uniform(0, 1)
        for idx, value in enumerate(cumulative_fitness):
            if rand_num <= value:
                parents.append(population[idx])
                break
    return parents

# Crossover between two parents to create offspring
def crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        cut_point = random.randint(1, len(parents[i]) - 1)
        offspring.append(parents[i][:cut_point] + parents[i + 1][cut_point:])
        offspring.append(parents[i + 1][:cut_point] + parents[i][cut_point:])
    return offspring

# Mutate a chromosome (set of weights)
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.uniform(-1, 1)  # Randomly change the weight
    return chromosome

# Get the best solution (individual) from the population based on fitness
def final_fitness(population):
    fitness = [fitness_func(ind) for ind in population]
    best_idx = np.argmax(fitness)
    return population[best_idx], fitness[best_idx]

# Genetic algorithm to evolve the weights (linear regression model)
def genetic_algorithm():
    population = initial_population(pop_size, chromosome_count)

    for generation in range(1, generation_count + 1):
        # Calculate fitness for each individual
        fitness = [fitness_func(ind) for ind in population]
        
        # Select parents based on fitness using roulette wheel
        parents = roulette_wheel(fitness, population)
        
        # Perform crossover to create offspring
        population = crossover(parents)
        
        # Apply mutation to the population
        for i in range(len(population)):
            population[i] = mutate(population[i], mutation_rate)

        # Get the best solution in the current generation
        best_solution, best_fitness_value = final_fitness(population)
        print(f"Generation {generation}: Best fitness (negative MSE) = {best_fitness_value}")

    return best_solution, -best_fitness_value  # Return the best weights and the minimized error

###--MAIN--###

best_solution, best_fitness_value = genetic_algorithm()
print("Final best solution (weights):", best_solution)
print("Final best fitness (negative MSE):", best_fitness_value)

# You can use the best solution (weights) to predict the price of a new diamond
# Assuming the new diamond data is provided as a dictionary or list
new_diamond = {
    'carat': 0.23,
    'cut': 'Ideal',  # Categorical variable
    'color': 'E',  # Categorical variable
    'clarity': 'SI2',  # Categorical variable
    'depth': 61.5,
    'table': 55,
    'x': 3.95,
    'y': 3.98,
    'z': 2.43
}

# 1. One-hot encode the categorical features
# Create a DataFrame from the new diamond and perform the same encoding as before
new_diamond_df = pd.DataFrame([new_diamond])

# One-hot encode categorical variables ('cut', 'color', 'clarity')
new_diamond_encoded = pd.get_dummies(new_diamond_df, columns=['cut', 'color', 'clarity'])

# 2. Normalize the numerical features using the same mean and std from the training data
# Extract the numerical features to normalize
numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
for col in numerical_columns:
    new_diamond_encoded[col] = (new_diamond_encoded[col] - data[col].mean()) / data[col].std()

# 3. Ensure the new data has the same number of columns as the training data
# If there are columns that were created by one-hot encoding, they should match
# For example, if 'cut' had 3 categories in the training data, it should also have 3 columns in the new data


# Align new_diamond_encoded with the columns in X
for col in X.columns:
    if col not in new_diamond_encoded.columns:
        new_diamond_encoded[col] = 0  # Add the missing column with a default value of 0

# Reorder columns to match the training data's feature columns
new_diamond_features = new_diamond_encoded[X.columns]



new_diamond_features = new_diamond_encoded[X.columns]

# 4. Predict the price using the evolved weights
predicted_price = new_diamond_features.dot(best_solution)
print(f"Predicted price for the new diamond: {predicted_price.iloc[0]}")
