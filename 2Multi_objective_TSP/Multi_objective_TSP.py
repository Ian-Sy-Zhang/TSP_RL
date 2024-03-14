import random
import numpy as np
import random
from itertools import permutations
import csv
import matplotlib.pyplot as plt
import pandas as pd


# Calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


# Load data, including the location of each point and profit for each customer
def load_data(filename):
    points = []
    profits = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip header
        for row in csvreader:
            x, y = float(row[1]), float(row[2])
            profit = float(row[3])
            points.append((x, y))
            profits.append(profit)
    return points, profits


# Generate a random individual
def create_individual(points):
    individual = list(range(len(points)))
    random.shuffle(individual)
    return individual


# Create the initial population of individuals
def create_population(pop_size, points):
    return [create_individual(points) for _ in range(pop_size)]


# Calculate the total distance of the travel for an individual
def total_distance(individual, points):
    distance = 0
    for i in range(len(individual)):
        start_point = points[individual[i]]
        end_point = points[individual[(i + 1) % len(individual)]]
        distance += calculate_distance(start_point, end_point)
    return distance


# Calculate total profit with respect to order in the individual
def calculate_profit(individual, profits):
    total_profit = 0
    for i, city_index in enumerate(individual):
        total_profit += profits[city_index] - i
    return total_profit


# Fitness function
def fitness(individual, points, profits):
    return calculate_profit(individual, profits) - total_distance(individual, points)


# Selection
def select(population, points, profits):
    fitnesses = [fitness(individual, points, profits) for individual in population]
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return list(np.random.choice(population, size=len(population), p=selection_probs))


# Crossover
def crossover(parent1, parent2):
    size = len(parent1)
    crossover_point = random.randint(0, size - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Mutation
def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual


# Genetic algorithm
def genetic_algorithm(filename, pop_size, generations, mutation_rate):
    points, profits = load_data(filename)
    population = create_population(pop_size, points)

    fitnesses = []
    for generation in range(generations):
        population = select(population, points, profits)

        next_generation = []
        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1, mutation_rate))
            if len(next_generation) < pop_size:
                next_generation.append(mutate(child2, mutation_rate))

        population = next_generation
        best_individual = max(population, key=lambda ind: fitness(ind, points, profits))
        best_fitness = fitness(best_individual, points, profits)
        fitnesses.append(best_fitness)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_individual, best_fitness, fitnesses


def run_genetic_algorithm(filename, pop_size=100, num_generations=500, mutation_rate=0.01, crossover_rate=0.95):
    best_route, best_distance, best_distances = genetic_algorithm(filename, pop_size, num_generations, mutation_rate)
    print(f"The best route found is: {best_route}")
    print(f"With a total distance of: {best_distance}")

    plt.figure()
    # 绘制每代的最佳距离
    plt.plot(best_distances)
    plt.title('Best Distance by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    # plt.show()
    plt_name = 'pop_size:' + str(pop_size) + '_' + 'num_generations:' + str(
        num_generations) + '_' + 'mutation_rate:' + str(mutation_rate) + '_' + 'crossover_rate:' + str(
        crossover_rate) + '.png'
    plt.savefig(plt_name)

    return best_route, best_distance


# 主程序入口点
if __name__ == "__main__":
    filename = '../TSP.csv'  # CSV文件的路径

    # 待测试的参数范围
    pop_sizes = [100, 500, 1000]
    num_generationss = [100, 300, 500]
    mutation_rates = [0.005, 0.01, 0.02]
    crossover_rates = [0.7, 0.85, 0.95]

    # 用于存储测试结果的列表
    results = []

    # 对每个参数进行测试
    for pop_size in pop_sizes:
        for num_generations in num_generationss:
            for mutation_rate in mutation_rates:
                for crossover_rate in crossover_rates:
                    print(f"Testing with pop_size: {pop_size}, "
                          f"num_generations: {num_generations}, "
                          f"mutation_rate: {mutation_rate}, "
                          f"crossover_rate: {crossover_rate}")

                    # 运行算法并获取结果
                    best_route, best_distance = run_genetic_algorithm(
                        filename, pop_size, num_generations, mutation_rate, crossover_rate
                    )

                    # 将结果添加到列表中
                    results.append({
                        'pop_size': pop_size,
                        'num_generations': num_generations,
                        'mutation_rate': mutation_rate,
                        'crossover_rate': crossover_rate,
                        'best_distance': best_distance
                    })

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)

    # 将DataFrame保存为CSV文件
    results_df.to_csv('tsp_results.csv', index=False)
