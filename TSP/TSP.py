import numpy as np
import random
from itertools import permutations
import csv


# 计算两点之间的距离
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 计算路线的总距离
def total_distance(route, points):
    distance = 0
    for i in range(len(route)):
        distance += calculate_distance(points[route[i - 1]], points[route[i]])
    return distance


# 创建初始种群
def create_initial_population(pop_size, num_points):
    population = []
    for _ in range(pop_size):
        individual = list(range(num_points))
        random.shuffle(individual)
        population.append(individual)
    return population


# 选择
def select_parents(population, fitness, num_parents):
    parents = []
    for _ in range(num_parents):
        max_fitness_idx = np.argmin(fitness)
        parents.append(population[max_fitness_idx])
        fitness[max_fitness_idx] = 99999999999
    return parents


# 交叉
def crossover(parents, offspring_size):
    offspring = []
    for k in range(offspring_size):
        while True:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            if parent1 != parent2:
                break
        cross_point = random.randint(0, len(parent1) - 1)
        child = parent1[:cross_point] + parent2[cross_point:]
        fix(child)
        offspring.append(child)
    return offspring


# 去除重复的元素并添加遗漏的元素
def fix(child):
    missing = set(range(len(child))) - set(child)
    duplicates = set([x for x in child if child.count(x) > 1])
    for d in duplicates:
        i = child.index(d)
        child[i] = missing.pop()


# 变异
def mutate(offspring):
    for i in range(len(offspring)):
        if random.uniform(0, 1) < mutation_rate:
            swapped = False
            while not swapped:
                geneA = int(random.uniform(0, len(offspring[i])))
                geneB = int(random.uniform(0, len(offspring[i])))
                if geneA != geneB:
                    offspring[i][geneA], offspring[i][geneB] = offspring[i][geneB], offspring[i][geneA]
                    swapped = True
    return offspring


# 加载客户点
def load_data(filename):
    points = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # 跳过表头
        for row in csvreader:
            if row:  # 确保行不是空的
                x, y = float(row[0]), float(row[1])  # 假设XCOORD.是第一列，YCOORD.是第二列
                points.append((x, y))
    return points


# 主函数
def genetic_algorithm(filename, pop_size, num_generations, mutation_rate):
    points = load_data(filename)
    num_points = len(points)
    population = create_initial_population(pop_size, num_points)
    best_distance = float('inf')
    best_route = None

    for generation in range(num_generations):
        fitness = np.array([total_distance(individual, points) for individual in population])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_distance:
            best_distance = fitness[best_idx]
            best_route = population[best_idx]

        print(f"Generation {generation} - Best Distance: {best_distance}")

        parents = select_parents(population, fitness, num_parents=pop_size // 2)
        offspring = crossover(parents, offspring_size=pop_size - len(parents))
        offspring = mutate(offspring)
        population[0:len(parents)] = parents
        population[len(parents):] = offspring

    return best_route, best_distance


# 遗传算法参数
filename = '../TSP.csv'
pop_size = 100
num_generations = 500
mutation_rate = 0.01

# 运行遗传算法
best_route, best_distance = genetic_algorithm(filename, pop_size, num_generations, mutation_rate)
print(f"The best route found is: {best_route}")
print(f"With a total distance of: {best_distance}")
