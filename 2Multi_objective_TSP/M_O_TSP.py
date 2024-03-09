import numpy as np
import csv
import random


# 距离计算函数，计算总旅程距离
def total_distance(route, points):
    distance = 0
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        distance += np.sqrt(
            (points[from_city][0] - points[to_city][0]) ** 2 + (points[from_city][1] - points[to_city][1]) ** 2)
    return distance


# 加载数据，包括点的位置和每个客户的利润
def load_data(filename):
    points = []
    profits = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 跳过头部
        for row in csvreader:
            x, y = float(row[1]), float(row[2])  # XCOORD YCOORD
            profits.append(float(row[3]))   # PROFIT
    return points, profits


# 初始化种群
def create_initial_population(pop_size, num_points):
    return [random.sample(range(num_points), num_points) for _ in range(pop_size)]


# 适应度函数
def calculate_fitness(route, points, profits, distance_weight=1.0, profit_weight=1.0):
    # 计算总距离
    total_dist = total_distance(route, points)
    # 计算总利润，减去顺序的影响
    total_prof = sum(profits[i] - (idx + 1) for idx, i in enumerate(route))
    # 计算加权适应度
    fitness = (distance_weight * total_dist) + (profit_weight * total_prof)
    return fitness


# 选择函数，这里简单用轮盘赌选择
def select(population, fitnesses):
    total_fitness = sum(fit for fit in fitnesses)
    rel_fitness = [fit / total_fitness for fit in fitnesses]
    probs = [sum(rel_fitness[:i + 1]) for i in range(len(rel_fitness))]
    new_population = []
    for _ in population:
        r = random.random()
        for (i, individual) in enumerate(population):
            if r <= probs[i]:
                new_population.append(individual)
                break
    return new_population


# 交叉函数，这里使用顺序交叉
def crossover(parent1, parent2):
    size = min(len(parent1), len(parent2))
    idx1, idx2 = sorted(random.sample(range(size), 2))
    child1 = parent1[idx1:idx2] + [item for item in parent2 if item not in parent1[idx1:idx2]]
    child2 = parent2[idx1:idx2] + [item for item in parent1 if item not in parent2[idx1:idx2]]
    return child1, child2


# 变异函数，这里使用交换变异
def mutate(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


# 遗传算法主要函数
def genetic_algorithm(filename, pop_size, num_generations, mutation_rate):
    points, profits = load_data(filename)
    population = create_initial_population(pop_size, len(points))

    for _ in range(num_generations):
        fitnesses = [calculate_fitness(individual, points, profits) for individual in population]
        population = select(population, fitnesses)
        next_generation = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = population[i], population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([mutate(child1), mutate(child2)])
        population = next_generation

    '''
    # 假设 `evaluate` 函数返回的是一个元组 (distance, profit)
    final_fitnesses = [evaluate(individual, points, profits) for individual in population]
    
    # 确保所有的适应度都是元组形式
    assert all(isinstance(fitness, tuple) and len(fitness) == 2 for fitness in final_fitnesses)
    
    # 通过列表推导式计算每个个体的适应度分数，并找到最大值的索引
    best_individual_index = np.argmax([profit - distance for distance, profit in final_fitnesses])
    '''
    final_fitnesses = [calculate_fitness(individual, points, profits) for individual in population]
    best_individual_index = np.argmax([profit - distance for distance, profit in final_fitnesses])
    best_individual = population[best_individual_index]
    return best_individual, final_fitnesses[best_individual_index]


# 主程序
if __name__ == "__main__":
    # 遗传算法参数
    filename = '../TSP.csv'  # CSV文件的路径
    pop_size = 100  # 种群大小
    num_generations = 500  # 代数
    mutation_rate = 0.01  # 变异率

    # 运行遗传算法
    genetic_algorithm(filename, pop_size, num_generations, mutation_rate)
