import random
import numpy as np
import random
from itertools import permutations
import csv
import matplotlib.pyplot as plt
import pandas as pd


def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


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


# 创建初始种群
def create_initial_population(pop_size, num_points):
    population = []
    for _ in range(pop_size):
        individual = list(range(num_points))
        random.shuffle(individual)
        population.append(individual)
    return population


def total_distance(individual, points):
    distance = 0
    for i in range(len(individual)):
        start_point = points[individual[i]]
        end_point = points[individual[(i + 1) % len(individual)]]
        distance += calculate_distance(start_point, end_point)
    return distance


# 对于个体路径中的每个客户，它从profits列表中获取相应的利润值，并减去客户被访问的顺序编号（i）
def calculate_profit(individual, profits):
    total_profit = 0
    for i, city_index in enumerate(individual):
        total_profit += profits[city_index] - i
    return total_profit


# 计算Fitness
def cal_fitness(individual, points, profits):
    return calculate_profit(individual, profits) - total_distance(individual, points)
# def cal_fitness(individual, points, profits, alpha=0.5):
#     # 计算总利润
#     profit = calculate_profit(individual, profits)
#     # 计算总距离
#     distance = total_distance(individual, points)
#     # 标准化利润和距离
#     normalized_profit = (profit - min_profit) / (max_profit - min_profit)
#     normalized_distance = (distance - min_distance) / (max_distance - min_distance)
#     # 计算加权适应度
#     fitness = alpha * normalized_profit - (1 - alpha) * normalized_distance
#     return fitness


def select_parents(population, fitness, num_parents):
    inverted_fitness = 1 / fitness
    total_fit = np.sum(inverted_fitness)
    probabilities = inverted_fitness / total_fit

    # 轮盘赌
    parents_indices = np.random.choice(range(len(population)), size=num_parents, p=probabilities, replace=False)
    parents = [population[idx] for idx in parents_indices]

    return parents


# 交叉
def crossover(parents, offspring_size, crossover_rate):
    offspring = []  # 存放生成的后代
    for k in range(offspring_size):
        # 决定是否进行交叉
        if random.random() < crossover_rate:
            while True:
                # 从父代中随机选择两个不同的个体
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                if parent1 != parent2:
                    break  # 确保两个父代不是同一个个体
            # 随机选择一个交叉点
            cross_point = random.randint(0, len(parent1) - 1)
            # 创建新的后代，前半部分来自parent1，后半部分来自parent2
            child = parent1[:cross_point] + parent2[cross_point:]
        else:
            # 如果不进行交叉，则随机选择一个父代复制
            child = random.choice(parents).copy()

        fix(child)
        # 将修复后的后代添加到后代列表中
        offspring.append(child)
    return offspring  # 返回生成的后代列表


# Mutation
def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual


# 去除重复的元素并添加遗漏的元素
def fix(child):
    missing = set(range(len(child))) - set(child)
    duplicates = set([x for x in child if child.count(x) > 1])
    for d in duplicates:
        i = child.index(d)
        child[i] = missing.pop()


# 修改后的遗传算法
def genetic_algorithm(filename, pop_size, num_generations, mutation_rate, crossover_rate):
    # 加载数据，包括点和利润
    points, profits = load_data(filename)
    num_points = len(points)
    population = create_initial_population(pop_size, num_points)
    best_fitness = float('-inf')  # 由于我们希望最大化利润，因此初始值为负无穷
    best_individual = None
    best_fitnesses = []  # 用于存储每一代的最佳适应度

    for generation in range(num_generations):
        # 计算适应度，这里的适应度是基于距离和利润的
        fitness = np.array([cal_fitness(individual, points, profits) for individual in population])
        # 找到最佳个体
        best_idx = np.argmax(fitness)
        current_best_fitness = fitness[best_idx]
        # 更新全局最佳解
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[best_idx]
        # 将当前最佳适应度添加到列表中
        best_fitnesses.append(current_best_fitness)

        print(f"Generation {generation} - Best Fitness: {current_best_fitness}")

        # 选择、交叉和变异过程
        parents = select_parents(population, fitness, num_parents=pop_size // 2)
        offspring = crossover(parents, offspring_size=pop_size - len(parents), crossover_rate=crossover_rate)
        offspring = mutate(offspring, mutation_rate)
        # 更新种群
        population[0:len(parents)] = parents
        population[len(parents):] = offspring

    return best_individual, best_fitness, best_fitnesses  # 返回最佳个体、最佳适应度和每代最佳适应度的列表


def run_genetic_algorithm(filename, pop_size=100, num_generations=500, mutation_rate=0.01, crossover_rate=0.95):
    best_route, best_distance, best_distances = genetic_algorithm(filename, pop_size, num_generations, mutation_rate,
                                                                  crossover_rate)
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

    # 待测试的参数范围
    # pop_sizes = [100]
    # num_generationss = [100]
    # mutation_rates = [0.005]
    # crossover_rates = [0.7]

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
                        'best_fitness': best_distance
                    })

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)

    # 将DataFrame保存为CSV文件
    results_df.to_csv('tsp_results.csv', index=False)
