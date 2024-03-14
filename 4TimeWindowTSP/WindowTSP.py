import numpy as np
import random
import csv
import matplotlib.pyplot as plt
import pandas as pd


# 计算两点之间的距离
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 计算时间窗口违规值
def calculate_time_window_violations(route, time_windows, arrival_times):
    violations = 0
    for idx, customer in enumerate(route):
        ready_time, due_time = time_windows[customer]
        arrival_time = arrival_times[idx]
        if arrival_time < ready_time:
            violations += ready_time - arrival_time
        elif arrival_time > due_time:
            violations += arrival_time - due_time
    return violations


# 计算总利润
def total_profit(route, profits):
    return sum(profits[customer] for customer in route)


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
# def select_parents(population, fitness, num_parents):
#     parents = []
#     for _ in range(num_parents):
#         max_fitness_idx = np.argmin(fitness)
#         parents.append(population[max_fitness_idx])
#         fitness[max_fitness_idx] = 99999999999
#     return parents
def select_parents(population, fitness, num_parents):
    # Normalize fitness values to probabilities
    # Note: Since we are minimizing distance, we invert the fitness values
    # to work with probabilities (higher is better).
    inverted_fitness = 1 / fitness
    total_fit = np.sum(inverted_fitness)
    probabilities = inverted_fitness / total_fit

    # Select parents using roulette wheel selection
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

        # 假定 'fix' 是一个函数，用于修复后代（例如，在TSP中修复重复的城市）
        fix(child)
        # 将修复后的后代添加到后代列表中
        offspring.append(child)
    return offspring  # 返回生成的后代列表


# 去除重复的元素并添加遗漏的元素
def fix(child):
    missing = set(range(len(child))) - set(child)
    duplicates = set([x for x in child if child.count(x) > 1])
    for d in duplicates:
        i = child.index(d)
        child[i] = missing.pop()


# 变异
def mutate(offspring, mutation_rate):
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


# 加载客户点及相关数据
def load_data(filename):
    points = []
    time_windows = []
    profits = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # 跳过表头
        for row in csvreader:
            if row:  # 确保行不是空的
                x, y = float(row[1]), float(row[2])  # XCOORD, YCOORD
                profit = float(row[3])  # PROFIT
                ready_time, due_time = float(row[4]), float(row[5])  # READYTIME, DUETIME
                points.append((x, y))
                time_windows.append((ready_time, due_time))
                profits.append(profit)
    return points, time_windows, profits


# 其他函数（create_initial_population, select_parents, crossover, fix, mutate）保持不变

# 主函数
def genetic_algorithm(filename, pop_size, num_generations, mutation_rate, crossover_rate):
    points, time_windows, profits = load_data(filename)
    num_points = len(points)
    population = create_initial_population(pop_size, num_points)
    best_fitness = float('inf')
    best_route = None

    best_fitnesses = []
    for generation in range(num_generations):
        fitness_scores = []
        for individual in population:
            route_distance = total_distance(individual, points)
            route_profit = total_profit(individual, profits)
            # 计算到达时间
            arrival_times = [0]  # 假设从仓库出发时间为0
            for i in range(1, len(individual)):
                arrival_times.append(
                    arrival_times[i - 1] + calculate_distance(points[individual[i - 1]], points[individual[i]])
                )
            time_violations = calculate_time_window_violations(individual, time_windows, arrival_times)
            # 适应度评分：距离要最小，利润最大，时间违规值最小
            fitness = route_distance - route_profit + time_violations
            fitness_scores.append(fitness)
            # 保存最佳个体
            if fitness < best_fitness:
                best_fitness = fitness
                best_route = individual

        print(f"Generation {generation} - Best Fitness: {best_fitness}")
        best_fitnesses.append(best_fitness)
        # 选择、交叉和变异
        fitness = np.array(fitness_scores)
        parents = select_parents(population, fitness, num_parents=pop_size // 2)
        offspring = crossover(parents, offspring_size=pop_size - len(parents), crossover_rate=crossover_rate)
        offspring = mutate(offspring, mutation_rate)
        population[0:len(parents)] = parents
        population[len(parents):] = offspring

    return best_route, best_fitness, best_fitnesses


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
    # 遗传算法参数
    filename = '../TSP.csv'  # CSV文件的路径
    # pop_size = 100  # 种群大小
    # num_generations = 500  # 代数
    # mutation_rate = 0.01  # 变异率
    # crossover_rate = 0.95
    #
    # # 运行遗传算法
    # best_route, best_fitness, fitness_scores = genetic_algorithm(filename, pop_size, num_generations, mutation_rate, crossover_rate)
    # print(f"The best route found is: {best_route}")
    # print(f"With a fitness score of: {best_fitness}")

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
