import numpy as np
import random
import csv


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
def genetic_algorithm(filename, pop_size, num_generations, mutation_rate):
    points, time_windows, profits = load_data(filename)
    num_points = len(points)
    population = create_initial_population(pop_size, num_points)
    best_fitness = float('inf')
    best_route = None

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

        # 选择、交叉和变异
        fitness = np.array(fitness_scores)
        parents = select_parents(population, fitness, num_parents=pop_size // 2)
        offspring = crossover(parents, offspring_size=pop_size - len(parents))
        offspring = mutate(offspring)
        population[0:len(parents)] = parents
        population[len(parents):] = offspring

    return best_route, best_fitness


# 主程序入口点
if __name__ == "__main__":
    # 遗传算法参数
    filename = '../TSP.csv'  # CSV文件的路径
    pop_size = 100  # 种群大小
    num_generations = 500  # 代数
    mutation_rate = 0.01  # 变异率

    # 运行遗传算法
    best_route, best_fitness = genetic_algorithm(filename, pop_size, num_generations, mutation_rate)
    print(f"The best route found is: {best_route}")
    print(f"With a fitness score of: {best_fitness}")
