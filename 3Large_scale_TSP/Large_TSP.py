import numpy as np
import random
from sklearn.cluster import KMeans
import csv
import pandas as pd


# 加载客户点
def load_data(filename):
    points = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # 跳过表头
        for row in csvreader:
            if row:  # 确保行不是空的
                x, y = float(row[1]), float(row[2])  # XCOORD YCOORD
                points.append((x, y))
    return points


# 计算两点之间距离的函数
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 计算路径总距离的函数
def total_distance(route, points):
    distance = calculate_distance(points[route[-1]], points[route[0]])
    for i in range(len(route) - 1):
        distance += calculate_distance(points[route[i]], points[route[i + 1]])
    return distance


def total_distance_1(route):
    # 计算路径总距离
    # distance = calculate_distance(route[-1], route[0])
    distance = 0
    for i in range(len(route) - 1):
        distance += calculate_distance(route[i], route[i + 1])
    return distance


# K-means 聚类函数
def cluster_points(points, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(points)
    clusters = kmeans.predict(points)
    clustered_points = {}
    for i, cluster in enumerate(clusters):
        if cluster not in clustered_points:
            clustered_points[cluster] = []
        clustered_points[cluster].append(points[i])  # 添加点的坐标
    return clustered_points, kmeans.cluster_centers_


# 遗传算法中使用的函数和遗传操作
def create_initial_population(pop_size, num_points):
    return [random.sample(range(num_points), num_points) for _ in range(pop_size)]


def select_parents(population, fitness, num_parents):
    parents = np.argsort(fitness)[:num_parents]
    return [population[i] for i in parents]


def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(parents, 2)
        idx1, idx2 = sorted(random.sample(range(len(parent1)), 2))
        child = parent1[idx1:idx2] + [item for item in parent2 if item not in parent1[idx1:idx2]]
        offspring.append(child)
    return offspring


def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual


# 遗传算法函数
def genetic_algorithm(clustered_points, pop_size, num_generations, mutation_rate,crossover_rates):
    best_solution_per_cluster = {}
    for cluster_index, points_indices in clustered_points.items():
        population = create_initial_population(pop_size, len(points_indices))
        best_distance = float('inf')
        best_solution = None

        for generation in range(num_generations):
            # 计算种群中每个个体的适应度
            fitness = np.array([total_distance(individual, points_indices) for individual in population])

            # 选择最优个体
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_distance:
                best_distance = fitness[best_idx]
                best_solution = population[best_idx]

            # 选择、交叉和变异
            parents = select_parents(population, fitness, num_parents=pop_size // 2)
            offspring = crossover(parents, offspring_size=pop_size - len(parents))
            offspring = [mutate(child, mutation_rate) for child in offspring]

            # 新一代种群
            population[:len(parents)] = parents
            population[len(parents):] = offspring

        best_solution_per_cluster[cluster_index] = [points_indices[i] for i in best_solution]
    return best_solution_per_cluster


def _________________________________():
    return


def cluster_create_route(representatives):
    route = list(representatives.keys())  # 获取所有cluster编号的列表
    random.shuffle(route)  # 随机打乱列表，生成一条随机路线
    return route


def cluster_create_initial_population(pop_size, representatives):
    return [cluster_create_route(representatives) for _ in range(pop_size)]


def cluster_rank_routes(population, representatives):
    fitness_results = {}
    for i, individual in enumerate(population):
        # 计算拼接后的点的列表的总距离的倒数作为适应度
        fitness_results[i] = 1 / float(total_distance_cluster_new(individual, representatives))
    # 根据适应度得分排序，适应度得分越高（总距离的倒数越大），排名越前
    return sorted(fitness_results.items(), key=lambda x: x[1], reverse=True)


def total_distance_cluster_new(route, representatives):
    # 根据 route 中指定的顺序，拼接所有点的列表
    all_points = []
    for cluster_number in route:
        # all_points.extend([representatives[cluster_number][0], representatives[cluster_number][-1]])
        all_points.extend(representatives[cluster_number])
    # 计算拼接后的列表的总距离
    return total_distance_1(all_points)


def cluster_crossover(parent1, parent2):
    # 实现顺序交叉（OX）或部分映射交叉（PMX）等交叉策略
    # 这里我们提供一个简化的单点交叉实现
    cross_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cross_point] + [x for x in parent2 if x not in parent1[:cross_point]]
    child2 = parent2[:cross_point] + [x for x in parent1 if x not in parent2[:cross_point]]
    return child1, child2


def cluster_crossover_population(mating_pool, pop_size):
    children = []
    while len(children) < pop_size:
        parent1 = random.choice(mating_pool)
        parent2 = random.choice(mating_pool)
        child1, child2 = cluster_crossover(parent1, parent2)
        children.extend([child1, child2])
    return children[:pop_size]


def cluster_mutate(individual, mutation_rate):
    # 实现交换变异
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = random.randint(0, len(individual) - 1)
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual


def cluster_mutate_population(population, mutation_rate):
    mutated_pop = [cluster_mutate(individual, mutation_rate) for individual in population]
    return mutated_pop


def connect_clusters(clustered_points, best_order):
    # 连接聚类中的点，根据best_order来确定访问代表点的顺序
    final_path = []
    for index in best_order:
        final_path.extend(clustered_points[index])
    return final_path


def total_distance_cluster(route, representatives):
    # 计算路径的总距离
    distance = 0
    for i in range(len(route)):
        from_representative = representatives[route[i]]
        to_representative = representatives[route[(i + 1) % len(route)]]
        distance += np.linalg.norm(np.array(from_representative) - np.array(to_representative))
    return distance


def cluster_genetic_algorithm(representatives, pop_size, num_generations, mutation_rate, crossover_rates):
    pop = cluster_create_initial_population(pop_size, representatives)
    print("Initial distance: " + str(1 / cluster_rank_routes(pop, representatives)[0][1]))

    for i in range(num_generations):
        ranked_pop = cluster_rank_routes(pop, representatives)
        selection_results = [x[0] for x in ranked_pop]
        matingpool = [pop[x] for x in selection_results[:5]]  # Selecting the top routes
        children = cluster_crossover_population(matingpool, pop_size)
        pop = cluster_mutate_population(children, mutation_rate)

        if (i % 100 == 0):
            current_best = 1 / cluster_rank_routes(pop, representatives)[0][1]
            print("Generation " + str(i) + " | Distance: " + str(current_best))

    print("Final distance: " + str(1 / cluster_rank_routes(pop, representatives)[0][1]))
    best_route_index = cluster_rank_routes(pop, representatives)[0][0]
    best_route = pop[best_route_index]
    return best_route


# 主函数
def run_cluster_genetic_algorithm(num_clusters=25,
         pop_size=100,
         num_generations=500,  # 代数
         mutation_rate=0.01,
         crossover_rate=0.95):  # 变异率):

    # 加载数据
    points = load_data('./TSP_large.csv')
    clustered_points, _ = cluster_points(points, num_clusters)

    # 应用遗传算法
    best_solution_per_cluster = genetic_algorithm(clustered_points, pop_size, num_generations, mutation_rate, crossover_rate)

    # 打印每个聚类的最佳路径和距离
    for cluster_index, route in best_solution_per_cluster.items():
        print(f"Cluster {cluster_index}: Best Path: {route}")
        print(f"Cluster {cluster_index}: Distance: {total_distance_1(route)}")

    # ————————————————————————————————————————

    # 使用遗传算法解决代表点之间的TSP问题
    best_order = cluster_genetic_algorithm(best_solution_per_cluster, pop_size=pop_size, num_generations=num_generations,
                                           mutation_rate=mutation_rate, crossover_rates=crossover_rate)

    final_path = connect_clusters(clustered_points, best_order)

    # 打印最终路径和总距离
    print("Final path:", final_path)

    final_distance = total_distance_1(final_path)

    return final_path, final_distance


if __name__ == "__main__":
    # main()
    filename = '../TSP.csv'  # CSV文件的路径

    # 待测试的参数范围
    pop_sizes = [100, 500, 1000]
    num_generationss = [100, 300, 500]
    num_clusters = [5, 15, 25]

    # 用于存储测试结果的列表
    results = []

    # 对每个参数进行测试
    for pop_size in pop_sizes:
        for num_generations in num_generationss:
            for num_cluster in num_clusters:

                    print(f"Testing with pop_size: {pop_size}, "
                          f"num_generations: {num_generations}, ")


                    # 运行算法并获取结果
                    best_route, best_distance = run_cluster_genetic_algorithm(
                        num_clusters=num_cluster, pop_size=pop_size, num_generations=num_generations)

                    # 将结果添加到列表中
                    results.append({
                        'pop_size': pop_size,
                        'num_generations': num_generations,
                        'best_distance': best_distance
                    })

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)

    # 将DataFrame保存为CSV文件
    results_df.to_csv('tsp_results.csv', index=False)
