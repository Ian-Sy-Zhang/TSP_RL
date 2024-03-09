from math import sqrt

import numpy as np
import random
import csv


# 计算 TSP 路线的总距离
def total_distance(path, points):
    # 使用欧几里得距离计算路径总长度
    return sum(
        sqrt((points[path[i - 1]][0] - points[path[i]][0]) ** 2 + (points[path[i - 1]][1] - points[path[i]][1]) ** 2)
        for i in range(len(path)))


# 创建初始种群
def create_initial_population(pop_size, num_points):
    # 为种群中的每个个体生成一个随机的城市访问顺序
    return [{'genome': random.sample(range(num_points), num_points)} for _ in range(pop_size)]


# 为个体分配技能因子
# 分配技能因子，确保每个任务有最低比例的个体
def assign_skill_factor(population, num_tasks, min_task_proportion):
    pop_size = len(population)
    min_task_size = [int(proportion * pop_size) for proportion in min_task_proportion]

    # 确保最低任务比例总和不超过 1
    assert sum(min_task_proportion) <= 1, 'The sum of min_task_proportion should not exceed 1.'

    # 首先分配每个任务的最低比例的个体
    assigned_counts = [0] * num_tasks
    for i in range(pop_size):
        for task_id in range(num_tasks):
            if assigned_counts[task_id] < min_task_size[task_id]:
                population[i]['skill_factor'] = task_id
                assigned_counts[task_id] += 1
                break

    # 分配剩余的个体
    for i in range(pop_size):
        if 'skill_factor' not in population[i]:
            task_id = random.choice([
                i for i in range(num_tasks) if
                assigned_counts[i] < (min_task_size[i] + (pop_size - sum(min_task_size)) / num_tasks)
            ])
            population[i]['skill_factor'] = task_id
            assigned_counts[task_id] += 1

    return population


# 评估个体适应度的函数
def evaluate_fitness(individual, tasks):
    # 根据个体的技能因子，计算其适应度（即完成其任务的路径总长度）
    skill_factor = individual['skill_factor']
    individual['fitness'] = total_distance(individual['genome'], tasks[skill_factor])


# 轮盘赌选择的函数
def roulette_wheel_selection(population):
    # 根据个体的适应度进行选择
    total_fitness = sum(ind['fitness'] for ind in population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind in population:
        current += ind['fitness']
        if current > pick:
            return ind


# 有序交叉的函数（Ordered Crossover）
def ordered_crossover(parent1, parent2):
    # 在两个父代个体之间进行有序交叉来生成子代
    size = len(parent1['genome'])
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1['genome'][start:end]

    child_p = end
    parent_p = end
    while -1 in child:
        if parent2['genome'][parent_p] not in child:
            if child_p >= size:
                child_p = 0
            child[child_p] = parent2['genome'][parent_p]
            child_p += 1
        parent_p = (parent_p + 1) % size

    return {'genome': child, 'skill_factor': parent1['skill_factor'], 'fitness': float('inf')}


# 变异函数（交换两个城市）
def swap_mutation(individual):
    # 在个体的基因组中随机选择两个城市并交换它们
    idx1, idx2 = random.sample(range(len(individual['genome'])), 2)
    individual['genome'][idx1], individual['genome'][idx2] = individual['genome'][idx2], individual['genome'][idx1]


# 加载TSP数据的函数（假设它是一个带有城市坐标的CSV文件）
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


# 多任务进化算法主函数
def mfea(tasks_filenames, pop_size, num_generations, mutation_rate, crossover_rate, min_task_proportion):
    tasks = [load_data(filename) for filename in tasks_filenames]
    num_tasks = len(tasks)
    population = create_initial_population(pop_size, len(tasks[0]))
    population = assign_skill_factor(population, num_tasks, min_task_proportion)

    for generation in range(num_generations):
        # 为种群中的每个个体评估适应度
        for individual in population:
            evaluate_fitness(individual, tasks)

        # 生成新的种群
        new_population = []
        for _ in range(pop_size):
            # 轮盘赌方式选择父本
            parent1 = roulette_wheel_selection(population)
            child = parent1.copy()

            # 若随机数小于交叉率，则进行交叉
            if random.random() < crossover_rate:
                # 再次使用轮盘赌方式选择另一个父本
                parent2 = roulette_wheel_selection(population)
                # 如果两个父本有相同的技能因子，则进行有序交叉
                if parent1['skill_factor'] == parent2['skill_factor']:
                    child = ordered_crossover(parent1, parent2)

            # 若随机数小于变异率，则进行变异
            if random.random() < mutation_rate:
                swap_mutation(child)

            # 评估新个体的适应度
            evaluate_fitness(child, tasks)
            # 将新个体添加到新种群中
            new_population.append(child)

        # 更新种群
        population = new_population

    # 提取每个任务的最优个体
    best_individuals = []
    for task_idx in range(num_tasks):
        # 筛选出对应技能因子的个体
        task_individuals = [ind for ind in population if ind['skill_factor'] == task_idx]
        # 找出每个任务的最优适应度个体
        if len(task_individuals) > 0:
            print(len(task_individuals))
            best_individuals.append(min(task_individuals, key=lambda ind: ind['fitness']))

    # 返回每个任务的最优个体
    return best_individuals


def main():
    # 定义算法参数
    filepath1 = 'TSP_large_1.csv'
    filepath2 = 'TSP_large.csv'
    pop_size = 1000
    num_generations = 12
    mutation_rate = 0.1
    crossover_rate = 0.2
    min_task_proportion = [0.2, 0.2]  # 两个任务的最低比例

    # 运行 MFEA 算法
    best_individuals = mfea(
        tasks_filenames=[filepath1, filepath2],
        pop_size=pop_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        min_task_proportion=min_task_proportion
    )
    # 打印每个任务的最佳路径和总距离
    for idx, individual in enumerate(best_individuals):
        print(f"Task {idx + 1} Best Practice: {individual['genome']} Total Distance: {individual['fitness']}")


if __name__ == '__main__':
    main()
