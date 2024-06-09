import os
import re

# 定义文件路径和命令
libFile = os.path.abspath('./project/lib/7nm/7nm.lib')
logFile = 'circuit.log'

# 每个动作代码对应的Yosys合成操作
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

def calculate_eval(circuitName, actions):
    circuitPath = os.path.abspath(f'C:/CS3308/project/InitialAIG/train/{circuitName}.aig')
    action_cmd = ''
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = f'C:/msys64/mingw64/bin/yosys-abc -c "read {circuitPath}; {action_cmd} read_lib {libFile}; map; topo; stime" > {logFile}'
    os.system(abcRunCmd)
    with open(logFile) as f:
        lines = f.readlines()
        areaInformation = re.findall('[a-zA-Z0-9.-]+', lines[-1])
        eval = float(areaInformation[-6]) * float(areaInformation[-3])
    return eval

def find_best_action_for_position(circuitName, actions):
    best_eval = -float('inf')
    best_action = -1

    for action in range(7):
        next_actions = actions + str(action)
        current_eval = calculate_eval(circuitName, next_actions)
        if current_eval > best_eval:
            best_eval = current_eval
            best_action = action

    return best_action, best_eval

def find_best_sequence(initial_state, max_depth):
    circuitName, actions = initial_state.split('_')
    current_actions = ''

    for depth in range(max_depth):
        best_action, best_eval = find_best_action_for_position(circuitName, current_actions)
        current_actions += str(best_action)
        print(f"Depth {depth + 1} for {circuitName}: Best action so far is {best_action} with eval {best_eval}")

    return f'{circuitName}_{current_actions}', best_eval

circuit_names = [
    'adder', 'alu2', 'apex3', 'apex5', 'arbiter', 'b2', 'c1355', 'c2670', 'c5315', 'c6288',
    'ctrl', 'frg1', 'i7', 'i8', 'int2float', 'log2', 'm3', 'max', 'max512', 'multiplier',
    'priority', 'prom2', 'table5'
]
# circuit_names = [
#     'adder'
# ]

max_depth = 10

results = {}

for circuit_name in circuit_names:
    initial_state = f'{circuit_name}_'
    best_sequence, best_eval = find_best_sequence(initial_state, max_depth)
    results[circuit_name] = (best_sequence, best_eval)
    print(f"Best sequence for {circuit_name}: {best_sequence}")
    print(f"Best eval for {circuit_name}: {best_eval}")

# 将结果写入文件
output_file_path = 'C:/CS3308/output1.txt'
with open(output_file_path, 'w') as output_file:
    for circuit_name, (best_sequence, best_eval) in results.items():
        output_file.write(f"{circuit_name}: Best sequence = {best_sequence}, Best eval = {best_eval}\n")

print(f"Results have been written to {output_file_path}")



import os
import re
import random
import math

# 定义文件路径和命令
libFile = os.path.abspath('./project/lib/7nm/7nm.lib')
logFile = 'circuit.log'

# 每个动作代码对应的Yosys合成操作
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

def calculate_eval(circuitName, actions):
    circuitPath = os.path.abspath(f'C:/CS3308/project/InitialAIG/train/{circuitName}.aig')
    action_cmd = ''
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = f'C:/msys64/mingw64/bin/yosys-abc -c "read {circuitPath}; {action_cmd} read_lib {libFile}; map; topo; stime" > {logFile}'
    os.system(abcRunCmd)
    with open(logFile) as f:
        lines = f.readlines()
        areaInformation = re.findall('[a-zA-Z0-9.-]+', lines[-1])
        eval = float(areaInformation[-6]) * float(areaInformation[-3])
    return eval

def random_neighbor(actions):
    actions_list = list(actions)
    pos = random.randint(0, len(actions_list) - 1)
    new_action = str(random.randint(0, 6))
    actions_list[pos] = new_action
    return ''.join(actions_list)

def simulated_annealing(circuitName, initial_actions, max_steps, initial_temp, cooling_rate):
    current_actions = initial_actions
    current_eval = calculate_eval(circuitName, current_actions)
    best_actions = current_actions
    best_eval = current_eval
    temperature = initial_temp

    for step in range(max_steps):
        new_actions = random_neighbor(current_actions)
        new_eval = calculate_eval(circuitName, new_actions)
        if new_eval > best_eval:
            best_actions = new_actions
            best_eval = new_eval
        if new_eval > current_eval or random.uniform(0, 1) < math.exp((new_eval - current_eval) / temperature):
            current_actions = new_actions
            current_eval = new_eval
        temperature *= cooling_rate

        print(f"Step {step + 1} for {circuitName}, Temp {temperature:.2f}: Current best eval is {best_eval}")

    return best_actions, best_eval

circuit_names = [
    'adder', 'alu2', 'apex3', 'apex5', 'arbiter', 'b2', 'c1355', 'c2670', 'c5315', 'c6288',
    'ctrl', 'frg1', 'i7', 'i8', 'int2float', 'log2', 'm3', 'max', 'max512', 'multiplier',
    'priority', 'prom2', 'table5'
]

max_steps = 1000
initial_temp = 100.0
cooling_rate = 0.99

results = {}

for circuit_name in circuit_names:
    initial_state = '0' * 10  # 初始状态假设为 0000000000
    best_sequence, best_eval = simulated_annealing(circuit_name, initial_state, max_steps, initial_temp, cooling_rate)
    results[circuit_name] = (best_sequence, best_eval)
    print(f"Best sequence for {circuit_name}: {circuit_name}_{best_sequence}")
    print(f"Best eval for {circuit_name}: {best_eval}")

# 将结果写入文件
output_file_path = 'C:/CS3308/output2.txt'
with open(output_file_path, 'w') as output_file:
    for circuit_name, (best_sequence, best_eval) in results.items():
        output_file.write(f"{circuit_name}: Best sequence = {circuit_name}_{best_sequence}, Best eval = {best_eval}\n")

print(f"Results have been written to {output_file_path}")

import os
import re
import random

# 定义文件路径和命令
libFile = os.path.abspath('./project/lib/7nm/7nm.lib')
logFile = 'circuit.log'

# 每个动作代码对应的Yosys合成操作
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

def calculate_eval(circuitName, actions):
    circuitPath = os.path.abspath(f'C:/CS3308/project/InitialAIG/train/{circuitName}.aig')
    action_cmd = ''
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = f'C:/msys64/mingw64/bin/yosys-abc -c "read {circuitPath}; {action_cmd} read_lib {libFile}; map; topo; stime" > {logFile}'
    os.system(abcRunCmd)
    with open(logFile) as f:
        lines = f.readlines()
        areaInformation = re.findall('[a-zA-Z0-9.-]+', lines[-1])
        eval = float(areaInformation[-6]) * float(areaInformation[-3])
    return eval

def create_initial_population(size, length):
    population = []
    for _ in range(size):
        individual = ''.join(random.choice('0123456') for _ in range(length))
        population.append(individual)
    return population

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    parents = random.choices(population, weights=selection_probs, k=2)
    return parents

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, mutation_rate):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice('0123456')
    return ''.join(individual)

def genetic_algorithm(circuitName, population_size, sequence_length, generations, mutation_rate):
    population = create_initial_population(population_size, sequence_length)
    best_individual = None
    best_eval = -float('inf')

    for generation in range(generations):
        fitnesses = [calculate_eval(circuitName, individual) for individual in population]
        for i, fitness in enumerate(fitnesses):
            if fitness > best_eval:
                best_eval = fitness
                best_individual = population[i]

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = new_population
        print(f"Generation {generation + 1} for {circuitName}: Best eval is {best_eval}")

    return best_individual, best_eval

circuit_names = [
    'adder', 'alu2', 'apex3', 'apex5', 'arbiter', 'b2', 'c1355', 'c2670', 'c5315', 'c6288',
    'ctrl', 'frg1', 'i7', 'i8', 'int2float', 'log2', 'm3', 'max', 'max512', 'multiplier',
    'priority', 'prom2', 'table5'
]

population_size = 20
sequence_length = 10
generations = 50
mutation_rate = 0.25

results = {}

for circuit_name in circuit_names:
    best_sequence, best_eval = genetic_algorithm(circuit_name, population_size, sequence_length, generations, mutation_rate)
    results[circuit_name] = (best_sequence, best_eval)
    print(f"Best sequence for {circuit_name}: {circuit_name}_{best_sequence}")
    print(f"Best eval for {circuit_name}: {best_eval}")

# 将结果写入文件
output_file_path = 'C:/CS3308/output3.txt'
with open(output_file_path, 'w') as output_file:
    for circuit_name, (best_sequence, best_eval) in results.items():
        output_file.write(f"{circuit_name}: Best sequence = {circuit_name}_{best_sequence}, Best eval = {best_eval}\n")

print(f"Results have been written to {output_file_path}")

import os
import re

# 定义文件路径和命令
libFile = os.path.abspath('./project/lib/7nm/7nm.lib')
logFile = 'circuit.log'

# 每个动作代码对应的Yosys合成操作
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

def calculate_eval(circuitName, actions):
    circuitPath = os.path.abspath(f'C:/CS3308/project/InitialAIG/test/{circuitName}.aig')
    action_cmd = ''
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = f'C:/msys64/mingw64/bin/yosys-abc -c "read {circuitPath}; {action_cmd} read_lib {libFile}; map; topo; stime" > {logFile}'
    os.system(abcRunCmd)
    with open(logFile) as f:
        lines = f.readlines()
        areaInformation = re.findall('[a-zA-Z0-9.-]+', lines[-1])
        eval = float(areaInformation[-6]) * float(areaInformation[-3])
    return eval

def find_best_action_for_position(circuitName, actions):
    best_eval = -float('inf')
    best_action = -1

    for action in range(7):
        next_actions = actions + str(action)
        current_eval = calculate_eval(circuitName, next_actions)
        if current_eval > best_eval:
            best_eval = current_eval
            best_action = action

    return best_action, best_eval

def find_best_sequence(initial_state, max_depth):
    circuitName, actions = initial_state.split('_')
    current_actions = ''

    for depth in range(max_depth):
        best_action, best_eval = find_best_action_for_position(circuitName, current_actions)
        current_actions += str(best_action)
        print(f"Depth {depth + 1} for {circuitName}: Best action so far is {best_action} with eval {best_eval}")

    return f'{circuitName}_{current_actions}', best_eval

circuit_names = [
    'alu4', 'apex1', 'apex2', 'apex4', 'b9', 'bar', 'c880', 'c7552', 'cavlc', 'div', 
    'i9', 'm4', 'max1024', 'pair', 'prom1', 'router', 'sqrt', 'square', 'voter'
]

max_depth = 10

results = {}

for circuit_name in circuit_names:
    initial_state = f'{circuit_name}_'
    best_sequence, best_eval = find_best_sequence(initial_state, max_depth)
    results[circuit_name] = (best_sequence, best_eval)
    print(f"Best sequence for {circuit_name}: {best_sequence}")
    print(f"Best eval for {circuit_name}: {best_eval}")

# 将结果写入文件
output_file_path = 'C:/CS3308/output4.txt'
with open(output_file_path, 'w') as output_file:
    for circuit_name, (best_sequence, best_eval) in results.items():
        output_file.write(f"{circuit_name}: Best sequence = {best_sequence}, Best eval = {best_eval}\n")

print(f"Results have been written to {output_file_path}")
import os
import re
import random
import math

# 定义文件路径和命令
libFile = os.path.abspath('./project/lib/7nm/7nm.lib')
logFile = 'circuit.log'

# 每个动作代码对应的Yosys合成操作
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

def calculate_eval(circuitName, actions):
    circuitPath = os.path.abspath(f'C:/CS3308/project/InitialAIG/test/{circuitName}.aig')
    action_cmd = ''
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = f'C:/msys64/mingw64/bin/yosys-abc -c "read {circuitPath}; {action_cmd} read_lib {libFile}; map; topo; stime" > {logFile}'
    os.system(abcRunCmd)
    with open(logFile) as f:
        lines = f.readlines()
        areaInformation = re.findall('[a-zA-Z0-9.-]+', lines[-1])
        eval = float(areaInformation[-6]) * float(areaInformation[-3])
    return eval

def random_neighbor(actions):
    actions_list = list(actions)
    pos = random.randint(0, len(actions_list) - 1)
    new_action = str(random.randint(0, 6))
    actions_list[pos] = new_action
    return ''.join(actions_list)

def simulated_annealing(circuitName, initial_actions, max_steps, initial_temp, cooling_rate):
    current_actions = initial_actions
    current_eval = calculate_eval(circuitName, current_actions)
    best_actions = current_actions
    best_eval = current_eval
    temperature = initial_temp

    for step in range(max_steps):
        new_actions = random_neighbor(current_actions)
        new_eval = calculate_eval(circuitName, new_actions)
        if new_eval > best_eval:
            best_actions = new_actions
            best_eval = new_eval
        if new_eval > current_eval or random.uniform(0, 1) < math.exp((new_eval - current_eval) / temperature):
            current_actions = new_actions
            current_eval = new_eval
        temperature *= cooling_rate

        print(f"Step {step + 1} for {circuitName}, Temp {temperature:.2f}: Current best eval is {best_eval}")

    return best_actions, best_eval

circuit_names = [
    'alu4', 'apex1', 'apex2', 'apex4', 'b9', 'bar', 'c880', 'c7552', 'cavlc', 'div', 
    'i9', 'm4', 'max1024', 'pair', 'prom1', 'router', 'sqrt', 'square', 'voter'
]

max_steps = 1000
initial_temp = 100.0
cooling_rate = 0.99

results = {}

for circuit_name in circuit_names:
    initial_state = '0' * 10  # 初始状态假设为 0000000000
    best_sequence, best_eval = simulated_annealing(circuit_name, initial_state, max_steps, initial_temp, cooling_rate)
    results[circuit_name] = (best_sequence, best_eval)
    print(f"Best sequence for {circuit_name}: {circuit_name}_{best_sequence}")
    print(f"Best eval for {circuit_name}: {best_eval}")

# 将结果写入文件
output_file_path = 'C:/CS3308/output5.txt'
with open(output_file_path, 'w') as output_file:
    for circuit_name, (best_sequence, best_eval) in results.items():
        output_file.write(f"{circuit_name}: Best sequence = {circuit_name}_{best_sequence}, Best eval = {best_eval}\n")

print(f"Results have been written to {output_file_path}")
import os
import re
import random

# 定义文件路径和命令
libFile = os.path.abspath('./project/lib/7nm/7nm.lib')
logFile = 'circuit.log'

# 每个动作代码对应的Yosys合成操作
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

def calculate_eval(circuitName, actions):
    circuitPath = os.path.abspath(f'C:/CS3308/project/InitialAIG/test/{circuitName}.aig')
    action_cmd = ''
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = f'C:/msys64/mingw64/bin/yosys-abc -c "read {circuitPath}; {action_cmd} read_lib {libFile}; map; topo; stime" > {logFile}'
    os.system(abcRunCmd)
    with open(logFile) as f:
        lines = f.readlines()
        areaInformation = re.findall('[a-zA-Z0-9.-]+', lines[-1])
        eval = float(areaInformation[-6]) * float(areaInformation[-3])
    return eval

def create_initial_population(size, length):
    population = []
    for _ in range(size):
        individual = ''.join(random.choice('0123456') for _ in range(length))
        population.append(individual)
    return population

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    parents = random.choices(population, weights=selection_probs, k=2)
    return parents

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, mutation_rate):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice('0123456')
    return ''.join(individual)

def genetic_algorithm(circuitName, population_size, sequence_length, generations, mutation_rate):
    population = create_initial_population(population_size, sequence_length)
    best_individual = None
    best_eval = -float('inf')

    for generation in range(generations):
        fitnesses = [calculate_eval(circuitName, individual) for individual in population]
        for i, fitness in enumerate(fitnesses):
            if fitness > best_eval:
                best_eval = fitness
                best_individual = population[i]

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = new_population
        print(f"Generation {generation + 1} for {circuitName}: Best eval is {best_eval}")

    return best_individual, best_eval

circuit_names = [
    'alu4', 'apex1', 'apex2', 'apex4', 'b9', 'bar', 'c880', 'c7552', 'cavlc', 'div', 
    'i9', 'm4', 'max1024', 'pair', 'prom1', 'router', 'sqrt', 'square', 'voter'
]

population_size = 20
sequence_length = 10
generations = 50
mutation_rate = 0.25

results = {}

for circuit_name in circuit_names:
    best_sequence, best_eval = genetic_algorithm(circuit_name, population_size, sequence_length, generations, mutation_rate)
    results[circuit_name] = (best_sequence, best_eval)
    print(f"Best sequence for {circuit_name}: {circuit_name}_{best_sequence}")
    print(f"Best eval for {circuit_name}: {best_eval}")

# 将结果写入文件
output_file_path = 'C:/CS3308/output6.txt'
with open(output_file_path, 'w') as output_file:
    for circuit_name, (best_sequence, best_eval) in results.items():
        output_file.write(f"{circuit_name}: Best sequence = {circuit_name}_{best_sequence}, Best eval = {best_eval}\n")

print(f"Results have been written to {output_file_path}")

import os
import re
import random
import math
import multiprocessing as mp

# 定义文件路径和命令
libFile = os.path.abspath('./project/lib/7nm/7nm.lib')
logFile = 'circuit.log'

# 每个动作代码对应的Yosys合成操作
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

def calculate_eval(circuitName, actions):
    circuitPath = os.path.abspath(f'C:/CS3308/project/InitialAIG/test/{circuitName}.aig')
    action_cmd = ''
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = f'C:/msys64/mingw64/bin/yosys-abc -c "read {circuitPath}; {action_cmd} read_lib {libFile}; map; topo; stime" > {logFile}'
    os.system(abcRunCmd)
    with open(logFile) as f:
        lines = f.readlines()
        areaInformation = re.findall('[a-zA-Z0-9.-]+', lines[-1])
        eval = float(areaInformation[-6]) * float(areaInformation[-3])
    return eval

def calculate_eval_parallel(args):
    return calculate_eval(*args)

def create_initial_population(size, length):
    population = []
    for _ in range(size):
        individual = ''.join(random.choice('0123456') for _ in range(length))
        population.append(individual)
    return population

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    parents = random.choices(population, weights=selection_probs, k=2)
    return parents

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[:point]
    return child1, child2

def mutate(individual, mutation_rate):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice('0123456')
    return ''.join(individual)

def genetic_algorithm(circuitName, population_size, sequence_length, generations, mutation_rate):
    population = create_initial_population(population_size, sequence_length)
    best_individual = None
    best_eval = -float('inf')

    with mp.Pool(mp.cpu_count()) as pool:
        for generation in range(generations):
            fitnesses = pool.map(calculate_eval_parallel, [(circuitName, individual) for individual in population])
            for i, fitness in enumerate(fitnesses):
                if fitness > best_eval:
                    best_eval = fitness
                    best_individual = population[i]

            new_population = []
            for _ in range(population_size // 2):
                parent1, parent2 = select_parents(population, fitnesses)
                child1, child2 = crossover(parent1, parent2)
                new_population.append(mutate(child1, mutation_rate))
                new_population.append(mutate(child2, mutation_rate))

            population = new_population
            print(f"Generation {generation + 1} for {circuitName}: Best eval is {best_eval}")

    return best_individual, best_eval

# circuit_names = [
#     'alu4', 'apex1', 'apex2', 'apex4', 'b9', 'bar', 'c880', 'c7552', 'cavlc', 'div', 
#     'i9', 'm4', 'max1024', 'pair', 'prom1', 'router', 'sqrt', 'square', 'voter'
# ]
circuit_names = [
    'alu4'
]

population_size = 20
sequence_length = 10
generations = 50
mutation_rate = 0.25

results = {}

for circuit_name in circuit_names:
    best_sequence, best_eval = genetic_algorithm(circuit_name, population_size, sequence_length, generations, mutation_rate)
    results[circuit_name] = (best_sequence, best_eval)
    print(f"Best sequence for {circuit_name}: {circuit_name}_{best_sequence}")
    print(f"Best eval for {circuit_name}: {best_eval}")

# 将结果写入文件
output_file_path = 'C:/CS3308/output6.txt'
with open(output_file_path, 'w') as output_file:
    for circuit_name, (best_sequence, best_eval) in results.items():
        output_file.write(f"{circuit_name}: Best sequence = {circuit_name}_{best_sequence}, Best eval = {best_eval}\n")

# print(f"Results have been written to {output_file_path}")
