import os
import pickle
import hashlib
import torch
import numpy as np
import abc_py
from torch_geometric.data import Data

# 初始化 abc_py 接口
abc = abc_py.AbcInterface()
abc.start()

# 定义路径
data_dir = './task2/project_data2'
initial_aig_dir = './InitialAIG/train'
output_dir = './processed_data2'
os.makedirs(output_dir, exist_ok=True)

# 获取所有数据文件
data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]

def hash_actions(actions):
    actions_str = ''.join(actions)
    return hashlib.md5(actions_str.encode()).hexdigest()

def process_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        input_states = data['input']
        future_rewards = data['target']

    for idx, state in enumerate(input_states):
        circuit_name, actions = state.split('_', 1)
        circuit_path = os.path.join(initial_aig_dir, f'{circuit_name}.aig')
        lib_file = './lib/7nm/7nm.lib'
        log_file = f'{circuit_name}.log'
        
        abc.read(circuit_path)
        num_nodes = abc.numNodes()
        node_type = np.zeros(num_nodes, dtype=int)
        edge_src_index = []
        edge_target_index = []

        for node_idx in range(num_nodes):
            aig_node = abc.aigNode(node_idx)
            node_type[node_idx] = aig_node.nodeType()
            if aig_node.hasFanin0():
                fanin0 = aig_node.fanin0()
                edge_src_index.append(fanin0)
                edge_target_index.append(node_idx)
            if aig_node.hasFanin1():
                fanin1 = aig_node.fanin1()
                edge_src_index.append(fanin1)
                edge_target_index.append(node_idx)

        edge_index = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
        node_type = torch.tensor(node_type, dtype=torch.float).unsqueeze(1)
        actions_tensor = torch.tensor([int(action) for action in actions], dtype=torch.long)

        graph_data = Data(
            x=node_type,
            edge_index=edge_index,
            y=torch.tensor(future_rewards[idx], dtype=torch.float).unsqueeze(0)
        )
        graph_data.actions = actions_tensor

        action_hash = hash_actions(actions)
        output_file = os.path.join(output_dir, f'{circuit_name}_{action_hash}.pkl')
        with open(output_file, 'wb') as out_f:
            pickle.dump(graph_data, out_f)

for file_path in data_files:
    process_data(file_path)

# 关闭 abc_py 接口
abc.stop()

print("Data preprocessing completed.")
