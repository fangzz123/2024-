import os
import pickle
import abc_py
import torch
import numpy as np

# 定义路径
data_dir = './task1/project_data'
aig_dir = './InitialAIG/train'
lib_file = './lib/7nm/7nm.lib'
output_dir = './task1/processed_data'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 获取所有pkl文件
data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

# 启动abc_py接口
abc = abc_py.AbcInterface()
abc.start()

for data_file in data_files:
    file_path = os.path.join(data_dir, data_file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        inputs = data['input']
        targets = data['target']
        
        for state, label in zip(inputs, targets):
            # 拆分state得到电路名和动作序列
            circuit_name, *actions = state.split('_')
            actions = ''.join(actions)
            
            # 构建初始AIG文件路径
            circuit_path = os.path.join(aig_dir, f"{circuit_name}.aig")
            
            # 解析初始AIG文件
            abc.read(circuit_path)
            node_type = []
            edge_src_index = []
            edge_target_index = []
            
            num_nodes = abc.numNodes()
            for node_idx in range(num_nodes):
                aig_node = abc.aigNode(node_idx)
                node_type.append(aig_node.nodeType())
                
                if aig_node.hasFanin0():
                    edge_src_index.append(node_idx)
                    edge_target_index.append(aig_node.fanin0())
                if aig_node.hasFanin1():
                    edge_src_index.append(node_idx)
                    edge_target_index.append(aig_node.fanin1())
            
            # 将特征数据存储为张量
            graph_data = {
                'edge_index': torch.tensor([edge_src_index, edge_target_index], dtype=torch.long),
                'node_type': torch.tensor(node_type, dtype=torch.long),
                'num_nodes': num_nodes,
                'actions': torch.tensor([int(a) for a in actions], dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float)
            }
            
            # 保存处理后的图数据为新的.pkl文件
            output_file = os.path.join(output_dir, f'{circuit_name}_{actions}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(graph_data, f)

# 删除abc接口对象
del abc

print(f'Total files processed and saved: {len(data_files)}')
