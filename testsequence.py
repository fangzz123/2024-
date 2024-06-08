import os
import numpy as np
import torch
import pickle
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import abc_py as abcPy

# 定义GNN模型
class GNNModel(nn.Module):
    def __init__(self, node_feature_dim, action_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(303, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        x, edge_index, actions = data.x, data.edge_index, data.actions

        if edge_index.max() >= x.size(0):
            raise IndexError(f"Edge index out of bounds. Max index: {edge_index.max()}, Num nodes: {x.size(0)}")

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        batch = torch.zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)

        actions_expanded = actions.expand(x.size(0), -1)

        x = torch.cat([x, actions_expanded.float()], dim=1)

        # 填充x，使其形状为[1, 303]
        if x.shape[1] < 303:
            padding = 303 - x.shape[1]
            x = F.pad(x, (0, padding))

        x = torch.relu(self.fc1(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 设置设备
device = torch.device('cpu')

# 定义文件路径和命令
libFile = os.path.abspath('./lib/7nm/7nm.lib')
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

class AbcPyInterface:
    def __init__(self):
        self._abc = abcPy.AbcInterface()
        self._abc.start()
    
    def create_graph(self, state, action):
        try:
            state_file_path = f'./InitialAIG/test/{state}.aig'
            print(f"Reading state from: {state_file_path}")
            if not os.path.exists(state_file_path):
                print(f"State file {state_file_path} does not exist.")
                return None
            
            self._abc.read(state_file_path)
        except Exception as e:
            print(f"Error reading state {state_file_path}: {e}")
            return None

        try:
            num_nodes = self._abc.numNodes()
            if num_nodes == 0:
                raise ValueError(f"Empty network for state {state_file_path}")
            print(f"Number of nodes: {num_nodes}")

            node_type = np.zeros(num_nodes, dtype=int)
            edge_src_index = []
            edge_target_index = []

            for node_idx in range(num_nodes):
                aig_node = self._abc.aigNode(node_idx)
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

            actions_tensor = torch.tensor(action, dtype=torch.float).unsqueeze(0)

            graph_data = Data(
                x=node_type,
                edge_index=edge_index
            )
            graph_data.actions = actions_tensor

            return graph_data
        except Exception as e:
            print(f"Error creating graph from state {state_file_path}: {e}")
            return None

def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=device)  # 加载到 CPU
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def calculate_eval(circuitName, action):
    state = circuitName
    abc_interface = AbcPyInterface()
    data = abc_interface.create_graph(state, action)

    if data is None:
        print(f"Failed to create graph for state {state}")
        return -float('inf')

    model_path = 'model_epoch_1.pth'
    node_feature_dim = 1
    action_dim = 9
    hidden_dim = 256
    output_dim = 1
    model = GNNModel(node_feature_dim, action_dim, hidden_dim, output_dim).to(device)
    model = load_model(model_path, model)
    model.eval()

    try:
        with torch.no_grad():
            eval_result = model(data)
        print(f"Evaluation result for state {state}, action {action}: {eval_result.item()}")
        return eval_result.item()
    except Exception as e:
        print(f"Error evaluating model for state {state}: {e}")
        return -float('inf')

def find_best_action_for_position(circuitName, accumulated_actions):
    best_eval = -float('inf')
    best_action = -1

    for action in range(7):
        temp_actions = accumulated_actions.copy()
        temp_actions[action] = 1
        current_eval = calculate_eval(circuitName, temp_actions)
        if current_eval > best_eval:
            best_eval = current_eval
            best_action = action
            best_action_sequence = temp_actions.copy()

    return best_action, best_eval, best_action_sequence

def find_best_sequence(initial_state, max_depth):
    circuitName, _ = initial_state.split('_')
    accumulated_actions = [0] * 9
    best_action_sequence = []

    for depth in range(max_depth):
        best_action, best_eval, accumulated_actions = find_best_action_for_position(circuitName, accumulated_actions)
        best_action_sequence.append(best_action)
        print(f"Depth {depth + 1} for {circuitName}: Best action so far is {best_action} with eval {best_eval}")

    return best_action_sequence, best_eval

circuit_names = [
    'alu4', 'apex1', 'apex4', 'b9', 'bar'
]

max_depth = 10

results = {}

for circuit_name in circuit_names:
    initial_state = f'{circuit_name}_'
    print(f"Processing circuit {circuit_name}")
    best_sequence, best_eval = find_best_sequence(initial_state, max_depth)
    results[circuit_name] = (best_sequence, best_eval)
    print(f"Best sequence for {circuit_name}: {best_sequence}")
    print(f"Best eval for {circuit_name}: {best_eval}")

# 将结果写入文件
output_file_path = 'output1.txt'
with open(output_file_path, 'w') as output_file:
    for circuit_name, (best_sequence, best_eval) in results.items():
        output_file.write(f"{circuit_name}: Best sequence = {best_sequence}, Best eval = {best_eval}\n")

print(f"Results have been written to {output_file_path}")
