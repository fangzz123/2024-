import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from tqdm import tqdm

# 定义路径
output_dir = './processed_data'
model_save_dir = './saved_models'
os.makedirs(model_save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        edge_index = data['edge_index']
        node_type = data['node_type']
        num_nodes = data['num_nodes']
        actions = data['actions']
        label = data['label']

        # 检查edge_index中的索引是否在有效范围内
        if edge_index.max() >= num_nodes:
            return None

        graph_data = Data(
            x=node_type.unsqueeze(1).float(),  # 将节点类型作为节点特征
            edge_index=edge_index,
            y=label.unsqueeze(0)
        )
        graph_data.actions = actions  # 将动作序列添加到数据中
        return graph_data

def load_data(processed_files):
    with Pool() as pool:
        data_list = pool.map(process_file, processed_files)
    return [data for data in data_list if data is not None]

if __name__ == '__main__':
    # 获取所有处理后的pkl文件
    processed_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.pkl')]

    # 使用多进程加载所有文件
    data_list = load_data(processed_files)

    # 划分数据集
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

    # 构建数据加载器
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 定义更复杂的GNN模型
    class ComplexGNNModel(nn.Module):
        def __init__(self, node_feature_dim, action_dim, hidden_dim, output_dim):
            super(ComplexGNNModel, self).__init__()
            self.conv1 = GCNConv(node_feature_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
            self.dropout = nn.Dropout(p=0.5)
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm3 = nn.BatchNorm1d(hidden_dim // 2)

        def forward(self, data):
            x, edge_index, batch, actions = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.actions.to(device)
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.batch_norm1(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            x = torch.relu(x)
            x = self.conv3(x, edge_index)
            x = torch.relu(x)
            x = global_mean_pool(x, batch)

            if actions.dim() == 1:
                actions_expanded = actions.unsqueeze(0).expand(x.size(0), -1)
            else:
                actions_expanded = actions.expand(x.size(0), -1)

            in_features = x.shape[1] + actions_expanded.shape[1]
            if self.fc1.in_features != in_features:
                self.fc1 = nn.Linear(in_features, self.fc1.out_features).to(device)

            x = torch.cat([x, actions_expanded.float()], dim=1)
            x = torch.relu(self.fc1(x))
            x = self.batch_norm2(x)
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.batch_norm3(x)
            x = self.fc3(x)
            return x

    if len(data_list) > 0 and 'actions' in data_list[0]:
        action_dim = data_list[0].actions.size(0)
    else:
        raise ValueError("No valid data found with actions")

    node_feature_dim = 1
    hidden_dim = 256
    output_dim = 1

    # 初始化模型并移动到GPU
    model = ComplexGNNModel(node_feature_dim, action_dim, hidden_dim, output_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    def compute_relative_error(output, target):
        relative_error = torch.abs((output - target) / (target + (target == 0).float()))
        return relative_error

    def train(model, train_loader, test_loader, criterion, optimizer, epochs=100):
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_relative_error = 0
            train_loader_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for data in train_loader_iter:
                data = data.to(device)  # 将数据移动到GPU
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data.y.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_relative_error += compute_relative_error(output, data.y.view(-1, 1)).sum().item()

                train_loader_iter.set_postfix({
                    'Train Loss': total_loss / len(train_loader),
                    'Train Avg Relative Error': total_relative_error / len(train_loader.dataset)
                })

            avg_relative_error = total_relative_error / len(train_loader.dataset)
            print(f'Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader)}, Train Avg Relative Error: {avg_relative_error}')

            # 保存模型
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_epoch_{epoch + 1}.pth'))

            # 测试模型在测试集上的表现
            test_loss, test_avg_relative_error = test(model, test_loader, criterion)
            print(f'Epoch {epoch + 1}, Test Loss: {test_loss}, Test Avg Relative Error: {test_avg_relative_error}')

    def test(model, test_loader, criterion):
        model.eval()
        total_loss = 0
        total_relative_error = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)  # 将数据移动到GPU
                output = model(data)
                loss = criterion(output, data.y.view(-1, 1))
                total_loss += loss.item()
                total_relative_error += compute_relative_error(output, data.y.view(-1, 1)).sum().item()

        avg_relative_error = total_relative_error / len(test_loader.dataset)
        return total_loss / len(test_loader), avg_relative_error

    # 训练并测试模型
    train(model, train_loader, test_loader, criterion, optimizer, epochs=100)






