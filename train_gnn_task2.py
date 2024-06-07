import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 定义路径
output_dir = './processed_data2'
model_save_dir = './saved_models'
os.makedirs(model_save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取所有处理后的pkl文件
processed_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.pkl')]

# 加载所有数据
data_list = []
for file_path in processed_files:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            edge_index = data['edge_index']
            node_type = data['x']
            actions = data['actions']
            future_reward = data['y']

            # 检查edge_index中的索引是否在有效范围内
            if edge_index.max() >= node_type.size(0):
                continue

            graph_data = Data(
                x=node_type.float(),  # 将节点类型作为节点特征
                edge_index=edge_index,
                y=future_reward
            )
            graph_data.actions = actions  # 将动作序列添加到数据中
            data_list.append(graph_data)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Error reading file {file_path}: {e}")
        continue

# 确保data_list不为空
if len(data_list) == 0:
    raise ValueError("No valid data found")

# 划分数据集
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# 构建数据加载器
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# 定义GNN模型
class GNNModel(nn.Module):
    def __init__(self, node_feature_dim, action_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        x, edge_index, batch, actions = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.actions.to(device)
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
        x = global_mean_pool(x, batch)  # 全局平均池化

        # 确保actions_expanded与x的形状匹配
        actions_expanded = actions.unsqueeze(0).expand(x.size(0), -1)

        in_features = x.shape[1] + actions_expanded.shape[1]
        if self.fc1.in_features != in_features:
            self.fc1 = nn.Linear(in_features, self.fc1.out_features).to(device)
            self.batch_norm3 = nn.BatchNorm1d(self.fc1.out_features).to(device)

        x = torch.cat([x, actions_expanded.float()], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 确保data_list不为空，并且每个元素都有有效的actions
if len(data_list) > 0 and hasattr(data_list[0], 'actions'):
    action_dim = data_list[0].actions.size(0)  # 动作序列长度
else:
    raise ValueError("No valid data found with actions")

# 模型参数
node_feature_dim = data_list[0].x.size(1)  # 节点特征维度
hidden_dim = 256  # 增加隐藏层维度
output_dim = 1  # 输出维度（未来奖励）

# 初始化模型并移动到GPU
model = GNNModel(node_feature_dim, action_dim, hidden_dim, output_dim).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00002)  # 降低学习率

def compute_relative_error(output, target):
    # 如果target为0，直接计算output的绝对误差
    relative_error = torch.abs((output - target) / (target + (target == 0).float()))
    return relative_error

def train(model, train_loader, test_loader, criterion, optimizer, epochs=100):  # 增加训练轮数
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_relative_error = 0
        train_loader_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for data in train_loader_iter:
            data = data.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.view(-1, 1))  # 调整目标值形状
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 计算相对误差
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
            loss = criterion(output, data.y.view(-1, 1))  # 调整目标值形状
            total_loss += loss.item()

            # 计算相对误差
            total_relative_error += compute_relative_error(output, data.y.view(-1, 1)).sum().item()

    avg_relative_error = total_relative_error / len(test_loader.dataset)
    return total_loss / len(test_loader), avg_relative_error

# 训练并测试模型
train(model, train_loader, test_loader, criterion, optimizer, epochs=100)



