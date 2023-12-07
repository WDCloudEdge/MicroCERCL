import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from graph import EdgeType, HeteroWithGraphIndex, NodeType
from typing import Dict
from model_aggregate import AggrUnsupervisedGNN
from model_time_series import TimeUnsupervisedGNN


# 2. 结合时序异常、拓扑中心点聚集的无监督的图神经网络模型
class UnsupervisedGNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, graphs: Dict[str, HeteroWithGraphIndex]):
        super(UnsupervisedGNN, self).__init__()
        self.aggr_conv = AggrUnsupervisedGNN(out_channels=out_channels, hidden_size=hidden_size, graphs=graphs)
        self.time_conv = TimeUnsupervisedGNN(out_channels=out_channels, hidden_size=hidden_size, graphs=graphs)

    def forward(self):
        aggr_feat, aggr_index = self.aggr_conv()
        time_feat = self.time_conv()
        return aggr_feat, aggr_index, time_feat

    def loss(self, aggr_feat, aggr_index, time_feat, time_index):
        return th.mean([self.aggr_conv.loss(aggr_feat, aggr_index), self.time_conv.loss(time_feat, time_index)])


def train(graphs: Dict[str, HeteroWithGraphIndex]):
    # 3. 初始化模型和优化器
    model = UnsupervisedGNN(in_channels=5, out_channels=1, hidden_size=64, graphs=graphs)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. 训练模型
    for epoch in range(1000):
        optimizer.zero_grad()
        aggr_feat, aggr_index, time_feat = model()

        # 使用节点的 L2 范数作为无监督的目标函数
        # loss = th.norm(output, dim=1).mean()
        loss = model.loss(aggr_feat, aggr_index, time_feat, [])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # 5. 获取学到的节点表示
    with th.no_grad():
        model.eval()
        node_embeddings = model().detach().numpy()

    # 6. 在学到的节点表示上执行中心性，图聚类，子图划分，因果推断等操作，进一步分析根因
    print(node_embeddings)
