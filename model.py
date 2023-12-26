import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
from graph import HeteroWithGraphIndex, NodeType
from typing import Dict
from model_aggregate import AggrUnsupervisedGNN
from model_time_series import TimeUnsupervisedGNN
import wandb
from datetime import datetime


class EarlyStopping:
    def __init__(self, patience=5, delta=1e-6):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss >= self.best_loss:
            self.counter = 0
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# 2. 结合时序异常、拓扑中心点聚集的无监督的图神经网络模型
class UnsupervisedGNN(nn.Module):
    def __init__(self, out_channels, hidden_size, graphs: Dict[str, HeteroWithGraphIndex]):
        super(UnsupervisedGNN, self).__init__()
        self.aggr_conv = AggrUnsupervisedGNN(out_channels=out_channels, hidden_size=hidden_size, graphs=graphs)
        self.time_conv = TimeUnsupervisedGNN(out_channels=out_channels, hidden_size=hidden_size, graphs=graphs)

    def forward(self):
        aggr_feat, aggr_center_index, aggr_anomaly_index = self.aggr_conv()
        time_series_feat, anomaly_time_series_index_list = self.time_conv()
        return aggr_feat, aggr_center_index, aggr_anomaly_index, time_series_feat, anomaly_time_series_index_list

    def loss(self, aggr_feat, aggr_center_index, aggr_anomaly_index, time_feat, time_index):
        if len(time_index) == 0:
            return self.aggr_conv.loss(aggr_feat, aggr_center_index, aggr_anomaly_index)
        return self.aggr_conv.loss(aggr_feat, aggr_center_index, aggr_anomaly_index) + self.time_conv.loss(time_feat, time_index)


def train(graphs: Dict[str, HeteroWithGraphIndex], dir: str = '', is_train: bool = False, learning_rate=0.01):
    model = UnsupervisedGNN(out_channels=1, hidden_size=64, graphs=graphs)
    if is_train:
        wandb.init(project="MicroCERC", name="MicroCERC " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        wandb.watch(model)
        early_stopping = EarlyStopping(patience=5, delta=1e-5)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(500):
            optimizer.zero_grad()
            aggr_feat, aggr_center_index, aggr_anomaly_index, time_series_feat, anomaly_time_series_index_list = model()

            loss = model.loss(aggr_feat, aggr_center_index, aggr_anomaly_index, time_series_feat, anomaly_time_series_index_list)

            loss.backward()
            optimizer.step()
            early_stopping(loss)

            if epoch % 10 == 0:
                wandb.log({"loss": loss})
            if early_stopping.early_stop:
                print(f"Early stopping with epoch: {epoch}, loss: {loss.item()}")
                break
        torch.save(model.state_dict(), dir + '/model_weights.pth')
    else:
        model.load_state_dict(torch.load(dir + '/model_weights.pth'))
    with th.no_grad():
        model.eval()
        output = model()[0]
        output = output.reshape(output.shape[0], -1)
        time_sorted = sorted(graphs.keys())
        total_num_list = []
        total_index_node = {}
        for idx, time in enumerate(time_sorted):
            graph = graphs[time]
            total_num = graph.hetero_graph.num_nodes()
            total_num_list.append(total_num)
            node_num = graph.hetero_graph.num_nodes(NodeType.NODE.value)
            svc_num = graph.hetero_graph.num_nodes(NodeType.SVC.value)
            pod_num = graph.hetero_graph.num_nodes(NodeType.POD.value)
            graph_index = graph.hetero_graph_index
            if idx == 0:
                padding_num = 0
            else:
                padding_num = sum(total_num_list[:idx])
            for i in range(total_num):
                if i < node_num:
                    total_index_node[i + padding_num] = [key for key, value in graph_index.node_index.items() if value == i][0]
                elif node_num <= i < node_num + pod_num:
                    total_index_node[i + padding_num] = [key for key, value in graph_index.pod_index.items() if value == i - node_num][0]
                elif node_num + pod_num <= i < total_num:
                    total_index_node[i + padding_num] = [key for key, value in graph_index.svc_index.items() if value == i - node_num - pod_num][0]
        output = torch.sum(output, dim=1, keepdim=True).T.numpy().flatten().tolist()
        output_score = {}
        for idx, score in enumerate(output):
            s = output_score.get(total_index_node[idx], 0)
            output_score[total_index_node[idx]] = s + score
        sorted_dict = dict(sorted(output_score.items(), key=lambda item: item[1]))
        for key, value in list(sorted_dict.items())[:10]:
            print(f"{key}: {value}")
