import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
from graph import HeteroWithGraphIndex, NodeType
from typing import Dict
from model_aggregate import AggrUnsupervisedGNN
from model_time_series import TimeUnsupervisedGNN
from Config import RnnType, TrainType
from torch.nn import init
from torch.optim.lr_scheduler import StepLR
import wandb
from datetime import datetime


class EarlyStopping:
    def __init__(self, patience=5, delta=1e-6, max_loss=1):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.output_list = []
        self.max_loss = max_loss

    def __call__(self, val_loss, output):
        if val_loss >= self.best_loss or val_loss > self.max_loss:
            self.counter = 0
            self.output_list.clear()
        elif val_loss < self.best_loss:
            if self.best_loss - val_loss > self.delta:
                self.counter = 0
                self.output_list.clear()
                self.best_loss = val_loss
            else:
                self.counter += 1
                self.output_list.append(output)
                if self.counter >= self.patience:
                    self.early_stop = True


# 2. 结合时序异常、拓扑中心点聚集的无监督的图神经网络模型
class UnsupervisedGNN(nn.Module):
    def __init__(self, out_channels, hidden_size, graphs: Dict[str, HeteroWithGraphIndex], rnn: RnnType = RnnType.LSTM, attention: bool = False):
        super(UnsupervisedGNN, self).__init__()
        self.aggr_conv = AggrUnsupervisedGNN(out_channels=out_channels, hidden_size=hidden_size, graphs=graphs, rnn=rnn, attention=attention)
        self.time_conv = TimeUnsupervisedGNN(out_channels=out_channels, hidden_size=hidden_size, graphs=graphs, rnn=rnn)
        self.epoch = 0

    def forward(self):
        aggr_feat, aggr_center_index, aggr_anomaly_index = self.aggr_conv()
        time_series_feat, anomaly_time_series_index_list = self.time_conv()
        return aggr_feat, aggr_center_index, aggr_anomaly_index, time_series_feat, anomaly_time_series_index_list

    def loss(self, aggr_feat, aggr_center_index, aggr_anomaly_index, time_feat, time_index):
        if len(time_index) == 0 or len([item for sublist in time_index for item in sublist]) == 0:
            return self.aggr_conv.loss(aggr_feat, aggr_center_index, aggr_anomaly_index)
        return self.aggr_conv.loss(aggr_feat, aggr_center_index, aggr_anomaly_index) + self.time_conv.loss(time_feat, time_index)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def train(graphs: Dict[str, HeteroWithGraphIndex], dir: str = '', is_train: TrainType = TrainType.EVAL, learning_rate=0.01, rnn: RnnType = RnnType.LSTM, attention: bool = False):
    model = UnsupervisedGNN(out_channels=1, hidden_size=64, graphs=graphs, rnn=rnn, attention=attention)
    if torch.cuda.is_available():
        model = model.to('cuda')
    label = 'label11'
    root_cause_file = label + '_' + rnn.value + ('_atten' if attention else '')
    model_file = 'model_weights' + '_' + label + '_' + rnn.value + ('_atten' if attention else '') + '.pth'
    root_cause = 'productcatalogservice-5ff5f57dc8-mpw5r'
    with open(dir + '/result-' + root_cause_file + '.log', "a") as output_file:
        print(f"root_cause: {root_cause}", file=output_file)
        early_stopping = EarlyStopping(patience=5, delta=1e-5)
        if is_train == TrainType.TRAIN or is_train == TrainType.TRAIN_CHECKPOINT:
            if is_train == TrainType.TRAIN:
                model.initialize_weights()
            elif is_train == TrainType.TRAIN_CHECKPOINT:
                model.load_state_dict(torch.load(dir + '/' + model_file))
            # wandb.init(project="MicroCERC", name="MicroCERC " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            # wandb.watch(model)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
            for epoch in range(model.epoch, 10000, 1):
                model.set_epoch(epoch)
                optimizer.zero_grad()
                aggr_feat, aggr_center_index, aggr_anomaly_index, time_series_feat, anomaly_time_series_index_list = model()

                loss = model.loss(aggr_feat, aggr_center_index, aggr_anomaly_index, time_series_feat, anomaly_time_series_index_list)

                early_stopping(loss, model()[0])
                loss.backward()
                optimizer.step()
                scheduler.step()

                if epoch % 10 == 0:
                    # wandb.log({"loss": loss})
                    print(f"epoch: {epoch}, loss: {loss}", file=output_file)
                    print(f"epoch: {epoch}, loss: {loss}")
                if early_stopping.early_stop:
                    print(f"Early stopping with epoch: {epoch}, loss: {loss.item()}", file=output_file)
                    break
            torch.save(model.state_dict(), dir + '/' + model_file)
        elif is_train == TrainType.EVAL:
            model.load_state_dict(torch.load(dir + '/' + model_file))
        with th.no_grad():
            model.eval()
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
            if is_train == TrainType.TRAIN or is_train == TrainType.TRAIN_CHECKPOINT:
                output_list = early_stopping.output_list
                if len(output_list) == 0:
                    output_list = [model()[0]]
            elif is_train == TrainType.EVAL:
                output_list = [model()[0]]
            output_score = {}
            for output in output_list:
                output = output.T.numpy().flatten().tolist()
                for idx, score in enumerate(output):
                    s = output_score.get(total_index_node[idx], 0)
                    if s != 0:
                        output_score[total_index_node[idx]] = (s + abs(score)) / 2
                    else:
                        output_score[total_index_node[idx]] = abs(score)
            sorted_dict = dict(sorted(output_score.items(), key=lambda item: item[1]))
            top_k = 0
            is_top_k = False
            for key, value in list(sorted_dict.items()):
                if not is_top_k:
                    top_k += 1
                print(f"{key}: {value}", file=output_file)
                if key == root_cause:
                    is_top_k = True
            print(f"top_k: {top_k}", file=output_file)
