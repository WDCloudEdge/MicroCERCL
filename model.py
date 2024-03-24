import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
from graph import HeteroWithGraphIndex, NodeType, calculate_graph_score
from typing import Dict
from model_aggregate import AggrUnsupervisedGNN
from model_time_series import TimeUnsupervisedGNN
from Config import RnnType, TrainType
from torch.nn import init
from torch.optim.lr_scheduler import StepLR
from util.utils import top_k_node, top_k_node_time_series
import wandb
from datetime import datetime


class EarlyStopping:
    def __init__(self, patience=5, delta=1e-6, max_loss=5, min_epoch=500):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.output_list = []
        self.max_loss = max_loss
        self.min_epoch = min_epoch

    def __call__(self, val_loss, output, epoch):
        if val_loss > self.max_loss or epoch < self.min_epoch:
            self.counter = 0
            self.output_list.clear()
        elif self.best_loss == float('inf'):
            self.best_loss = val_loss
        elif abs(self.best_loss - val_loss) < self.delta:
            self.counter += 1
            self.output_list.append(output)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self.output_list.clear()
            if val_loss < self.best_loss:
                self.best_loss = val_loss


# 2. 结合时序异常、拓扑中心点聚集的无监督的图神经网络模型
class UnsupervisedGNN(nn.Module):
    def __init__(self, anomaly_index, out_channels, hidden_size, graphs: Dict[str, HeteroWithGraphIndex], rnn: RnnType = RnnType.LSTM,
                 attention: bool = False):
        super(UnsupervisedGNN, self).__init__()
        graph = graphs[next(iter(graphs))]
        self.aggr_conv = AggrUnsupervisedGNN(anomaly_index, out_channels=out_channels, hidden_size=hidden_size, rnn=rnn,
                                             attention=attention,
                                             svc_feat_num=
                                             graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[2],
                                             node_feat_num=None,
                                             instance_feat_num=
                                             graph.hetero_graph.nodes[NodeType.POD.value].data['feat'].shape[2])
        # self.time_conv = TimeUnsupervisedGNN(out_channels=out_channels, hidden_size=hidden_size, graphs=graphs, rnn=rnn)
        self.epoch = 0

    def forward(self, graphs: Dict[str, HeteroWithGraphIndex]):
        aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series = self.aggr_conv(
            graphs)
        # time_series_feat, anomaly_time_series_index_list = self.time_conv()
        # return aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, time_series_feat, anomaly_time_series_index_list
        return aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series

    # def loss(self, aggr_feat, aggr_center_index, aggr_anomaly_index, time_feat, time_index):
    def loss(self, aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes,
             window_anomaly_time_series):
        return self.aggr_conv.loss(aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index,
                                   window_time_series_sizes, window_anomaly_time_series)
        # if len(time_index) == 0 or len([item for sublist in time_index for item in sublist]) == 0:
        #     return self.aggr_conv.loss(aggr_feat, aggr_center_index, aggr_anomaly_index)
        # return self.aggr_conv.loss(aggr_feat, aggr_center_index, aggr_anomaly_index) + self.time_conv.loss(time_feat,
        #                                                                                                    time_index)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def train(config, label: str, root_cause: str, anomaly_index: Dict[str, int], graphs: Dict[str, HeteroWithGraphIndex], dir: str = '',
          is_train: TrainType = TrainType.EVAL,
          learning_rate=0.01, rnn: RnnType = RnnType.LSTM, attention: bool = False):
    model = UnsupervisedGNN(anomaly_index, out_channels=1, hidden_size=64, graphs=graphs, rnn=rnn, attention=attention)
    if torch.cuda.is_available():
        model = model.to('cpu')
    label = label
    root_cause_file = label + '_' + rnn.value + ('_atten' if attention else '')
    model_file = 'model_weights' + '_' + label + '_' + rnn.value + (
        '_atten' if attention else '') + '.pth'
    root_cause = root_cause
    with open(dir + '/result-' + root_cause_file + '.log', "a") as output_file:
        print(f"root_cause: {root_cause}", file=output_file)
        early_stopping = EarlyStopping(patience=config.patience, delta=config.delta, min_epoch=config.min_epoch)
        if is_train == TrainType.TRAIN or is_train == TrainType.TRAIN_CHECKPOINT:
            if is_train == TrainType.TRAIN:
                model.initialize_weights()
            elif is_train == TrainType.TRAIN_CHECKPOINT:
                model.load_state_dict(torch.load(dir + '/' + model_file))
            # wandb.init(project="MicroCERC", name="MicroCERC " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            # wandb.watch(model)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = StepLR(optimizer, step_size=200, gamma=0.5)
            for epoch in range(model.epoch, 1000, 1):
                model.set_epoch(epoch)
                optimizer.zero_grad()
                # aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, time_series_feat, anomaly_time_series_index_list = model(graphs)
                aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series = model(
                    graphs)

                # loss = model.loss(aggr_feat, aggr_center_index, aggr_anomaly_index, time_series_feat,
                #                   anomaly_time_series_index_list)
                loss = model.loss(aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index,
                                  window_time_series_sizes, window_anomaly_time_series)
                early_stopping(loss, aggr_feat, epoch)
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
            # aggr_feat_list, aggr_center_index, aggr_anomaly_index, window_graphs_index, time_series_feat, anomaly_time_series_index_list = model(graphs)
            aggr_feat_list, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series = model(
                graphs)
            # time_sorted = sorted(graphs.keys())
            # total_num_list = []
            # total_index_node = {}
            # for idx, time in enumerate(time_sorted):
            #     graph = graphs[time]
            #     total_num = graph.hetero_graph.num_nodes()
            #     total_num_list.append(total_num)
            #     node_num = graph.hetero_graph.num_nodes(NodeType.NODE.value)
            #     svc_num = graph.hetero_graph.num_nodes(NodeType.SVC.value)
            #     pod_num = graph.hetero_graph.num_nodes(NodeType.POD.value)
            #     graph_index = graph.hetero_graph_index
            #     if idx == 0:
            #         padding_num = 0
            #     else:
            #         padding_num = sum(total_num_list[:idx])
            #     for i in range(total_num):
            #         if i < node_num:
            #             total_index_node[i + padding_num] = \
            #                 [key for key, value in graph_index.node_index.items() if value == i][0]
            #         elif node_num <= i < node_num + pod_num:
            #             total_index_node[i + padding_num] = \
            #                 [key for key, value in graph_index.pod_index.items() if value == i - node_num][0]
            #         elif node_num + pod_num <= i < total_num:
            #             total_index_node[i + padding_num] = \
            #                 [key for key, value in graph_index.svc_index.items() if value == i - node_num - pod_num][0]
            if is_train == TrainType.TRAIN or is_train == TrainType.TRAIN_CHECKPOINT:
                output_list = early_stopping.output_list
                if len(output_list) == 0:
                    output_list = [aggr_feat_list]
            elif is_train == TrainType.EVAL:
                output_list = [aggr_feat_list]
            times = graphs.keys()
            times_sorted = sorted(times)
            # output_score = {}
            output_score_node = {}
            # 如果是单图，将aggr_feat按照图节点索引，将特征还原到图上
            if aggr_feat_list.shape[0] == 1:
                output = aggr_feat_list[0]
                window_graph_index = window_graphs_index[0]
                graph = graphs[times_sorted[0]]
                n_graph = graph.n_graph.copy()
                node_features = {}
                for i in range(output.shape[0]):
                    node_features[i] = torch.sum(output[0]).item()
                sorted_dict_node_pagerank = calculate_graph_score(n_graph, node_features, window_graph_index)
                top_k_node(sorted_dict_node_pagerank, root_cause, output_file)
            for aggr_feat_list in output_list:
                for idx, window_graph_index in enumerate(window_graphs_index):
                    output = aggr_feat_list[idx]
                    window_graph_index_reverse = {window_graph_index[key]: key for key in window_graph_index}
                    # graph = graphs[times_sorted[idx]]
                    # graph_index_time_map = graph.graph_index_time_map
                    # aggr_feat = aggr_feat_list[idx]
                    # flattened_tensor = aggr_feat.flatten()
                    # sorted_indices = torch.argsort(flattened_tensor, descending=True)
                    # rows = sorted_indices // aggr_feat.size(1)
                    # cols = sorted_indices % aggr_feat.size(1)
                    # for i in range(len(rows)):
                    #     node_time = window_graph_index_reverse[rows[i].item()] + '-' + str(
                    #         graph_index_time_map[cols[i].item()])
                    #     node_time_score = output_score.get(node_time, 0)
                    #     node_time_score += flattened_tensor[sorted_indices[i].item()].item()
                    #     output_score[node_time] = node_time_score
                    # output = torch.sum(aggr_feat, dim=1,
                    #                    keepdim=True).T.numpy().flatten().tolist()
                    for idx, score in enumerate(output):
                        node = window_graph_index_reverse[idx]
                        s = output_score_node.get(node, 0)
                        if s != 0:
                            output_score_node[node] = s + abs(score.item())
                        else:
                            output_score_node[node] = abs(score.item())
            # sorted_dict = dict(sorted(output_score.items(), key=lambda item: item[1], reverse=True))
            sorted_dict_node = dict(sorted(output_score_node.items(), key=lambda item: item[1], reverse=True))
            # top_k_node_time_series(sorted_dict, root_cause, output_file)
            return top_k_node(sorted_dict_node, root_cause, output_file)
