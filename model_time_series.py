import torch
import torch as th
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from graph import EdgeType, HeteroWithGraphIndex, NodeType
from typing import Dict
from Config import RnnType


class TimeHGraphConvLayer(nn.Module):
    def __init__(self, hidden_size, out_channel, svc_feat_num, instance_feat_num, node_feat_num,
                 svc_num, instance_num, node_num, total_node_num):
        super(TimeHGraphConvLayer, self).__init__()
        self.svc_feat_num = svc_feat_num
        self.instance_feat_num = instance_feat_num
        self.node_feat_num = node_feat_num
        self.svc_num = svc_num
        self.instance_num = instance_num
        self.node_num = node_num
        self.total_node_num = total_node_num
        self.conv = dglnn.HeteroGraphConv({
            EdgeType.SVC_CALL_EDGE.value: dglnn.GraphConv(self.svc_feat_num, hidden_size),
            EdgeType.INSTANCE_NODE_EDGE.value: dglnn.GraphConv(self.instance_feat_num, hidden_size),
            EdgeType.NODE_INSTANCE_EDGE.value: dglnn.GraphConv(self.node_feat_num, hidden_size),
            EdgeType.INSTANCE_INSTANCE_EDGE.value: dglnn.GraphConv(self.instance_feat_num, hidden_size)
        },
            aggregate='sum')
        if th.cuda.is_available():
            self.conv = self.conv.to('cuda')
        self.linear_map = {}
        self.linear_map[NodeType.NODE.value] = nn.Linear(self.node_num, out_channel)
        self.linear_map[NodeType.SVC.value] = nn.Linear(self.svc_num, out_channel)
        self.linear_map[NodeType.POD.value] = nn.Linear(self.instance_num, out_channel)
        self.linear_map['total'] = nn.Linear(self.total_node_num, out_channel)
        if th.cuda.is_available():
            self.linear_map['total'] = self.linear_map['total'].to('cuda')
        self.activation = nn.LeakyReLU(negative_slope=1e-2)

    def forward(self, graph, feat_dict):
        dict = self.conv(graph, feat_dict)
        return self.linear_map['total'](
            th.cat([self.activation(dict[key]) for key in dict], dim=0).T).T


class TimeHGraphConvWindow(nn.Module):
    def __init__(self, input_size, hidden_size, graph: HeteroWithGraphIndex, rnn: RnnType = RnnType.LSTM):
        super(TimeHGraphConvWindow, self).__init__()
        self.graph = graph
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.window_simple_size = graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[1]
        if rnn == RnnType.LSTM:
            self.rnn_layer = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2,
                                     batch_first=True)
        elif rnn == RnnType.GRU:
            self.rnn_layer = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2,
                                    batch_first=True)
        if th.cuda.is_available():
            self.rnn_layer = self.rnn_layer.to('cuda')
        self.hGraph_conv_layer_list = []
        self.svc_feat_num = graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[2]
        self.instance_feat_num = graph.hetero_graph.nodes[NodeType.POD.value].data['feat'].shape[2]
        self.node_feat_num = graph.hetero_graph.nodes[NodeType.NODE.value].data['feat'].shape[2]
        self.svc_num = graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[0]
        self.instance_num = graph.hetero_graph.nodes[NodeType.POD.value].data['feat'].shape[0]
        self.node_num = graph.hetero_graph.nodes[NodeType.NODE.value].data['feat'].shape[0]
        for i in range(self.window_simple_size):
            self.hGraph_conv_layer_list.append(
                TimeHGraphConvLayer(self.input_size, self.hidden_size, self.svc_feat_num, self.instance_feat_num,
                                    self.node_feat_num, self.svc_num, self.instance_num, self.node_num,
                                    graph.hetero_graph.num_nodes()))

    def forward(self):
        input_data_list = []

        def get_data_at_time_index(n, data):
            data_at_time_index = []
            for index in range(data.shape[0]):
                data_at_time_index.append(data[index][n])
            return th.stack(data_at_time_index, dim=1).T

        for i, layer in enumerate(self.hGraph_conv_layer_list):
            feat_dict = {
                NodeType.NODE.value: get_data_at_time_index(i, self.graph.hetero_graph.nodes[NodeType.NODE.value].data[
                    'feat']),
                NodeType.SVC.value: get_data_at_time_index(i, self.graph.hetero_graph.nodes[NodeType.SVC.value].data[
                    'feat']),
                NodeType.POD.value: get_data_at_time_index(i, self.graph.hetero_graph.nodes[NodeType.POD.value].data[
                    'feat'])
            }
            single_graph_feat = layer(self.graph.hetero_graph, feat_dict)
            input_data_list.append(single_graph_feat)
        return self.rnn_layer(th.stack(input_data_list, dim=0))[0], self.graph.anomaly_time_series_index


class TimeHGraphConvWindows(nn.Module):
    def __init__(self, out_channel, hidden_channel, graphs: Dict[str, HeteroWithGraphIndex],
                 rnn: RnnType = RnnType.LSTM):
        super(TimeHGraphConvWindows, self).__init__()
        self.hidden_size = hidden_channel
        self.window_size = len(graphs)
        time_sorted = sorted(graphs.keys())
        if rnn == RnnType.LSTM:
            self.rnn_layer = nn.LSTM(input_size=self.hidden_size ** 2, hidden_size=self.hidden_size, num_layers=2,
                                     batch_first=True)
        elif rnn == RnnType.GRU:
            self.rnn_layer = nn.GRU(input_size=self.hidden_size ** 2, hidden_size=self.hidden_size, num_layers=2,
                                    batch_first=True)
        self.time_metrics_layer_list = []
        for time in time_sorted:
            self.time_metrics_layer_list.append(TimeHGraphConvWindow(32, self.hidden_size, graphs[time], rnn))
        self.linear = nn.Linear(self.hidden_size, out_channel)
        self.output_layer = nn.Softmax(dim=0)

    def forward(self):
        input_data_list = []
        anomaly_time_series_index_list = []
        for idx, layer in enumerate(self.time_metrics_layer_list):
            single_window_feat, anomaly_time_series_index = layer()
            anomaly_time_series_index_list.append(anomaly_time_series_index)
            input_data_list.append(single_window_feat.reshape(single_window_feat.shape[0], -1))
            if idx != 0:
                anomaly_time_series_index_list[idx] = [item + sum([i.shape[0] for i in input_data_list[:idx]]) for item
                                                       in anomaly_time_series_index_list[idx]]
        return self.output_layer(self.linear(self.rnn_layer(th.cat(input_data_list, dim=0))[0])), anomaly_time_series_index_list


# 2. 定义无监督的图神经网络模型
class TimeUnsupervisedGNN(nn.Module):
    def __init__(self, out_channels, hidden_size, graphs: Dict[str, HeteroWithGraphIndex], rnn: RnnType = RnnType.LSTM):
        super(TimeUnsupervisedGNN, self).__init__()
        self.conv = TimeHGraphConvWindows(out_channel=out_channels, hidden_channel=hidden_size, graphs=graphs, rnn=rnn)
        self.criterion = nn.MSELoss()

    def forward(self):
        time_series_feat, anomaly_time_series_index_list = self.conv()
        return time_series_feat, anomaly_time_series_index_list

    def loss(self, time_series_feat, time_series_anomaly_index):
        if len(time_series_anomaly_index) == 0:
            return 0
        sum_criterion = 0
        for time_anomaly_index in time_series_anomaly_index:
            if len(time_anomaly_index) == 0:
                continue
            time_series_anomaly_center = time_series_feat[time_anomaly_index]
            sum_criterion += self.criterion(time_series_anomaly_center, torch.zeros_like(time_series_anomaly_center))
        return sum_criterion
