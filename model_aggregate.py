import torch
import torch as th
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from dgl import (
    broadcast_nodes,
    max_nodes,
    mean_nodes,
    softmax_nodes,
    sum_nodes,
    topk_nodes,
    DGLHeteroGraph,
    to_homogeneous
)
from graph import EdgeType, HeteroWithGraphIndex, NodeType
from typing import Dict, Set
import sys
import copy
from itertools import combinations
from Config import RnnType


class AggrHGraphConvLayer(nn.Module):
    def __init__(self, out_channel, svc_feat_num, instance_feat_num, node_feat_num):
        super(AggrHGraphConvLayer, self).__init__()
        self.svc_feat_num = svc_feat_num
        self.instance_feat_num = instance_feat_num
        self.node_feat_num = node_feat_num
        self.conv = dglnn.HeteroGraphConv({
            EdgeType.SVC_CALL_EDGE.value: dglnn.GraphConv(self.svc_feat_num, out_channel),
            EdgeType.INSTANCE_NODE_EDGE.value: dglnn.GraphConv(self.instance_feat_num, out_channel),
            EdgeType.NODE_INSTANCE_EDGE.value: dglnn.GraphConv(self.node_feat_num, out_channel),
            EdgeType.INSTANCE_INSTANCE_EDGE.value: dglnn.GraphConv(self.instance_feat_num, out_channel),
            EdgeType.SVC_INSTANCE_EDGE.value: dglnn.GraphConv(self.svc_feat_num, out_channel),
            EdgeType.INSTANCE_SVC_EDGE.value: dglnn.GraphConv(self.instance_feat_num, out_channel)
        },
            aggregate='mean')
        if th.cuda.is_available():
            self.conv = self.conv.to('cpu')
        self.activation = nn.ReLU()

    def forward(self, graph: HeteroWithGraphIndex, feat_dict):
        dict = self.conv(graph.hetero_graph, feat_dict)
        node_feat = dict[NodeType.NODE.value]
        instance_feat = dict[NodeType.POD.value]
        svc_feat = dict[NodeType.SVC.value]
        return self.activation(th.cat([node_feat, instance_feat, svc_feat], dim=0))


class AggrHGraphConvWindow(nn.Module):
    def __init__(self, hidden_size, output_size, svc_feat_num, instance_feat_num, node_feat_num, num_heads: int = 2, rnn: RnnType = RnnType.LSTM):
        super(AggrHGraphConvWindow, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        if rnn == RnnType.LSTM:
            self.rnn_layer = nn.LSTM(input_size=self.hidden_size, hidden_size=self.output_size, num_layers=2,
                                     batch_first=True)
        elif rnn == RnnType.GRU:
            self.rnn_layer = nn.GRU(input_size=self.hidden_size, hidden_size=self.output_size, num_layers=2,
                                    batch_first=True)
        self.hGraph_conv_layer_list = []
        self.svc_feat_num = svc_feat_num
        self.instance_feat_num = instance_feat_num
        self.node_feat_num = node_feat_num
        self.activation = nn.ReLU()
        self.hGraph_conv_layer = AggrHGraphConvLayer(self.hidden_size, self.svc_feat_num, self.instance_feat_num,
                                                     self.node_feat_num)

    def forward(self, graph: HeteroWithGraphIndex):
        input_data_list = []

        def get_data_at_time_index(n, data):
            data_at_time_index = []
            for index in range(data.shape[0]):
                data_at_time_index.append(data[index][n])
            if th.cuda.is_available():
                return th.stack(data_at_time_index, dim=1).to('cpu').T
            else:
                return th.stack(data_at_time_index, dim=1).T

        for i in range(graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[1]):
            feat_dict = {
                NodeType.NODE.value: get_data_at_time_index(i, graph.hetero_graph.nodes[NodeType.NODE.value].data[
                    'feat']),
                NodeType.SVC.value: get_data_at_time_index(i, graph.hetero_graph.nodes[NodeType.SVC.value].data[
                    'feat']),
                NodeType.POD.value: get_data_at_time_index(i, graph.hetero_graph.nodes[NodeType.POD.value].data[
                    'feat'])
            }
            single_graph_time_series_feat = self.hGraph_conv_layer(graph, feat_dict)
            input_data_list.append(single_graph_time_series_feat.T)
        node_num = len(graph.hetero_graph_index.index[NodeType.NODE.value])
        instance_num = len(graph.hetero_graph_index.index[NodeType.POD.value])
        graph_time_series_feat = th.stack(input_data_list, dim=1).T
        hetero_graph_feat_dict = {NodeType.NODE.value: graph_time_series_feat[:node_num],
                                  NodeType.POD.value: graph_time_series_feat[node_num:node_num + instance_num],
                                  NodeType.SVC.value: graph_time_series_feat[node_num + instance_num:]}
        return hetero_graph_feat_dict, self.activation(self.rnn_layer(graph_time_series_feat)[0]), graph.center_type_name, graph.anomaly_name


class HeteroGlobalAttentionPooling(nn.Module):
    def __init__(self, gate_nn, feat_nn=None):
        super(HeteroGlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn
        self.activation = nn.Softmax(dim=0)

    def forward(self, h_graph_index: HeteroWithGraphIndex, feat_dict, get_attention=False):
        h_graph = h_graph_index.hetero_graph
        assert isinstance(h_graph, DGLHeteroGraph), "graph is not an instance of DGLHeteroGraph"
        ntypes = h_graph.ntypes
        index = {}
        count = 0
        for ntype in h_graph.ntypes:
            for key in h_graph_index.hetero_graph_index.index[ntype]:
                index[key] = h_graph_index.hetero_graph_index.index[ntype][key] + count
            count += len(h_graph_index.hetero_graph_index.index[ntype].keys())
        feat_all = th.cat([feat_dict[ntype] for ntype in ntypes], dim=0)
        graph = to_homogeneous(h_graph)
        with graph.local_scope():
            gate = self.gate_nn(feat_all)
            assert (
                    gate.shape[-1] == 1
            ), "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat_all) if self.feat_nn else feat_all

            graph.ndata["gate"] = gate
            gate = softmax_nodes(graph, "gate")
            graph.ndata.pop("gate")

            graph.ndata["r"] = feat * gate
            readout = th.sum(sum_nodes(graph, "r"), dim=1)
            graph.ndata.pop("r")
            time_series_size = gate.shape[1]
            if get_attention:
                attention_scores = torch.max(input=gate, dim=1)[0]
                return readout, index, time_series_size, attention_scores
            else:
                return readout, index, time_series_size


class AggrHGraphConvWindows(nn.Module):
    def __init__(self, out_channel, hidden_channel, svc_feat_num, instance_feat_num, node_feat_num,
                 rnn: RnnType = RnnType.LSTM):
        super(AggrHGraphConvWindows, self).__init__()
        self.hidden_size = hidden_channel
        if rnn == RnnType.LSTM:
            self.rnn_layer = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2,
                                     batch_first=True)
        elif rnn == RnnType.GRU:
            self.rnn_layer = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2,
                                    batch_first=True)
        self.graph_window_conv = AggrHGraphConvWindow(64, self.hidden_size, svc_feat_num, instance_feat_num,
                                                      node_feat_num, 2, rnn)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.output_layer = nn.Softmax(dim=0)
        self.activation = nn.ReLU()
        self.pooling = HeteroGlobalAttentionPooling(gate_nn=nn.Linear(self.hidden_size, out_channel))

    def forward(self, graphs: Dict[str, HeteroWithGraphIndex]):
        output_data_list = []
        window_graphs_center_node_index = []
        window_graphs_anomaly_node_index = []
        window_graphs_index = []
        window_time_series_sizes = []
        window_anomaly_time_series = []
        times = graphs.keys()
        times_sorted = sorted(times)
        atten_sorted = []
        for time in times_sorted:
            graph = graphs[time]
            hetero_graph_feat_dict, single_graph_window_feat, graphs_center_node_name, graphs_anomaly_node_name = self.graph_window_conv(
                graph)
            output_feat, index, time_series_size, attention_scores = self.pooling(graph, hetero_graph_feat_dict, True)
            window_time_series_sizes.append(time_series_size)
            window_anomaly_time_series.append(graph.anomaly_time_series)
            output_data_list.append(output_feat)
            window_graphs_index.append(index)
            atten_sorted.append(attention_scores)
            # convert to the index of the current graph
            graph_center_node_index = {}
            for center in graphs_center_node_name:
                if center not in graph_center_node_index:
                    graph_center_node_index[center] = {}
                graph_center_node_name = graphs_center_node_name[center]
                for node_type in graph_center_node_name:
                    if node_type not in graph_center_node_index:
                        graph_center_node_index[center][node_type] = []
                    graph_center_nodes = graph_center_node_name[node_type]
                    for graph_center_node in graph_center_nodes:
                        graph_center_node_index[center][node_type].append(index[graph_center_node])
            graphs_anomaly_node_index = {}
            for anomaly in graphs_anomaly_node_name:
                if anomaly not in graphs_anomaly_node_index:
                    graphs_anomaly_node_index[anomaly] = {}
                graph_anomaly_node_name = graphs_anomaly_node_name[anomaly]
                for node_type in graph_anomaly_node_name:
                    graph_anomaly_nodes = graph_anomaly_node_name[node_type]
                    for graph_anomaly_node in graph_anomaly_nodes:
                        is_neighbor = False
                        if 'neighbor' in graph_anomaly_node:
                            graph_anomaly_node = graph_anomaly_node[8:]
                            is_neighbor = True
                        if is_neighbor:
                            neighbors = graphs_anomaly_node_index[anomaly].get('neighbor', [])
                            neighbors.append(index[graph_anomaly_node])
                            graphs_anomaly_node_index[anomaly]['neighbor'] = neighbors
                        else:
                            graphs_anomaly_node_index[anomaly]['source'] = [index[graph_anomaly_node]]
            window_graphs_center_node_index.append(graph_center_node_index)
            window_graphs_anomaly_node_index.append(graphs_anomaly_node_index)
        output = self.activation(self.linear(self.rnn_layer(th.stack(output_data_list, dim=0))[0]))
        output = output.reshape(output.shape[0], -1)
        graphs_probability = self.output_layer(torch.sum(output, dim=1, keepdim=True))
        return [graphs_probability[g_index] * atten_sorted[g_index] for g_index in range(len(atten_sorted))], window_graphs_center_node_index, window_graphs_anomaly_node_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series


class AggrUnsupervisedGNN(nn.Module):
    def __init__(self, anomaly_index, out_channels, hidden_size, svc_feat_num, instance_feat_num, node_feat_num,
                 rnn: RnnType = RnnType.LSTM):
        super(AggrUnsupervisedGNN, self).__init__()
        self.conv = AggrHGraphConvWindows(out_channel=out_channels, hidden_channel=hidden_size,
                                          svc_feat_num=svc_feat_num, instance_feat_num=instance_feat_num,
                                          node_feat_num=node_feat_num, rnn=rnn)
        self.precessor_neighbor_weight = nn.Parameter(th.ones(1, len(anomaly_index), requires_grad=True, device='cpu'))
        self.anomaly_index = anomaly_index
        self.criterion = nn.MSELoss()

    def forward(self, graphs: Dict[str, HeteroWithGraphIndex]):
        aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series = self.conv(
            graphs)
        return aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series

    def loss(self, aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes,
             window_anomaly_time_series):
        aggr_index_combine_list = []
        for aggr in aggr_center_index:
            anomaly_index_combine = {}
            for center in aggr:
                for node_type in aggr[center]:
                    if center not in anomaly_index_combine:
                        anomaly_index_combine[center] = aggr[center][node_type]
                    else:
                        anomaly_index_combine[center].extend(aggr[center][node_type])
            aggr_index_combine_list.append(anomaly_index_combine)
        sum_criterion = 0
        for idx, anomaly_index_combine in enumerate(aggr_index_combine_list):
            aggr_feat_idx = aggr_feat[idx]
            for center in anomaly_index_combine:
                aggr_center = aggr_feat_idx[sorted(anomaly_index_combine[center])]
                aggr_mean = th.mean(aggr_center).item()
                mean = torch.full_like(aggr_center, aggr_mean)
                sum_criterion += self.criterion(aggr_center, mean)

        for idx, anomaly_index_combine in enumerate(aggr_anomaly_index):
            aggr_feat_idx = aggr_feat[idx]
            for anomaly in anomaly_index_combine:
                if len(anomaly_index_combine[anomaly]) > 0:
                    aggr_anomaly_nodes_index = anomaly_index_combine[anomaly]
                    rate = 1
                    precessor_rate = 1 * self.precessor_neighbor_weight[0][self.anomaly_index[anomaly[anomaly.find('-') + 1:]]]
                    aggr_feat_label_weight = torch.zeros_like(aggr_feat_idx)
                    source_index_matrix = torch.tensor(aggr_anomaly_nodes_index['source'])
                    aggr_feat_label_weight[source_index_matrix] = rate
                    if 'neighbor' in aggr_anomaly_nodes_index:
                        neighbor_index_matrix = torch.tensor(aggr_anomaly_nodes_index['neighbor'])
                        aggr_feat_label_weight[neighbor_index_matrix] = precessor_rate
                    sum_criterion += self.criterion(aggr_feat_idx, aggr_feat_label_weight)
        return sum_criterion
