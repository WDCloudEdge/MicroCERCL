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
            EdgeType.INSTANCE_INSTANCE_EDGE.value: dglnn.GraphConv(self.instance_feat_num, out_channel)
        },
            aggregate='mean')
        if th.cuda.is_available():
            self.conv = self.conv.to('cuda')
        self.activation = nn.LeakyReLU()

    def forward(self, graph: HeteroWithGraphIndex, feat_dict):
        dict = self.conv(graph.hetero_graph, feat_dict)
        node_feat = dict[NodeType.NODE.value]
        instance_feat = dict[NodeType.POD.value]
        svc_feat = dict[NodeType.SVC.value]
        return self.activation(th.cat([node_feat, instance_feat, svc_feat], dim=0))
        # dict[NodeType.NODE.value] = self.activation(dict[NodeType.NODE.value])
        # dict[NodeType.POD.value] = self.activation(dict[NodeType.POD.value])
        # dict[NodeType.SVC.value] = self.activation(dict[NodeType.SVC.value])
        # return dict
        # return th.cat([node_feat, instance_feat, svc_feat], dim=0)


class AggrHGraphConvWindow(nn.Module):
    def __init__(self, hidden_size, output_size, svc_feat_num, instance_feat_num, node_feat_num, num_heads: int = 2,
                 is_attention: bool = False, rnn: RnnType = RnnType.LSTM):
        super(AggrHGraphConvWindow, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.window_simple_size = graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[1]
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
        self.is_attention = is_attention
        # self.activation = nn.LeakyReLU(negative_slope=1e-2)
        if is_attention:
            self.num_heads = num_heads
            self.attention = nn.MultiheadAttention(self.hidden_size, num_heads)
        self.hGraph_conv_layer = AggrHGraphConvLayer(self.hidden_size, self.svc_feat_num, self.instance_feat_num,
                                                     self.node_feat_num)
        # for i in range(self.window_simple_size):
        #     self.hGraph_conv_layer_list.append(
        #         AggrHGraphConvLayer(self.hidden_size, self.svc_feat_num, self.instance_feat_num,
        #                             self.node_feat_num))

    def forward(self, graph: HeteroWithGraphIndex):
        input_data_list = []

        def get_data_at_time_index(n, data):
            data_at_time_index = []
            for index in range(data.shape[0]):
                data_at_time_index.append(data[index][n])
            if th.cuda.is_available():
                return th.stack(data_at_time_index, dim=1).to('cuda').T
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
        # 时序attention，单个时间窗口内时序前者对所有时序后者的影响
        if self.is_attention:
            pass
            # batch_sequence_list = []
            # combinations_numbers = [list(combination) for combination in
            #                         combinations(range(self.window_simple_size), self.num_heads)]
            # for combination in combinations_numbers:
            #     batch_sequence_list.append([input_data_list[idx] for idx in combination])
            # for idx, batch_sequence in enumerate(batch_sequence_list):
            #     # batch_sequence[1] = self.attention(batch_sequence[0].T, batch_sequence[0].T, batch_sequence[1].T)[0].T
            #     input_data_list[combinations_numbers[idx][1]] = self.attention(batch_sequence[0].T, batch_sequence[0].T, batch_sequence[1].T)[0].T
        center_node_index: Dict[str, Set[str]] = {}
        graph_anomaly_node_index = {}
        for center in graph.center_type_index:
            center_node_index[center] = graph.center_type_index[center][NodeType.NODE.value]
        graph_center_node_index = copy.deepcopy(center_node_index)
        node_num = len(graph.hetero_graph_index.index[NodeType.NODE.value])
        instance_num = len(graph.hetero_graph_index.index[NodeType.POD.value])

        def graph_anomaly_index(graph_anomaly_node_index, anomaly, anomaly_map, node_num, instance_num):
            for node_type in anomaly_map:
                graph_index_by_anomaly = graph_anomaly_node_index.get(anomaly, [])
                if node_type == NodeType.NODE.value:
                    graph_index_by_anomaly.extend(node_index for node_index in anomaly_map[node_type])
                elif node_type == NodeType.POD.value:
                    graph_index_by_anomaly.extend(pod_index + node_num for pod_index in anomaly_map[node_type])
                elif node_type == NodeType.SVC.value:
                    graph_index_by_anomaly.extend(
                        svc_index + node_num + instance_num for svc_index in anomaly_map[node_type])
                graph_anomaly_node_index[anomaly] = graph_index_by_anomaly

        for anomaly in graph.anomaly_index:
            graph_anomaly_index(graph_anomaly_node_index, anomaly, graph.anomaly_index[anomaly], node_num,
                                instance_num)
        # return self.rnn_layer(th.stack(input_data_list, dim=1).T)[
        #     0], graphs_center_node_index, graphs_anomaly_node_index
        graph_time_series_feat = th.stack(input_data_list, dim=1).T
        hetero_graph_feat_dict = {NodeType.NODE.value: graph_time_series_feat[:node_num],
                                  NodeType.POD.value: graph_time_series_feat[node_num:node_num + instance_num],
                                  NodeType.SVC.value: graph_time_series_feat[node_num + instance_num:]}
        return hetero_graph_feat_dict, graph_time_series_feat, graph.center_type_name, graph.anomaly_name


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
                attention_scores = th.sum(gate, dim=-1)
                attention_scores[index[next(iter(h_graph_index.hetero_graph_index.node_index))]] = 0
                return readout, index, time_series_size, self.activation(attention_scores.view(-1)).view(attention_scores.shape[0], attention_scores.shape[1])
                # return readout, index, time_series_size, attention_scores
            else:
                return readout, index, time_series_size


class AggrHGraphConvWindows(nn.Module):
    def __init__(self, out_channel, hidden_channel, svc_feat_num, instance_feat_num, node_feat_num,
                 rnn: RnnType = RnnType.LSTM, attention: bool = False):
        super(AggrHGraphConvWindows, self).__init__()
        self.hidden_size = hidden_channel
        if rnn == RnnType.LSTM:
            self.rnn_layer = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2,
                                     batch_first=True)
        elif rnn == RnnType.GRU:
            self.rnn_layer = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2,
                                    batch_first=True)
        # 先尝试使用同一个图时序卷积网络对不同的衍化图作训练
        self.graph_window_conv = AggrHGraphConvWindow(64, self.hidden_size, svc_feat_num, instance_feat_num,
                                                      node_feat_num, 2, attention, rnn)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.output_layer = nn.Softmax(dim=0)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        # self.pooling = dglnn.GlobalAttentionPooling(gate_nn=nn.Linear(self.hidden_size, out_channel))
        self.pooling = HeteroGlobalAttentionPooling(gate_nn=nn.Linear(self.hidden_size, out_channel))

    def forward(self, graphs: Dict[str, HeteroWithGraphIndex]):
        output_data_list = []
        window_graphs_center_node_index = []
        window_graphs_anomaly_node_index = []
        window_graphs_index = []
        window_time_series_sizes = []
        window_anomaly_time_series = []
        # node_num = [0] * len(self.time_metrics_layer_list)
        # node_index = [0] * len(self.time_metrics_layer_list)
        times = graphs.keys()
        times_sorted = sorted(times)
        atten_sorted = []
        for time in times_sorted:
            graph = graphs[time]
            hetero_graph_feat_dict, single_graph_window_feat, graphs_center_node_name, graphs_anomaly_node_name = self.graph_window_conv(
                graph)
            # input_data_list.append(single_graph_window_feat)
            # if single_graph_window_feat.shape[1] < shortest_time:
            #     shortest_time = single_graph_window_feat.shape[1]
            output_feat, index, time_series_size, attention_scores = self.pooling(graph, hetero_graph_feat_dict, True)
            window_time_series_sizes.append(time_series_size)
            window_anomaly_time_series.append(graph.anomaly_time_series)
            output_data_list.append(output_feat)
            window_graphs_index.append(index)
            atten_sorted.append(attention_scores)
            # 转换为当前图的索引
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
                    graphs_anomaly_node_index[anomaly] = []
                graph_anomaly_node_name = graphs_anomaly_node_name[anomaly]
                for node_type in graph_anomaly_node_name:
                    graph_anomaly_nodes = graph_anomaly_node_name[node_type]
                    for graph_anomaly_node in graph_anomaly_nodes:
                        graphs_anomaly_node_index[anomaly].append(index[graph_anomaly_node])
            window_graphs_center_node_index.append(graph_center_node_index)
            window_graphs_anomaly_node_index.append(graphs_anomaly_node_index)
            # if index > 0:
            #     node_index[index] = single_window_feat.shape[0] + node_index[index - 1]
            #     node_num[index] = node_index[index - 1]
            # else:
            #     node_num[index] = 0
            #     node_index[index] = single_window_feat.shape[0]
        # todo 默认取前shortest_time时序位，有可能丢失重要特征，可以使用时间窗口平滑
        # for index, input_data in enumerate(input_data_list):
        #     input_data_list[index] = input_data[:, :shortest_time, :]
        # for index, center_index_map in enumerate(window_graphs_center_node_index):
        #     append_index = node_num[index]
        #     for center in center_index_map:
        #         for idx, idx_origin in enumerate(center_index_map[center]):
        #             center_index_map[center][idx] = idx_origin + append_index
        # for index, anomaly_dict in enumerate(window_graphs_anomaly_node_index):
        #     for anomaly in anomaly_dict:
        #         anomaly_dict[anomaly] = [idx + node_num[index] for idx in anomaly_dict[anomaly]]
        #     window_graphs_anomaly_node_index[index] = anomaly_dict
        # window_graphs_anomaly_node_index_combine = {}
        # for d in window_graphs_anomaly_node_index:
        #     for ano, idx_list in d.items():
        #         if ano in window_graphs_anomaly_node_index_combine:
        #             window_graphs_anomaly_node_index_combine[ano].extend(idx_list)
        #         else:
        #             window_graphs_anomaly_node_index_combine[ano] = idx_list
        # output = self.linear(self.rnn_layer(self.activation(th.cat(input_data_list, dim=0)))[0])
        output = self.linear(self.activation(self.rnn_layer(self.activation(th.stack(output_data_list, dim=0)))[0]))
        # output = self.linear(self.rnn_layer(th.stack(output_data_list, dim=0))[0])
        output = output.reshape(output.shape[0], -1)
        return self.output_layer(torch.sum(output, dim=1,
                         keepdim=True)).T * th.stack(atten_sorted, dim=0), window_graphs_center_node_index, window_graphs_anomaly_node_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series


class AggrUnsupervisedGNN(nn.Module):
    def __init__(self, out_channels, hidden_size, svc_feat_num, instance_feat_num, node_feat_num,
                 rnn: RnnType = RnnType.LSTM,
                 attention: bool = False):
        super(AggrUnsupervisedGNN, self).__init__()
        self.conv = AggrHGraphConvWindows(out_channel=out_channels, hidden_channel=hidden_size,
                                          svc_feat_num=svc_feat_num, instance_feat_num=instance_feat_num,
                                          node_feat_num=node_feat_num, rnn=rnn,
                                          attention=attention)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, graphs: Dict[str, HeteroWithGraphIndex]):
        aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series = self.conv(graphs)
        return aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series

    def loss(self, aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series):
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
                mean[22] = 0
                sum_criterion += self.criterion(aggr_center, mean)

        # anomaly_aggr_index_combine_list = []
        # for aggr_anomaly in aggr_anomaly_index:
        #     aggr_index_combine = {}
        #     for anomaly in aggr_anomaly:
        #         for node_type in aggr_anomaly[anomaly]:
        #             if anomaly not in aggr_index_combine:
        #                 aggr_index_combine[anomaly] = aggr_anomaly[anomaly][node_type]
        #             else:
        #                 aggr_index_combine[anomaly].extend(aggr_anomaly[anomaly][node_type])
        #     anomaly_aggr_index_combine_list.append(aggr_index_combine)
        for idx, anomaly_index_combine in enumerate(aggr_anomaly_index):
            aggr_feat_idx = aggr_feat[idx]
            for anomaly in anomaly_index_combine:
                if len(anomaly_index_combine[anomaly]) > 0:
                    aggr_anomaly_nodes_index = sorted(anomaly_index_combine[anomaly])
                    # aggr_anomaly_index = window_graphs_index[idx][anomaly[anomaly.find('-') + 1:]]
                    aggr_anomaly_time_series_index = window_anomaly_time_series[idx][anomaly[anomaly.find('-') + 1:]]
                    rate = 1 / len(aggr_anomaly_nodes_index) / len(aggr_anomaly_time_series_index)
                    # rate = aggr_feat_idx.max().item()
                    aggr_anomaly_time_series_index_map = {}
                    for time_series_index in aggr_anomaly_time_series_index:
                        count = aggr_anomaly_time_series_index_map.get(time_series_index, 0)
                        count += 1
                        aggr_anomaly_time_series_index_map[time_series_index] = count
                    aggr_feat_label = torch.zeros_like(aggr_feat_idx)
                    # aggr_feat_label = aggr_feat_idx.clone()
                    aggr_feat_label_weight = torch.zeros_like(aggr_feat_idx)
                    index_matrix = torch.cartesian_prod(torch.tensor(aggr_anomaly_nodes_index), torch.tensor(aggr_anomaly_time_series_index))
                    rows, cols = index_matrix.shape
                    for i in range(rows):
                        element = index_matrix[i]
                        count = aggr_anomaly_time_series_index_map[element[1].item()]
                        aggr_feat_label_weight[element[0], element[1]] = count
                    aggr_feat_label[index_matrix[:, 0], index_matrix[:, 1]] = rate
                    # aggr_feat_label[index_matrix[:, 0], index_matrix[:, 1]] = 0
                    # sum_criterion += self.criterion(aggr_feat_idx, aggr_feat_label * aggr_feat_label_weight)
                    sum_criterion += self.criterion(aggr_feat_idx, aggr_feat_label * aggr_feat_label_weight)
                    # sum_criterion += self.criterion(aggr_feat_idx, aggr_feat_label)
        return sum_criterion
