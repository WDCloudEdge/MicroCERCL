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
from model_attention import AttentionLayer
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
    def __init__(self, hidden_size, output_size, svc_feat_num, instance_feat_num, node_feat_num,
                 rnn: RnnType = RnnType.LSTM):
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
        feat_dict = {
            NodeType.NODE.value: graph.hetero_graph.nodes[NodeType.NODE.value].data[
                'feat'],
            NodeType.SVC.value: graph.hetero_graph.nodes[NodeType.SVC.value].data[
                'feat'],
            NodeType.POD.value: graph.hetero_graph.nodes[NodeType.POD.value].data[
                'feat'],
        }
        graph_time_series_feat = self.hGraph_conv_layer(graph, feat_dict)
        node_num = len(graph.hetero_graph_index.index[NodeType.NODE.value])
        instance_num = len(graph.hetero_graph_index.index[NodeType.POD.value])
        hetero_graph_feat_dict = {NodeType.NODE.value: graph_time_series_feat[:node_num],
                                  NodeType.POD.value: graph_time_series_feat[node_num:node_num + instance_num],
                                  NodeType.SVC.value: graph_time_series_feat[node_num + instance_num:]}
        return hetero_graph_feat_dict, self.activation(
            self.rnn_layer(graph_time_series_feat)[0]), graph.center_type_name, graph.anomaly_name


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
            # assert (
            #         gate.shape[-1] == 1
            # ), "The output of gate_nn should have size 1 at the last axis."
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
                # attention_scores = gate
                return readout, index, time_series_size, attention_scores
            else:
                return readout, index, time_series_size


class AggrHGraphConvWindows(nn.Module):
    def __init__(self, out_channel, hidden_channel, svc_feat_num, instance_feat_num, node_feat_num,
                 rnn: RnnType = RnnType.LSTM):
        super(AggrHGraphConvWindows, self).__init__()
        self.hidden_size = hidden_channel
        self.out_size = out_channel
        if rnn == RnnType.LSTM:
            self.rnn_layer = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2,
                                     batch_first=True)
        elif rnn == RnnType.GRU:
            self.rnn_layer = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2,
                                    batch_first=True)
        self.graph_window_conv = AggrHGraphConvWindow(64, self.hidden_size, svc_feat_num, instance_feat_num,
                                                      node_feat_num, rnn)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.output_layer = nn.Softmax(dim=0)
        self.activation = nn.ReLU()
        # self.pooling = HeteroGlobalAttentionPooling(gate_nn=nn.Linear(self.hidden_size, out_channel))
        self.pooling = HeteroGlobalAttentionPooling(gate_nn=nn.Linear(self.hidden_size, hidden_channel))
        self.center_attention = AttentionLayer(hidden_channel, out_channel, num_heads=1)
        self.node_attention = AttentionLayer(hidden_channel, out_channel, num_heads=1)

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
                anomaly_n = anomaly[anomaly.find('$') + 1:]
                if anomaly_n not in graph.node_exist:
                    continue
                if anomaly not in graphs_anomaly_node_index:
                    graphs_anomaly_node_index[anomaly] = {}
                graph_anomaly_node_name = graphs_anomaly_node_name[anomaly]
                for node_type in graph_anomaly_node_name:
                    graph_anomaly_nodes = graph_anomaly_node_name[node_type]
                    for graph_anomaly_node in graph_anomaly_nodes:
                        is_neighbor = 'neighbor' in graph_anomaly_node
                        if is_neighbor:
                            center = graph_anomaly_node[9:][:graph_anomaly_node[9:].find('$')]
                        else:
                            center = graph_anomaly_node[:graph_anomaly_node.find('$')]
                        graph_anomaly_node = graph_anomaly_node[graph_anomaly_node.rfind('$') + 1:]
                        if 'neighbor' not in graphs_anomaly_node_index[anomaly]:
                            graphs_anomaly_node_index[anomaly]['neighbor'] = {}
                        if is_neighbor:
                            neighbors_type = graphs_anomaly_node_index[anomaly]['neighbor'].get(center, [])
                            neighbors_type.append(index[graph_anomaly_node])
                            graphs_anomaly_node_index[anomaly]['neighbor'][center] = neighbors_type
                        else:
                            graphs_anomaly_node_index[anomaly]['source'] = [index[graph_anomaly_node]]
            # Apply center attention
            attention_scores_after_center = th.zeros([attention_scores.shape[0], self.out_size])
            center_embeddings = []
            for center in graph_center_node_index:
                center_nodes_index = []
                for _, nodes_index in graph_center_node_index[center].items():
                    center_nodes_index.extend(nodes_index)
                aggr_center = th.mean(attention_scores[sorted(center_nodes_index)], dim=0, keepdim=True)
                center_embeddings.append(aggr_center)
            center_embeddings = th.cat(center_embeddings, dim=0)
            aggr_feat_weighted, attention_weights_center = self.center_attention(center_embeddings, center_embeddings,
                                                                                 center_embeddings)
            for i, center in enumerate(graph_center_node_index):
                center_nodes_index = []
                for _, nodes_index in graph_center_node_index[center].items():
                    center_nodes_index.extend(nodes_index)
                attention_scores_after_center[center_nodes_index] = th.max(attention_scores[sorted(center_nodes_index)] * aggr_feat_weighted[i], dim=1)[0].unsqueeze(-1)
            atten_sorted.append(attention_scores_after_center)
            window_graphs_center_node_index.append(graph_center_node_index)
            window_graphs_anomaly_node_index.append(graphs_anomaly_node_index)
        output = self.activation(self.linear(self.rnn_layer(th.stack(output_data_list, dim=0))[0]))
        output = output.reshape(output.shape[0], -1)
        graphs_probability = self.output_layer(torch.sum(output, dim=1, keepdim=True))
        return [graphs_probability[g_index] * atten_sorted[g_index] for g_index in range(
            len(atten_sorted))], window_graphs_center_node_index, window_graphs_anomaly_node_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series


class AggrUnsupervisedGNN(nn.Module):
    def __init__(self, sorted_graphs, center_map, anomaly_index, out_channels, hidden_size, svc_feat_num,
                 instance_feat_num, node_feat_num,
                 rnn: RnnType = RnnType.LSTM):
        super(AggrUnsupervisedGNN, self).__init__()
        self.conv = AggrHGraphConvWindows(out_channel=out_channels, hidden_channel=hidden_size,
                                          svc_feat_num=svc_feat_num, instance_feat_num=instance_feat_num,
                                          node_feat_num=node_feat_num, rnn=rnn)
        # self.precessor_neighbor_center_weight = nn.Parameter(
        #     th.ones(len(anomaly_index), len(list(center_map.keys())), requires_grad=True, device='cpu'))
        anomaly_index_reverse = {idx: an for an, idx in anomaly_index.items()}
        anomaly_nodes_maps = [sorted_graph.anomaly_name for sorted_graph in sorted_graphs]
        self.graphs_anomaly_center_nodes = []
        for graph_idx in range(len(anomaly_nodes_maps)):
            graph_anomaly_center_nodess = {}
            anomaly_nodes_map = anomaly_nodes_maps[graph_idx]
            for i in range(len(anomaly_index)):
                ano = anomaly_index_reverse[i]
                graph_anomaly_center_nodes = {}
                if ano in anomaly_nodes_map:
                    for node_type in anomaly_nodes_map[ano]:
                        graph_anomaly_nodes = anomaly_nodes_map[ano][node_type]
                        for graph_anomaly_node in graph_anomaly_nodes:
                            is_neighbor = 'neighbor' in graph_anomaly_node
                            if is_neighbor:
                                center = graph_anomaly_node[9:][:graph_anomaly_node[9:].find('$')]
                            else:
                                continue
                            graph_anomaly_node = graph_anomaly_node[graph_anomaly_node.rfind('$') + 1:]
                            if center not in graph_anomaly_center_nodes:
                                graph_anomaly_center_nodes[center] = []
                            if is_neighbor:
                                neighbors_type = graph_anomaly_center_nodes.get(center, [])
                                neighbors_type.append(graph_anomaly_node)
                                graph_anomaly_center_nodes[center] = neighbors_type
                    graph_anomaly_center_nodess[ano] = graph_anomaly_center_nodes
            self.graphs_anomaly_center_nodes.append(graph_anomaly_center_nodess)

        class ParameterWrapper(nn.Module):
            def __init__(self, size, device):
                super(ParameterWrapper, self).__init__()
                self.param = nn.Parameter(th.ones(size, requires_grad=True, device=device))

            def forward(self):
                return self.param

        self.precessor_neighbor_node_weight = nn.ModuleList()
        for graph_anomaly_nodess in self.graphs_anomaly_center_nodes:
            graph_anomalies_weight = nn.ModuleDict()
            for a, graph_anomaly_center in graph_anomaly_nodess.items():
                if a not in graph_anomalies_weight:
                    graph_anomalies_weight[a] = nn.ModuleDict()
                for center in graph_anomaly_center:
                    graph_anomalies_weight[a][center] = ParameterWrapper(
                        size=len(graph_anomaly_center[center]),
                        device='cpu'
                    )
            self.precessor_neighbor_node_weight.append(graph_anomalies_weight)
        self.anomaly_index = anomaly_index
        self.center_map = center_map
        self.criterion = nn.MSELoss()

    def forward(self, graphs: Dict[str, HeteroWithGraphIndex]):
        aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series = self.conv(
            graphs)
        return aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series

    def loss(self, aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes,
             window_anomaly_time_series):
        aggr_index_combine_list = []
        # for aggr in aggr_center_index:
        #     anomaly_index_combine = {}
        #     for center in aggr:
        #         for node_type in aggr[center]:
        #             if center not in anomaly_index_combine:
        #                 anomaly_index_combine[center] = aggr[center][node_type]
        #             else:
        #                 anomaly_index_combine[center].extend(aggr[center][node_type])
        #     aggr_index_combine_list.append(anomaly_index_combine)
        sum_criterion = 0
        # for idx, anomaly_index_combine in enumerate(aggr_index_combine_list):
        #     aggr_feat_idx = aggr_feat[idx]
        #     for center in anomaly_index_combine:
        #         aggr_center = aggr_feat_idx[sorted(anomaly_index_combine[center])]
        #         aggr_mean = th.mean(aggr_center).item()
        #         mean = torch.full_like(aggr_center, aggr_mean)
        #         sum_criterion += self.criterion(aggr_center, mean)

        for idx, anomaly_index_combine in enumerate(aggr_anomaly_index):
            aggr_feat_idx = aggr_feat[idx]
            graph_anomaly_center_nodes_weight = self.precessor_neighbor_node_weight[idx]
            for anomaly in anomaly_index_combine:
                if len(anomaly_index_combine[anomaly]) > 0:
                    anomaly_graph_anomaly_center_nodes_weight = graph_anomaly_center_nodes_weight[anomaly]
                    aggr_anomaly_nodes_index = anomaly_index_combine[anomaly]
                    rate = 1
                    aggr_feat_label_weight = torch.zeros_like(aggr_feat_idx)
                    source_index_matrix = torch.tensor(aggr_anomaly_nodes_index['source'])
                    aggr_feat_label_weight[source_index_matrix] = rate
                    if 'neighbor' in aggr_anomaly_nodes_index:
                        for center in aggr_anomaly_nodes_index['neighbor']:
                            for ano_idx_idx, ano_idx in enumerate(aggr_anomaly_nodes_index['neighbor'][center]):
                                center_node_weight = anomaly_graph_anomaly_center_nodes_weight[center]
                                # precessor_rate = 1 * self.precessor_neighbor_center_weight[self.anomaly_index[anomaly]][
                                #     self.center_map[center]] * center_node_weight()[ano_idx_idx]
                                precessor_rate = 1 * center_node_weight()[ano_idx_idx]
                                neighbor_index_matrix = torch.tensor(
                                    aggr_anomaly_nodes_index['neighbor'][center][ano_idx_idx])
                                aggr_feat_label_weight[neighbor_index_matrix] = precessor_rate
                    sum_criterion += self.criterion(aggr_feat_idx, aggr_feat_label_weight)
        return sum_criterion
