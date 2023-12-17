import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from graph import EdgeType, HeteroWithGraphIndex, NodeType
from typing import Dict, Set
import sys
import copy


class AggrHGraphConvLayer(nn.Module):
    def __init__(self, hidden_size, out_channel, svc_feat_num, instance_feat_num, node_feat_num,
                 svc_num, instance_num, node_num, total_node_num):
        super(AggrHGraphConvLayer, self).__init__()
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
            EdgeType.NODE_INSTANCE_EDGE.value: dglnn.GraphConv(self.node_feat_num, hidden_size)},
            aggregate='sum')
        self.linear_map = {}
        self.linear_map[NodeType.NODE.value] = nn.Linear(self.node_num, out_channel)
        self.linear_map[NodeType.SVC.value] = nn.Linear(self.svc_num, out_channel)
        self.linear_map[NodeType.POD.value] = nn.Linear(self.instance_num, out_channel)
        self.linear_map['total'] = nn.Linear(self.total_node_num, out_channel)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, graph: HeteroWithGraphIndex, feat_dict):
        dict = self.conv(graph.hetero_graph, feat_dict)
        node_feat = self.activation(dict[NodeType.NODE.value])
        instance_feat = self.activation(dict[NodeType.POD.value])
        svc_feat = self.activation(dict[NodeType.SVC.value])
        return th.cat([node_feat, instance_feat, svc_feat], dim=0)
        # # The input is a dictionary of node features for each type
        # funcs = {}
        # for srctype, etype, dsttype in G.canonical_etypes:
        #     # 计算每一类etype的 W_r * h
        #     Wh = self.weight[etype](feat_dict[srctype])
        #     # Save it in graph for message passing
        #     G.nodes[srctype].data['Wh_%s' % etype] = Wh
        #     # 消息函数 copy_u: 将源节点的特征聚合到'm'中; reduce函数: 将'm'求均值赋值给 'h'
        #     funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # # Trigger message passing of multiple types.
        # # The first argument is the message passing functions for each relation.
        # # The second one is the type wise reducer, could be "sum", "max",
        # # "min", "mean", "stack"
        # G.multi_update_all(funcs, 'sum')
        # # return the updated node feature dictionary
        # return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}


class AggrHGraphConvWindow(nn.Module):
    def __init__(self, input_size, hidden_size, graph: HeteroWithGraphIndex):
        super(AggrHGraphConvWindow, self).__init__()
        self.graph = graph
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.window_simple_size = graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[1]
        self.lstm_layer = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2,
                                  batch_first=True)
        self.hGraph_conv_layer_list = []
        self.svc_feat_num = graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[2]
        self.instance_feat_num = graph.hetero_graph.nodes[NodeType.POD.value].data['feat'].shape[2]
        self.node_feat_num = graph.hetero_graph.nodes[NodeType.NODE.value].data['feat'].shape[2]
        self.svc_num = graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[0]
        self.instance_num = graph.hetero_graph.nodes[NodeType.POD.value].data['feat'].shape[0]
        self.node_num = graph.hetero_graph.nodes[NodeType.NODE.value].data['feat'].shape[0]
        for i in range(self.window_simple_size):
            self.hGraph_conv_layer_list.append(
                AggrHGraphConvLayer(self.input_size, self.hidden_size, self.svc_feat_num, self.instance_feat_num,
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
            single_graph_feat = layer(self.graph, feat_dict)
            input_data_list.append(single_graph_feat.T)
        center_node_index: Dict[str, Set[str]] = {}
        graphs_anomaly_node_index = {}
        for center in self.graph.center_type_index:
            center_node_index[center] = self.graph.center_type_index[center][NodeType.NODE.value]
        graphs_center_node_index = copy.deepcopy(center_node_index)
        node_num = len(self.graph.hetero_graph_index.index[NodeType.NODE.value])
        instance_num = len(self.graph.hetero_graph_index.index[NodeType.POD.value])

        # todo 以异常为中心，前继传播

        def graph_anomaly_index(graphs_anomaly_node_index, anomaly, anomaly_map ,node_num, instance_num):
            for node_type in anomaly_map:
                graph_index_by_anomaly = graphs_anomaly_node_index.get(anomaly, [])
                if node_type == NodeType.NODE.value:
                    graph_index_by_anomaly.extend(node_index for node_index in anomaly_map[node_type])
                elif node_type == NodeType.POD.value:
                    graph_index_by_anomaly.extend(pod_index + node_num for pod_index in anomaly_map[node_type])
                elif node_type == NodeType.SVC.value:
                    graph_index_by_anomaly.extend(
                        svc_index + node_num + instance_num for svc_index in anomaly_map[node_type])
                graphs_anomaly_node_index[anomaly] = graph_index_by_anomaly

        for anomaly in self.graph.anomaly_index:
            graph_anomaly_index(graphs_anomaly_node_index, anomaly, self.graph.anomaly_index[anomaly], node_num, instance_num)
        return self.lstm_layer(th.stack(input_data_list, dim=1).T)[0], graphs_center_node_index, graphs_anomaly_node_index


class AggrHGraphConvWindows(nn.Module):
    def __init__(self, out_channel, hidden_channel, graphs: Dict[str, HeteroWithGraphIndex]):
        super(AggrHGraphConvWindows, self).__init__()
        self.hidden_size = hidden_channel
        self.window_size = len(graphs)
        # todo 结合图结构时序/指标时序特征
        time_sorted = sorted(graphs.keys())
        self.lstm_layer = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2,
                                  batch_first=True)
        self.time_metrics_layer_list = []
        for time in time_sorted:
            self.time_metrics_layer_list.append(AggrHGraphConvWindow(32, self.hidden_size, graphs[time]))
        self.linear1 = nn.Linear(self.hidden_size, out_channel)

    def forward(self):
        input_data_list = []
        window_graphs_center_node_index = []
        window_graphs_anomaly_node_index = []
        shortest_time = sys.maxsize
        node_num = [0] * len(self.time_metrics_layer_list)
        node_index = [0] * len(self.time_metrics_layer_list)
        for index, layer in enumerate(self.time_metrics_layer_list):
            single_window_feat, graphs_center_node_index, graphs_anomaly_node_index = layer()
            input_data_list.append(single_window_feat)
            if single_window_feat.shape[1] < shortest_time:
                shortest_time = single_window_feat.shape[1]
            window_graphs_center_node_index.append(graphs_center_node_index)
            window_graphs_anomaly_node_index.append(graphs_anomaly_node_index)
            if index > 0:
                node_index[index] = single_window_feat.shape[0] + node_index[index - 1]
                node_num[index] = node_index[index - 1]
            else:
                node_num[index] = 0
                node_index[index] = single_window_feat.shape[0]
        # todo 默认取前shortest_time时序位，有可能丢失重要特征，可以使用时间窗口平滑
        for index, input_data in enumerate(input_data_list):
            input_data_list[index] = input_data[:, :shortest_time, :]
        for index, center_index_map in enumerate(window_graphs_center_node_index):
            append_index = node_num[index]
            for center in center_index_map:
                for idx, idx_origin in enumerate(center_index_map[center]):
                    center_index_map[center][idx] = idx_origin + append_index
        for index, anomaly_dict in enumerate(window_graphs_anomaly_node_index):
            for anomaly in anomaly_dict:
                anomaly_dict[anomaly] = [idx + node_num[index] for idx in anomaly_dict[anomaly]]
            window_graphs_anomaly_node_index[index] = anomaly_dict
        window_graphs_anomaly_node_index_combine = {}
        for d in window_graphs_anomaly_node_index:
            for ano, idx_list in d.items():
                if ano in window_graphs_anomaly_node_index_combine:
                    window_graphs_anomaly_node_index_combine[ano].extend(idx_list)
                else:
                    window_graphs_anomaly_node_index_combine[ano] = idx_list
        return self.linear1(self.lstm_layer(th.cat(input_data_list, dim=0))[
                                0]), window_graphs_center_node_index, window_graphs_anomaly_node_index_combine


class AggrUnsupervisedGNN(nn.Module):
    def __init__(self, out_channels, hidden_size, graphs: Dict[str, HeteroWithGraphIndex]):
        super(AggrUnsupervisedGNN, self).__init__()
        self.conv = AggrHGraphConvWindows(out_channel=out_channels, hidden_channel=hidden_size, graphs=graphs)
        self.criterion = nn.MSELoss()

    def forward(self):
        x, aggr_center_index, aggr_anomaly_index = self.conv()
        return x, aggr_center_index, aggr_anomaly_index

    def loss(self, aggr_feat, aggr_center_index, aggr_anomaly_index):
        aggr_index_combine = {}
        for aggr in aggr_center_index:
            for center in aggr:
                if center not in aggr_index_combine:
                    aggr_index_combine[center] = aggr[center]
                else:
                    aggr_index_combine[center].extend(aggr[center])
        sum_criterion = 0
        for center in aggr_index_combine:
            aggr_center = aggr_feat[aggr_index_combine[center]]
            sum_criterion += self.criterion(aggr_center, torch.zeros_like(aggr_center))
        for anomaly in aggr_anomaly_index:
            if len(anomaly) > 0:
                aggr_anomaly = aggr_feat[aggr_anomaly_index[anomaly]]
                sum_criterion += self.criterion(aggr_anomaly, torch.zeros_like(aggr_anomaly))
        return sum_criterion
