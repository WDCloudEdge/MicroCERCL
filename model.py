import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from graph import EdgeType, HeteroWithGraphIndex, NodeType
from typing import Dict


class HGraphConvLayer(nn.Module):
    def __init__(self, hidden_size, out_channel, svc_feat_num, instance_feat_num, node_feat_num):
        super(HGraphConvLayer, self).__init__()
        self.svc_feat_num = svc_feat_num
        self.instance_feat_num = instance_feat_num
        self.node_feat_num = node_feat_num
        self.conv = dglnn.HeteroGraphConv({
            EdgeType.SVC_CALL_EDGE.value: dglnn.GraphConv(self.svc_feat_num, hidden_size),
            EdgeType.INSTANCE_NODE_EDGE.value: dglnn.GraphConv(self.instance_feat_num, hidden_size),
            EdgeType.NODE_INSTANCE_EDGE.value: dglnn.GraphConv(self.node_feat_num, hidden_size)},
            aggregate='sum')
        self.linear_map = {}
        self.linear_map[NodeType.NODE.value] = nn.Linear(self.node_feat_num, out_channel)
        self.linear_map[NodeType.SVC.value] = nn.Linear(self.svc_feat_num, out_channel)
        self.linear_map[NodeType.POD.value] = nn.Linear(self.instance_feat_num, out_channel)

    def forward(self, graph, feat_dict):
        dict = self.conv(graph, feat_dict)
        return torch.mean([self.linear_map[key](nn.LeakyReLU(negative_slope=0.01)(dict[key].T)) for key in dict])
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


class HGraphConvWindow(nn.Module):
    def __init__(self, hidden_size, graph: HeteroWithGraphIndex):
        super(HGraphConvWindow, self).__init__()
        self.graph = graph
        self.hidden_size = hidden_size
        self.window_simple_size = graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[1]
        self.lstm_layer = nn.LSTM(input_size=self.window_simple_size, hidden_size=self.hidden_size, num_layers=2, batch_first=False)
        self.hGraph_conv_layer_list = []
        self.svc_feat_num = graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[2]
        self.instance_feat_num = graph.hetero_graph.nodes[NodeType.POD.value].data['feat'].shape[2]
        self.node_feat_num = graph.hetero_graph.nodes[NodeType.NODE.value].data['feat'].shape[2]
        for i in range(self.window_simple_size):
            self.hGraph_conv_layer_list.append(HGraphConvLayer(self.hidden_size, self.svc_feat_num, self.instance_feat_num, self.node_feat_num))
        # self.conv = dglnn.HeteroGraphConv({
        #     EdgeType.SVC_CALL.value: dglnn.GraphConv(self.svc_feat_num, 64),
        #     EdgeType.INSTANCE_NODE.value: dglnn.GraphConv(self.instance_feat_num, 64),
        #     EdgeType.NODE_INSTANCE.value: dglnn.SAGEConv(self.node_feat_num, 64, 'mean')},
        #     aggregate='sum')

    def forward(self):
        input_data_list = []

        def get_data_at_time_index(n, data):
            data_at_time_index = []
            for index in range(data.shape[0]):
                data_at_time_index.append(data[index][n])
            return th.stack(data_at_time_index, dim=1).T

        for i, layer in enumerate(self.hGraph_conv_layer_list):
            feat_dict = {
                NodeType.NODE.value: get_data_at_time_index(i, self.graph.hetero_graph.nodes[NodeType.NODE.value].data['feat']),
                NodeType.SVC.value: get_data_at_time_index(i, self.graph.hetero_graph.nodes[NodeType.SVC.value].data['feat']),
                NodeType.POD.value: get_data_at_time_index(i, self.graph.hetero_graph.nodes[NodeType.POD.value].data['feat'])
            }
            input_data_list.append(layer(self.graph.hetero_graph, feat_dict))
        return self.lstm_layer(th.cat(input_data_list, dim=1).T)


class HGraphConvWindows(nn.Module):
    def __init__(self, out_channel, hidden_channel, graphs: Dict[str, HeteroWithGraphIndex]):
        super(HGraphConvWindows, self).__init__()
        self.hidden_size = hidden_channel
        self.window_size = len(graphs)
        # todo 结合图结构/指标时序特征
        time_sorted = sorted(graphs.keys())
        self.lstm_layer = nn.LSTM(input_size=self.window_size, hidden_size=self.hidden_size, num_layers=2, batch_first=False)
        self.time_metrics_layer_list = []
        for time in time_sorted:
            self.time_metrics_layer_list.append(HGraphConvWindow(self.hidden_size, graphs[time]))
        self.linear1 = nn.Linear(self.hidden_size, out_channel)

    def forward(self):
        input_data_list = []
        for layer in self.time_metrics_layer_list:
            input_data_list.append(layer())
        return self.linear1(self.lstm_layer(th.cat(input_data_list, dim=0)))


# 2. 定义无监督的图神经网络模型
class UnsupervisedGNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, graphs: Dict[str, HeteroWithGraphIndex]):
        super(UnsupervisedGNN, self).__init__()
        self.conv1 = HGraphConvWindows(out_channel=out_channels, hidden_channel=hidden_size, graphs=graphs)

    def forward(self):
        x = self.conv1()
        return x


def train(graphs: Dict[str, HeteroWithGraphIndex]):
    # 3. 初始化模型和优化器
    model = UnsupervisedGNN(in_channels=5, out_channels=1, hidden_size=64, graphs=graphs)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. 训练模型
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model()

        # 使用节点的 L2 范数作为无监督的目标函数
        loss = th.norm(output, dim=1).mean()

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
