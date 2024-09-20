import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
from graph import HeteroWithGraphIndex, NodeType
from typing import Dict
from model_aggregate import AggrUnsupervisedGNN
from Config import RnnType, TrainType
from torch.nn import init
from torch.optim.lr_scheduler import StepLR
from util.utils import top_k_node


class EarlyStopping:
    def __init__(self, patience_output=5, patience=5, delta=1e-6, max_loss=5, min_epoch=500):
        self.patience = patience
        self.patience_output = patience_output
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

    def get_patience_output(self):
        if self.patience_output == 0:
            return []
        else:
            return self.output_list[-self.patience_output:]


# An Unsupervised Graph Neural Network Model Combining Failure Backpropagation and Topology Aggregation
class UnsupervisedGNN(nn.Module):
    def __init__(self, center_map, anomaly_index, out_channels, hidden_size, graphs: Dict[str, HeteroWithGraphIndex],
                 rnn: RnnType = RnnType.LSTM):
        super(UnsupervisedGNN, self).__init__()
        graph = graphs[next(iter(graphs))]
        sorted_graphs = [graphs[time_sorted] for time_sorted in sorted(graphs.keys())]
        self.aggr_conv = AggrUnsupervisedGNN(sorted_graphs, center_map, anomaly_index, out_channels=out_channels,
                                             hidden_size=hidden_size, rnn=rnn,
                                             svc_feat_num=
                                             graph.hetero_graph.nodes[NodeType.SVC.value].data['feat'].shape[2],
                                             node_feat_num=
                                             graph.hetero_graph.nodes[NodeType.NODE.value].data['feat'].shape[2],
                                             instance_feat_num=
                                             graph.hetero_graph.nodes[NodeType.POD.value].data['feat'].shape[2])
        self.epoch = 0

    def forward(self, graphs: Dict[str, HeteroWithGraphIndex]):
        aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series = self.aggr_conv(
            graphs)
        return aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series

    def loss(self, aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes,
             window_anomaly_time_series):
        return self.aggr_conv.loss(aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index,
                                   window_time_series_sizes, window_anomaly_time_series)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def train(config, label: str, root_cause: str, center_map: Dict[str, int], anomaly_index: Dict[str, int],
          graphs: Dict[str, HeteroWithGraphIndex],
          dir: str = '',
          is_train: TrainType = TrainType.EVAL,
          learning_rate=0.01, rnn: RnnType = RnnType.LSTM):
    model = UnsupervisedGNN(center_map, anomaly_index, out_channels=1, hidden_size=64, graphs=graphs, rnn=rnn)
    if torch.cuda.is_available():
        model = model.to('cpu')
    root_cause_file = label + '_' + rnn.value
    model_file = 'model_weights' + '_' + label + '_' + rnn.value + '.pth'
    root_cause = root_cause
    with open(dir + '/result-' + root_cause_file + '.log', "a") as output_file:
        print(f"root_cause: {root_cause}", file=output_file)
        early_stopping = EarlyStopping(patience_output=4, patience=config.patience, delta=config.delta, min_epoch=config.min_epoch)
        if is_train == TrainType.TRAIN or is_train == TrainType.TRAIN_CHECKPOINT:
            if is_train == TrainType.TRAIN:
                model.initialize_weights()
            elif is_train == TrainType.TRAIN_CHECKPOINT:
                model.load_state_dict(torch.load(dir + '/' + model_file))
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = StepLR(optimizer, step_size=200, gamma=0.5)
            for epoch in range(model.epoch, 1000, 1):
                model.set_epoch(epoch)
                optimizer.zero_grad()
                aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series = model(
                    graphs)
                loss = model.loss(aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index,
                                  window_time_series_sizes, window_anomaly_time_series)
                if loss == 0:
                    break
                early_stopping(loss, aggr_feat, epoch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if epoch % 10 == 0:
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
            aggr_feat_list, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series = model(
                graphs)
            if is_train == TrainType.TRAIN or is_train == TrainType.TRAIN_CHECKPOINT:
                output_list = early_stopping.get_patience_output()
                output_list.append(aggr_feat_list)
            elif is_train == TrainType.EVAL:
                output_list = [aggr_feat_list]
            output_score_node = {}
            for aggr_feat_list in output_list:
                for idx, window_graph_index in enumerate(window_graphs_index):
                    output = aggr_feat_list[idx]
                    window_graph_index_reverse = {window_graph_index[key]: key for key in window_graph_index}
                    for idx, score in enumerate(output):
                        node = window_graph_index_reverse[idx]
                        s = output_score_node.get(node, 0)
                        if s != 0:
                            output_score_node[node] = s + abs(score.item())
                        else:
                            output_score_node[node] = abs(score.item())
            sorted_dict_node = dict(sorted(output_score_node.items(), key=lambda item: item[1], reverse=True))
            return top_k_node(sorted_dict_node, root_cause, output_file)
