import sys
from collections import defaultdict
from typing import Dict, List, Set
import networkx as nx
import pandas as pd
import util.utils as util
from dgl import DGLHeteroGraph, heterograph
import torch as th
from enum import Enum
import json
from networkx.readwrite import json_graph
import numpy as np
import os


class NodeType(Enum):
    SVC = 'svc'
    POD = 'instance'
    NODE = 'node'
    NIC = 'nic'


class EdgeType(Enum):
    SVC_CALL = 'svc_call'
    SVC_CALL_EDGE = 'svc-svc_call-svc'
    INSTANCE_NODE = 'instance_locate'
    INSTANCE_NODE_EDGE = 'instance-instance_locate-node'
    NODE_INSTANCE = 'node_impact'
    NODE_INSTANCE_EDGE = 'node-node_impact-instance'
    INSTANCE_INSTANCE = 'instance_call'
    INSTANCE_INSTANCE_EDGE = 'instance-instance_call-instance'
    SVC_INSTANCE = 'load_balance'
    SVC_INSTANCE_EDGE = 'svc-load_balance-instance'
    INSTANCE_SVC = 'flow_entrance'
    INSTANCE_SVC_EDGE = 'instance-flow_entrance-svc'


class GraphIndex:
    def __init__(self, pod_index, svc_index, node_index, nic_index):
        self.index: Dict[str, Dict[str, int]] = {}
        self.pod_index = pod_index
        self.svc_index = svc_index
        self.node_index = node_index
        self.nic_index = nic_index
        self.index[NodeType.POD.value] = pod_index
        self.index[NodeType.SVC.value] = svc_index
        self.index[NodeType.NODE.value] = node_index
        self.index[NodeType.NIC.value] = nic_index


class HeteroWithGraphIndex:
    def __init__(self, hetero_graph: DGLHeteroGraph, hetero_graph_index: GraphIndex, n_graph: nx.DiGraph,
                 center_hetero_graph: Dict[str, DGLHeteroGraph], center_type_index, center_type_name,
                 anomaly_index, anomaly_name, anomaly_time_series_index, anomaly_time_series,
                 graph_index_time_map):
        self.hetero_graph = hetero_graph
        self.hetero_graph_index = hetero_graph_index
        self.n_graph = n_graph
        self.center_hetero_graph = center_hetero_graph
        self.center_type_index = center_type_index
        self.center_type_name = center_type_name
        self.anomaly_index = anomaly_index
        self.anomaly_name = anomaly_name
        self.anomaly_time_series_index = anomaly_time_series_index
        self.anomaly_time_series = anomaly_time_series
        self.graph_index_time_map = graph_index_time_map


def combine_graph(graphs: [nx.DiGraph]) -> nx.DiGraph:
    g = nx.DiGraph()
    for graph in graphs:
        for edge in graph.edges:
            source = edge[0]
            destination = edge[1]
            g.add_edge(source, destination)
            g.nodes[source]['type'] = graph.nodes[source]['type']
            g.nodes[destination]['type'] = graph.nodes[destination]['type']
            try:
                g.nodes[source]['center'] = graph.nodes[source]['center']
                g.nodes[destination]['center'] = graph.nodes[destination]['center']
            except:
                pass
            try:
                g.nodes[source]['data'] = graph.nodes[source]['data']
                g.nodes[destination]['data'] = graph.nodes[destination]['data']
            except:
                pass
    return g


def combine_timestamps_graph(graphs_at_timestamp: Dict[str, nx.DiGraph], namespace, topology_change_time_window_list,
                             window_size=600) -> \
        Dict[str, nx.DiGraph]:
    combine_graphs: Dict[str, nx.DiGraph] = {}
    for i in range(len(topology_change_time_window_list) - 1):
        begin = util.time_string_2_timestamp(topology_change_time_window_list[i])
        end = util.time_string_2_timestamp(topology_change_time_window_list[i + 1])
        combine = []
        for timestamp in graphs_at_timestamp:
            # time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp()
            time = int(timestamp)
            if begin <= time < end:
                combine.append(graphs_at_timestamp[timestamp])
        if len(combine) == 0:
            break
        else:
            key = str(begin) + '-' + str(end) + '-' + namespace
            combine_graphs[key] = combine_graph(combine)
    return combine_graphs


def combine_ns_graphs(graphs_time_window: Dict[str, nx.DiGraph]) -> Dict[str, nx.DiGraph]:
    graphs_ns_combine: Dict[str, List[nx.DiGraph]] = {}
    graphs_combine: Dict[str, nx.DiGraph] = {}
    for graphs_time_ns in graphs_time_window:
        graph = graphs_time_window[graphs_time_ns]
        graphs_time = graphs_time_ns[:graphs_time_ns.find('-', graphs_time_ns.find('-') + 1)]
        graph_list = graphs_ns_combine.get(graphs_time, [])
        graph_list.append(graph)
        graphs_ns_combine[graphs_time] = graph_list
    for graphs_combine_time in graphs_ns_combine:
        graphs_combine[graphs_combine_time] = combine_graph(graphs_ns_combine[graphs_combine_time])
    return graphs_combine


def graph_weight_ns(begin_time, end_time, graph: nx.DiGraph, dir, namespace):
    print('weight graph ns')
    ns_dir = dir + '/' + namespace + '/metrics'
    instance_df = util.df_time_limit_normalization(pd.read_csv(ns_dir + '/instance.csv'), begin_time, end_time)
    svc_df = util.df_time_limit_normalization(pd.read_csv(ns_dir + '/latency.csv'), begin_time, end_time)

    for node in graph.nodes:
        if graph.nodes[node]['type'] == NodeType.POD.value:
            instance = df_prefix_match(instance_df, node, [])
            if not instance.empty:
                graph.nodes[node]['data'] = instance
        elif graph.nodes[node]['type'] == NodeType.SVC.value:
            svc = df_prefix_match_svc(svc_df, node, [])
            if not svc.empty:
                graph.nodes[node]['data'] = svc


def graph_weight(begin_time, end_time, graph: nx.DiGraph, dir):
    print('weight graph')
    node_df = util.df_time_limit_normalization(pd.read_csv(dir + '/node' + '/node.csv'), begin_time, end_time)

    for node in graph.nodes:
        if graph.nodes[node]['type'] == NodeType.NODE.value:
            graph.nodes[node]['data'] = df_prefix_match(node_df, node, ['(node)' + node + '_edge_network'])
        elif graph.nodes[node]['type'] == NodeType.NIC.value:
            graph.nodes[node]['data'] = df_prefix_match(node_df, node + '_edge_network', [])


def graph_index(graph: nx.DiGraph) -> GraphIndex:
    print('index graph')
    pod_index = {}
    node_index = {}
    svc_index = {}
    nic_index = {}

    pod_count = 0
    node_count = 0
    svc_count = 0
    nic_count = 0

    for node in graph.nodes:
        if graph.nodes[node]['type'] == NodeType.POD.value:
            pod_index[node] = pod_count
            pod_count += 1
        elif graph.nodes[node]['type'] == NodeType.NODE.value:
            node_index[node] = node_count
            node_count += 1
        elif graph.nodes[node]['type'] == NodeType.SVC.value:
            svc_index[node] = svc_count
            svc_count += 1
        elif graph.nodes[node]['type'] == NodeType.NIC.value:
            nic_index[node] = nic_count
            nic_count += 1
    return GraphIndex(pod_index, svc_index, node_index, nic_index)


def df_prefix_match(df: pd.DataFrame, prefix: str, stop_words: []) -> pd.DataFrame:
    dataframe = pd.DataFrame()
    columns = df.columns

    def match(prefix):
        for stop_word in stop_words:
            if prefix == stop_word:
                return True
        return False

    for column in columns:
        if prefix in column and ((len(stop_words) != 0 and not match(prefix)) or len(stop_words) == 0):
            dataframe[column] = df[column]
    return dataframe


def df_prefix_match_svc(df: pd.DataFrame, prefix: str, stop_words: []) -> pd.DataFrame:
    dataframe = pd.DataFrame()
    columns = df.columns

    def match(prefix):
        for stop_word in stop_words:
            if prefix == stop_word:
                return True
        return False

    for column in columns:
        if prefix == column[:column.rfind('&')] and ((len(stop_words) != 0 and not match(prefix)) or len(stop_words) == 0):
            dataframe[column] = df[column]
    return dataframe


def get_hg(graphs: Dict[str, nx.DiGraph], graphs_index: Dict[str, GraphIndex], anomaly_nodes: Dict[str, List],
           graphs_anomaly_time_series_index: Dict[str, List],
           anomaly_time_series: Dict[str, Dict[str, List[int]]],
           graphs_index_time_map: Dict[str, Dict[int, str]]) -> Dict[str, HeteroWithGraphIndex]:
    hg_dict: Dict[str, HeteroWithGraphIndex] = {}
    for time_window in graphs:
        hg_data_dict = defaultdict(lambda: ([], []))
        center_index: Dict[str, Dict[str, List[int]]] = {}
        center_name: Dict[str, Dict[str, List[str]]] = {}
        graph = graphs[time_window]
        index = graphs_index[time_window]
        graph_anomaly_time_series_index = graphs_anomaly_time_series_index[time_window]
        anomaly_node = get_anomaly_time_window(anomaly_nodes, time_window)

        class NodeIndex:
            def __init__(self, name, index):
                self.name = name
                self.index = index

        type_map: Dict[str, Set[NodeIndex]] = {}
        node_exist: Set = set([])
        anomaly_index, anomaly_type_name = get_anomaly_index(index, anomaly_node, graph)
        for u, v, data in graph.edges(data=True):
            u_type = graph.nodes[u]['type']
            v_type = graph.nodes[v]['type']
            u_center = graph.nodes[u]['center']
            v_center = graph.nodes[v]['center']
            if len(v_center) != 0:
                center_type_list = center_index.get(v_center, {})
                c_list = center_type_list.get(v_type, [])
                c_list.append(index.index[v_type][v])
                center_type_list[v_type] = list(set(c_list))
                center_index[v_center] = center_type_list
                center_type_name_list = center_name.get(v_center, {})
                c_name_list = center_type_name_list.get(v_type, [])
                c_name_list.append(v)
                center_type_name_list[v_type] = list(set(c_name_list))
                center_name[v_center] = center_type_name_list
            if len(u_center) != 0:
                center_type_list = center_index.get(u_center, {})
                c_list = center_type_list.get(u_type, [])
                c_list.append(index.index[u_type][u])
                center_type_list[u_type] = list(set(c_list))
                center_index[u_center] = center_type_list
                center_type_name_list = center_name.get(u_center, {})
                c_name_list = center_type_name_list.get(u_type, [])
                c_name_list.append(u)
                center_type_name_list[u_type] = list(set(c_name_list))
                center_name[u_center] = center_type_name_list
            # todo 判断是否跨网段边
            # if len(v_center) != 0 and len(u_center) != 0 and u_center != v_center:
            #     u_type = u_type + '&' + u_center
            #     v_type = v_type + '&' + v_center
            edge_type = (u_type, f"{u_type}-{get_edge_type(u_type, v_type)}-{v_type}", v_type)

            hg_data_dict[edge_type][0].append(index.index[u_type][u])
            type_list = type_map.get(u_type, [])
            if u not in node_exist:
                type_list.append(NodeIndex(u, index.index[u_type][u]))
                node_exist.add(u)
            type_map[u_type] = type_list
            hg_data_dict[edge_type][1].append(index.index[v_type][v])
            type_list = type_map.get(v_type, [])
            if v not in node_exist:
                type_list.append(NodeIndex(v, index.index[v_type][v]))
                node_exist.add(v)
            type_map[v_type] = type_list
        _hg: DGLHeteroGraph = heterograph(
            {
                key: (th.tensor(src_list), th.tensor(dst_list))
                for key, (src_list, dst_list) in hg_data_dict.items()
            }
        )
        if th.cuda.is_available():
            _hg = _hg.to('cpu')
        for type in type_map:
            type_list = type_map[type]
            for node_index in type_list:
                if 'feat' not in _hg.nodes[type].data:
                    feat_zeros = th.zeros((_hg.number_of_nodes(type), graph.nodes[node_index.name]['data'].shape[0],
                                           graph.nodes[node_index.name]['data'].shape[1]), dtype=th.float32)
                    if th.cuda.is_available():
                        _hg.nodes[type].data['feat'] = feat_zeros.to('cpu')
                    else:
                        _hg.nodes[type].data['feat'] = feat_zeros
                feat_data = th.tensor(graph.nodes[node_index.name]['data'].values,
                                      dtype=th.float32)
                if th.cuda.is_available():
                    _hg.nodes[type].data['feat'][node_index.index] = feat_data.to('cpu')
                else:
                    _hg.nodes[type].data['feat'][node_index.index] = feat_data
        # 划分多中心子图
        center_subgraph: Dict[str, DGLHeteroGraph] = {}
        for center in center_index:
            center_type_list = center_index[center]
            center_subgraph[center] = _hg.subgraph({tp: list(center_type_list[tp]) for tp in center_type_list})
        hg = HeteroWithGraphIndex(_hg, index, graph, center_subgraph, center_index, center_name, anomaly_index,
                                  anomaly_type_name, graph_anomaly_time_series_index, anomaly_time_series[time_window],
                                  graphs_index_time_map[time_window])
        hg_dict[time_window] = hg
    return hg_dict


def get_edge_type(u_type, v_type):
    if u_type == NodeType.SVC.value and v_type == NodeType.SVC.value:
        return EdgeType.SVC_CALL.value
    if u_type == NodeType.POD.value and v_type == NodeType.NODE.value:
        return EdgeType.INSTANCE_NODE.value
    if u_type == NodeType.NODE.value and v_type == NodeType.POD.value:
        return EdgeType.NODE_INSTANCE.value
    if u_type == NodeType.POD.value and v_type == NodeType.POD.value:
        return EdgeType.INSTANCE_INSTANCE.value
    if u_type == NodeType.SVC.value and v_type == NodeType.POD.value:
        return EdgeType.SVC_INSTANCE.value
    if u_type == NodeType.POD.value and v_type == NodeType.SVC.value:
        return EdgeType.INSTANCE_SVC.value
    else:
        print(f'meet unexpected edge in graph: {u_type}-{v_type}')
        sys.exit()


def get_anomaly_index(index: GraphIndex, anomalies, graph: nx.DiGraph, is_neighbor: bool = True):
    anomaly_type_index = {}
    anomaly_type_name = {}
    for anomaly in anomalies:
        if anomaly in graph.nodes:
            type = graph.nodes[anomaly]['type']
            anomaly_key = type + '-' + anomaly
            anomaly_type_map = anomaly_type_index.get(anomaly_key, {})
            anomaly_type_list = anomaly_type_map.get(type, [])
            anomaly_type_list.append(index.index[type][anomaly])
            anomaly_type_map[type] = anomaly_type_list

            anomaly_name_map = anomaly_type_name.get(anomaly_key, {})
            anomaly_name_list = anomaly_name_map.get(type, [])
            anomaly_name_list.append(anomaly)
            anomaly_name_map[type] = anomaly_name_list
            if is_neighbor:
                neighbor_key = 'neighbor'
                # 找到异常传播前继节点
                predecessors = list(graph.predecessors(anomaly))
                for n in predecessors:
                    type = graph.nodes[n]['type']
                    anomaly_type_list = anomaly_type_map.get(type, [])
                    anomaly_type_list.append(neighbor_key + str(index.index[type][n]))
                    anomaly_type_map[type] = anomaly_type_list

                    anomaly_name_list = anomaly_name_map.get(type, [])
                    anomaly_name_list.append(neighbor_key + n)
                    anomaly_name_map[type] = anomaly_name_list
            anomaly_type_index[anomaly_key] = anomaly_type_map
            anomaly_type_name[anomaly_key] = anomaly_name_map
    return anomaly_type_index, anomaly_type_name


def get_anomaly_time_window(anomalies, graph_time_window):
    for time_window in anomalies:
        if graph_time_window.split('-')[0] >= time_window.split('-')[0] and graph_time_window.split('-')[1] <= \
                time_window.split('-')[1]:
            return anomalies[time_window]
    return None


def graph_dump(graph: nx.Graph, base_dir: str, dump_file):
    # graph dump
    graph_copy = graph.copy()
    anomaly_nodes = []
    for node, data in graph_copy.nodes(data=True):
        node_data_dict = {}
        try:
            for head in data['data'].columns:
                node_data_dict[head] = np.array(data['data'][head], dtype=float).tolist()
            graph_copy.nodes[node]['data'] = node_data_dict
        except:
            anomaly_nodes.append(node)
    graph.remove_nodes_from(anomaly_nodes)
    json_converted = json_graph.node_link_data(graph_copy)
    graph_dir = base_dir + '/graph/'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    with open(graph_dir + dump_file + '.json', 'w') as outfile:
        json.dump(json_converted, outfile, indent=4)


def graph_load(base_dir: str, file_name: str) -> nx.DiGraph:
    graph_path = base_dir + '/graph/' + file_name
    with open(graph_path, 'r') as json_file:
        json_data_read = json.load(json_file)
    graph = json_graph.node_link_graph(json_data_read, directed=True, multigraph=False)
    graph_new = nx.DiGraph()
    for edge in graph.edges:
        source = edge[0]
        destination = edge[1]
        graph_new.add_edge(source, destination)
        graph_new.nodes[source]['type'] = graph.nodes[source]['type']
        graph_new.nodes[destination]['type'] = graph.nodes[destination]['type']
        try:
            graph_new.nodes[source]['center'] = graph.nodes[source]['center']
            graph_new.nodes[destination]['center'] = graph.nodes[destination]['center']
        except:
            pass
    return graph_new


def calculate_graph_score(graph: nx.DiGraph, feature_summary, graph_index_map):
    # 计算Pagerank得分
    personalization = {node: feature_summary[graph_index_map[node]] for node in graph.nodes()}
    pagerank_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization)
    return dict(sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True))
