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
                 center_hetero_graph: Dict[str, DGLHeteroGraph], center_type_index,
                 anomaly_index, anomaly_time_series_index):
        self.hetero_graph = hetero_graph
        self.hetero_graph_index = hetero_graph_index
        self.n_graph = n_graph
        self.center_hetero_graph = center_hetero_graph
        self.center_type_index = center_type_index
        self.anomaly_index = anomaly_index
        self.anomaly_time_series_index = anomaly_time_series_index


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


def combine_timestamps_graph(graphs_at_timestamp: Dict[str, nx.DiGraph], namespace, topology_change_time_window_list, window_size=600) -> \
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
        graphs_time = graphs_time_ns[:graphs_time_ns.rfind('-')]
        graph_list = graphs_ns_combine.get(graphs_time, [])
        graph_list.append(graph)
        graphs_ns_combine[graphs_time] = graph_list
    for graphs_combine_time in graphs_ns_combine:
        graphs_combine[graphs_combine_time] = combine_graph(graphs_ns_combine[graphs_combine_time])
    return graphs_combine


def graph_weight_ns(begin_time, end_time, graph: nx.DiGraph, dir, namespace):
    print('weight graph ns')
    ns_dir = dir + '/' + namespace
    instance_df = util.df_time_limit_normalization(pd.read_csv(ns_dir + '/instance.csv'), begin_time, end_time)
    svc_df = util.df_time_limit_normalization(pd.read_csv(ns_dir + '/latency.csv'), begin_time, end_time)

    for node in graph.nodes:
        if graph.nodes[node]['type'] == NodeType.POD.value:
            graph.nodes[node]['data'] = df_prefix_match(instance_df, node, [])
        elif graph.nodes[node]['type'] == NodeType.SVC.value:
            graph.nodes[node]['data'] = df_prefix_match(svc_df, node, [])


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


def get_hg(graphs: Dict[str, nx.DiGraph], graphs_index: Dict[str, GraphIndex], anomaly_nodes: Dict[str, List],
           graphs_anomaly_time_series_index: Dict[str, List]) -> Dict[str, HeteroWithGraphIndex]:
    hg_dict: Dict[str, HeteroWithGraphIndex] = {}
    for time_window in graphs:
        hg_data_dict = defaultdict(lambda: ([], []))
        center_index: Dict[str, Dict[str, List[int]]] = {}
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
        anomaly_index = get_anomaly_index(index, anomaly_node, graph)
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
            if len(u_center) != 0:
                center_type_list = center_index.get(u_center, {})
                c_list = center_type_list.get(u_type, [])
                c_list.append(index.index[u_type][u])
                center_type_list[u_type] = list(set(c_list))
                center_index[u_center] = center_type_list
            # 判断是否跨网段边
            if len(v_center) != 0 and len(u_center) != 0 and u_center != v_center:
                u_type = u_type + '&' + u_center
                v_type = v_type + '&' + v_center
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
        for type in type_map:
            type_list = type_map[type]
            for node_index in type_list:
                if 'feat' not in _hg.nodes[type].data:
                    _hg.nodes[type].data['feat'] = th.zeros((_hg.number_of_nodes(type), graph.nodes[node_index.name]['data'].shape[0], graph.nodes[node_index.name]['data'].shape[1]), dtype=th.float32)
                _hg.nodes[type].data['feat'][node_index.index] = th.tensor(graph.nodes[node_index.name]['data'].values,
                                                                           dtype=th.float32)
        # 划分多中心子图
        center_subgraph: Dict[str, DGLHeteroGraph] = {}
        for center in center_index:
            center_type_list = center_index[center]
            center_subgraph[center] = _hg.subgraph({tp: list(center_type_list[tp]) for tp in center_type_list})
        hg = HeteroWithGraphIndex(_hg, index, graph, center_subgraph, center_index, anomaly_index, graph_anomaly_time_series_index)
        hg_dict[time_window] = hg
    return hg_dict


def get_edge_type(u_type, v_type):
    if u_type == NodeType.SVC.value and v_type == NodeType.SVC.value:
        return EdgeType.SVC_CALL.value
    if u_type == NodeType.POD.value and v_type == NodeType.NODE.value:
        return EdgeType.INSTANCE_NODE.value
    if u_type == NodeType.NODE.value and v_type == NodeType.POD.value:
        return EdgeType.NODE_INSTANCE.value
    else:
        pass


def get_anomaly_index(index: GraphIndex, anomalies, graph: nx.DiGraph, is_neighbor: bool = False):
    anomaly_type_index = {}
    for anomaly in anomalies:
        if anomaly in graph.nodes:
            type = graph.nodes[anomaly]['type']
            anomaly_key = type + '-' + anomaly
            anomaly_type_map = anomaly_type_index.get(anomaly_key, {})
            anomaly_type_list = anomaly_type_map.get(type, [])
            anomaly_type_list.append(index.index[type][anomaly])
            anomaly_type_map[type] = anomaly_type_list
            if is_neighbor:
                # 找到异常传播前继节点
                predecessors = list(graph.predecessors(anomaly))
                for n in predecessors:
                    type = graph.nodes[n]['type']
                    anomaly_type_list = anomaly_type_map.get(type, [])
                    anomaly_type_list.append(index.index[type][n])
                    anomaly_type_map[type] = anomaly_type_list
            anomaly_type_index[anomaly_key] = anomaly_type_map
    return anomaly_type_index


def get_anomaly_time_window(anomalies, graph_time_window):
    for time_window in anomalies:
        if graph_time_window.split('-')[0] >= time_window.split('-')[0] and graph_time_window.split('-')[1] <= time_window.split('-')[1]:
            return anomalies[time_window]
    return None


def graph_dump(graph: nx.Graph, base_dir: str, dump_file):
    # graph dump
    graph_copy = graph.copy()
    for node, data in graph_copy.nodes(data=True):
        node_data_dict = {}
        for head in data['data'].columns:
            node_data_dict[head] = np.array(data['data'][head], dtype=float).tolist()
        graph_copy.nodes[node]['data'] = node_data_dict
    json_converted = json_graph.node_link_data(graph_copy)
    graph_dir = base_dir + '/graph/'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    with open(graph_dir + dump_file + '.json', 'w') as outfile:
        json.dump(json_converted, outfile, indent=4)
