import sys

import networkx as nx

from Config import Config
import MetricCollector
import time
from typing import Dict, List
from graph import combine_ns_graphs, graph_weight_ns, graph_weight, GraphIndex, graph_index, get_hg, \
    HeteroWithGraphIndex
from model import train
from anomaly_detection import get_anomaly_by_df

if __name__ == "__main__":
    # namespaces = ['bookinfo', 'hipster', 'hipster2', 'sock-shop', 'horsecoder-test', 'horsecoder-minio']
    namespaces = ['bookinfo']
    config = Config()
    global_now_time = 1702349400
    global_end_time = 1702349700
    now = int(time.time())
    if global_now_time > now:
        sys.exit("begin time is after now time")
    if global_end_time > now:
        global_end_time = now

    folder = '.'
    graphs_time_window: Dict[str, nx.DiGraph] = {}
    base_dir = './data/' + str(config.dir)
    time_pair_list = []
    now_time = global_now_time
    end_time = global_end_time
    while now_time < end_time:
        config.start = int(round(now_time))
        config.end = int(round(now_time + config.duration))
        if config.end > end_time:
            config.end = end_time
        now_time += config.duration + 1
        time_pair_list.append((config.start, config.end))
    # 获取拓扑有变动的时间窗口
    topology_change_time_window_list = []
    for ns in namespaces:
        config.namespace = ns
        now_time = global_now_time
        end_time = global_end_time
        data_folder = base_dir + '/' + config.namespace
        time_change_ns = MetricCollector.collect_pod_num_and_build_graph_change_windows(config, data_folder, config.collect)
        topology_change_time_window_list.extend(time_change_ns)
    topology_change_time_window_list = sorted(list(set(topology_change_time_window_list)))
    for ns in namespaces:
        config.namespace = ns
        config.svcs.clear()
        config.pods.clear()
        count = 1
        now_time = global_now_time
        end_time = global_end_time
        data_folder = base_dir + '/' + config.namespace
        for time_pair in time_pair_list:
            config.start = time_pair[0]
            config.end = time_pair[1]
            print('第' + str(count) + '次获取 [' + config.namespace + '] 数据')
            # todo 在有扩缩容时重新构建图，而不是根据固定窗口
            graphs_ns_time_window = MetricCollector.collect_and_build_graphs(config, data_folder, topology_change_time_window_list,
                                                                             config.window_size, config.collect)
            graphs_time_window = {**graphs_time_window, **graphs_ns_time_window}
            config.pods.clear()
            count += 1
    MetricCollector.collect_node(config, base_dir + '/node', config.collect)
    # 非云边基于指标异常检测
    anomalies = {}
    anomaly_time_series = {}
    for time_pair in time_pair_list:
        for ns in namespaces:
            data_folder = base_dir + '/' + ns
            time_key = str(time_pair[0]) + '-' + str(time_pair[1])
            anomaly_list = anomalies.get(time_key, [])
            anomalies_ns, anomaly_time_series_index = get_anomaly_by_df(data_folder, time_pair[0], time_pair[1])
            anomaly_list.extend(anomalies_ns)
            anomalies[time_key] = anomaly_list
            anomaly_time_series_list = anomaly_time_series.get(time_key, [])
            anomaly_time_series_list.extend(anomaly_time_series_index)
            anomaly_time_series[time_key] = list(set(anomaly_time_series_list))
    # 赋权ns子图
    for time_window in graphs_time_window:
        graph: nx.DiGraph = graphs_time_window[time_window]
        begin_t = time_window.split('-')[0]
        end_t = time_window.split('-')[1]
        ns = time_window.split('-')[2]
        graph_weight_ns(begin_t, end_t, graph, base_dir, ns)
    # 合并混合部署图
    graphs_combine: Dict[str, nx.DiGraph] = combine_ns_graphs(graphs_time_window)
    for time in graphs_combine:
        graph = graphs_combine[time]
        begin_t = time.split('-')[0]
        end_t = time.split('-')[1]
        graph_weight(begin_t, end_t, graph, base_dir)
    graphs_combine_index: Dict[str, GraphIndex] = {t_index: graph_index(graphs_combine[t_index]) for t_index in
                                                   graphs_combine}
    # 转化为dgl构建图网络栈
    hetero_graphs_combine: Dict[str, HeteroWithGraphIndex] = get_hg(graphs_combine, graphs_combine_index, anomalies)
    train(hetero_graphs_combine)
