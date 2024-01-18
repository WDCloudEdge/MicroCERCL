import sys
import networkx as nx
from Config import Config
import MetricCollector
import time
from typing import Dict, List
from graph import combine_ns_graphs, graph_weight_ns, graph_weight, GraphIndex, graph_index, get_hg, \
    HeteroWithGraphIndex, graph_dump
from model import train
from anomaly_detection import get_anomaly_by_df
from util.utils import time_string_2_timestamp, timestamp_2_time_string, df_time_limit
from anomaly_detection import get_timestamp_index
import pandas as pd

if __name__ == "__main__":
    namespaces = ['bookinfo', 'hipster', 'hipster2', 'cloud-sock-shop', 'horsecoder-test']
    config = Config()


    class Simple:
        def __init__(self, global_now_time, global_end_time, label, root_cause, dir):
            self.global_now_time = global_now_time
            self.global_end_time = global_end_time
            self.label = label
            self.root_cause = root_cause
            self.dir = dir


    simples: List[Simple] = [
        Simple(
            1705125240, 1705125960, 'label1', 'details', 'abnormal/bookinfo/bookinfo-details/bookinfo-details-cpu-1',
        ),
        Simple(
            1705126080, 1705126800, 'label2', 'details', 'abnormal/bookinfo/bookinfo-details/bookinfo-details-cpu-2'
        ),
        Simple(
            1705126920, 1705127640, 'label3', 'details', 'abnormal/bookinfo/bookinfo-details/bookinfo-details-cpu-3'
        ),
        Simple(
            1705129320, 1705130040, 'label5', 'details', 'abnormal/bookinfo/bookinfo-details/bookinfo-details-mem-2'
        ),
        Simple(
            1705130160, 1705130880, 'label6', 'details', 'abnormal/bookinfo/bookinfo-details/bookinfo-details-mem-3'
        ),
        Simple(
            1705131180, 1705131900, 'label7', 'details', 'abnormal/bookinfo/bookinfo-details/bookinfo-details-net-1'
        ),
        Simple(
            1705132020, 1705132740, 'label8', 'details', 'abnormal/bookinfo/bookinfo-details/bookinfo-details-net-2'
        ),
        Simple(
            1705132860, 1705133580, 'label9', 'details', 'abnormal/bookinfo/bookinfo-details/bookinfo-details-net-3'
        )
    ]
    for simple in simples:
        global_now_time = simple.global_now_time
        global_end_time = simple.global_end_time
        now = int(time.time())
        if global_now_time > now:
            sys.exit("begin time is after now time")
        if global_end_time > now:
            global_end_time = now

        folder = '.'
        graphs_time_window: Dict[str, Dict[str, nx.DiGraph]] = {}
        base_dir = './data/' + str(simple.dir)
        time_pair_list = []
        time_pair_index = {}
        now_time = global_now_time
        end_time = global_end_time
        while now_time < end_time:
            config.start = int(round(now_time))
            config.end = int(round(now_time + config.duration))
            if config.end > end_time:
                config.end = end_time
            now_time += config.duration + config.step
            time_pair_list.append((config.start, config.end))
            df = pd.read_csv(base_dir + '/hipster/' + 'latency.csv')
            df = df_time_limit(df, config.start, config.end)
            df_time_index, df_index_time = get_timestamp_index(df)
            time_pair_index[(config.start, config.end)] = df_time_index
        # 获取拓扑有变动的时间窗口
        topology_change_time_window_list = []
        for ns in namespaces:
            config.namespace = ns
            data_folder = base_dir + '/' + config.namespace
            time_change_ns = [timestamp_2_time_string(global_now_time), timestamp_2_time_string(global_end_time)]
            topology_change_time_window_list.extend(time_change_ns)
        topology_change_time_window_list = sorted(list(set(topology_change_time_window_list)))
        for ns in namespaces:
            config.namespace = ns
            config.svcs.clear()
            config.pods.clear()
            count = 1
            data_folder = base_dir + '/' + config.namespace
            for time_pair in time_pair_list:
                config.start = time_pair[0]
                config.end = time_pair[1]
                print('第' + str(count) + '次获取 [' + config.namespace + '] 数据')
                graphs_ns_time_window = MetricCollector.collect_and_build_graphs(config, data_folder,
                                                                                 topology_change_time_window_list,
                                                                                 config.window_size, config.collect)
                graph_time_key = str(time_pair[0]) + '-' + str(time_pair[1])
                if graph_time_key not in graphs_time_window:
                    graphs_time_window[graph_time_key] = graphs_ns_time_window
                else:
                    graphs_time_window[graph_time_key].update(graphs_ns_time_window)
                config.pods.clear()
                count += 1
        config.start = global_now_time
        config.end = global_end_time
        MetricCollector.collect_node(config, base_dir + '/node', config.collect)
        # 非云边基于指标异常检测
        anomalies = {}
        anomaly_time_series = {}
        for time_pair in time_pair_list:
            for ns in namespaces:
                data_folder = base_dir + '/' + ns
                time_key = str(time_pair[0]) + '-' + str(time_pair[1])
                anomaly_list = anomalies.get(time_key, [])
                anomalies_ns, anomaly_time_series_index = get_anomaly_by_df(base_dir, data_folder, time_pair[0], time_pair[1])
                anomaly_list.extend(anomalies_ns)
                anomalies[time_key] = anomaly_list
                anomaly_time_series_list = anomaly_time_series.get(time_key, {})
                anomaly_time_series_list = {**anomaly_time_series_list, **anomaly_time_series_index}
                anomaly_time_series[time_key] = anomaly_time_series_list
        # 赋权ns子图
        for time_window in graphs_time_window:
            t_index_time_window = time_pair_index[(int(time_window.split('-')[0]), int(time_window.split('-')[1]))]
            for graph_time_window in graphs_time_window[time_window]:
                graph: nx.DiGraph = graphs_time_window[time_window][graph_time_window]
                begin_t = graph_time_window.split('-')[0]
                end_t = graph_time_window.split('-')[1]
                ns = graph_time_window[graph_time_window.index(end_t) + len(end_t) + 1:]
                graph_weight_ns(begin_t, end_t, graph, base_dir, ns)
            # 合并混合部署图
            graphs_combine: Dict[str, nx.DiGraph] = combine_ns_graphs(graphs_time_window[time_window])
            graphs_anomaly_time_series_index = {}
            graphs_anomaly_time_series_index_map = {}
            graphs_index_time_map = {}
            for time_combine in graphs_combine:
                graph_index_time_map = {}
                graph = graphs_combine[time_combine]
                begin_t = time_combine.split('-')[0]
                end_t = time_combine.split('-')[1]


                def get_t(begin_t, t_index_time_window):
                    index = len(t_index_time_window.keys()) - 1
                    for i, t in enumerate(sorted(t_index_time_window.keys())):
                        if int(begin_t) <= time_string_2_timestamp(t):
                            index = i
                            break
                    return index


                graph_weight(begin_t, end_t, graph, base_dir)
                # graph dump
                graph_dump(graph, base_dir, begin_t + '-' + end_t)
                for t in t_index_time_window:
                    if int(begin_t) <= time_string_2_timestamp(t) <= int(end_t):
                        graph_index_time_map[t_index_time_window[t] - get_t(begin_t, t_index_time_window)] = t
                graphs_index_time_map[time_combine] = graph_index_time_map
                # 赋值异常时序索引
                anomalies_series_time_window = anomaly_time_series[time_window]
                a_t_index = []
                anomaly_time_series_index = {}
                for anomaly in anomalies_series_time_window:
                    anomaly_t_index = []
                    anomaly_series_time_window = anomalies_series_time_window[anomaly]
                    anomaly_series_time_window = [time_string_2_timestamp(a) for a in anomaly_series_time_window]
                    if max(anomaly_series_time_window) < int(begin_t) or min(anomaly_series_time_window) > int(end_t):
                        continue
                    for t in anomaly_series_time_window:
                        if int(begin_t) <= t <= int(end_t):
                            a_t_index.append(t_index_time_window[timestamp_2_time_string(t)] - get_t(begin_t, t_index_time_window))
                            anomaly_t_index.append(t_index_time_window[timestamp_2_time_string(t)] - get_t(begin_t, t_index_time_window))
                    anomaly_time_series_index[anomaly] = anomaly_t_index
                a_t_index = list(set(a_t_index))
                graphs_anomaly_time_series_index[time_combine] = a_t_index
                graphs_anomaly_time_series_index_map[time_combine] = anomaly_time_series_index

            graphs_combine_index: Dict[str, GraphIndex] = {t_index: graph_index(graphs_combine[t_index]) for t_index in
                                                           graphs_combine}
            # 转化为dgl构建图网络栈
            hetero_graphs_combine: Dict[str, HeteroWithGraphIndex] = get_hg(graphs_combine, graphs_combine_index,
                                                                            anomalies,
                                                                            graphs_anomaly_time_series_index,
                                                                            graphs_anomaly_time_series_index_map
                                                                            , graphs_index_time_map)
            train(simple.label, simple.root_cause, hetero_graphs_combine, base_dir, config.train, rnn=config.rnn_type,
                  attention=config.attention)
