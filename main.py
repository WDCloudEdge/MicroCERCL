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
    # namespaces = ['bookinfo', 'hipster', 'hipster2', 'sock-shop', 'horsecoder-test', 'horsecoder-minio']
    namespaces = ['hipster']
    config = Config()


    class Simple:
        def __init__(self, global_now_time, global_end_time, label, root_cause):
            self.global_now_time = global_now_time
            self.global_end_time = global_end_time
            self.label = label
            self.root_cause = root_cause


    simples: List[Simple] = [
        Simple(
            1658466960, 1658467560, 'label1', 'cartservice'
        ),
        Simple(
            1658468340, 1658468940, 'label2', 'currencyservice-588fc9584d-cl9dm'
        ),
        Simple(
            1658470140, 1658470740, 'label3', 'emailservice'
        ),
        Simple(
            1658472240, 1658472840, 'label4', 'frontend-875b86bb8-lq7z8'
        ),
        Simple(
            1658473440, 1658474040, 'label5', 'paymentservice-6879f6c8c4-8pbjc'
        ),
        Simple(
            1658474580, 1658475180, 'label6', 'recommendationservice'
        ),
        Simple(
            1658475840, 1658476440, 'label7', 'shippingservice-589dc45c5d-sfzzg'
        ),
        Simple(
            1658477040, 1658477640, 'label8', 'shippingservice-589dc45c5d-sfzzg'
        ),
        Simple(
            1658478120, 1658478720, 'label9', 'shippingservice-589dc45c5d-sfzzg'
        ),
        Simple(
            1658479080, 1658479680, 'label10', 'recommendationservice-689548cfbd-ml8h2'
        ),
        Simple(
            1658480340, 1658480940, 'label11', 'productcatalogservice-5ff5f57dc8-mpw5r'
        ),
        Simple(
            1658481900, 1658482500, 'label12', 'paymentservice'
        ),
        Simple(
            1658482920, 1658483520, 'label13', 'paymentservice'
        ),
        Simple(
            1658484120, 1658484720, 'label14', 'emailservice-8848674-b7vr4'
        ),
        Simple(
            1658484960, 1658485560, 'label15', 'emailservice-8848674-b7vr4'
        ),
        Simple(
            1658487540, 1658488140, 'label16', 'currencyservice-588fc9584d-cl9dm'
        ),
        Simple(
            1658489220, 1658489820, 'label17', 'currencyservice-588fc9584d-cl9dm'
        ),
        Simple(
            1658490300, 1658490900, 'label18', 'checkoutservice-7d8cb45794-vfnmh'
        ),
        Simple(
            1658493360, 1658493960, 'label19', 'cartservice-75d494679c-vfldf'
        ),
        Simple(
            1658494740, 1658495340, 'label20', 'cartservice-75d494679c-qdl5c'
        ),
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
        base_dir = './data/' + str(config.dir)
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
                graphs_ns_time_window = MetricCollector.collect_and_build_graphs(config, base_dir,
                                                                                 topology_change_time_window_list,
                                                                                 config.window_size, config.collect)
                graphs_time_window[str(time_pair[0]) + '-' + str(time_pair[1])] = {**graphs_time_window,
                                                                                   **graphs_ns_time_window}
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
                anomalies_ns, anomaly_time_series_index = get_anomaly_by_df(data_folder, time_pair[0], time_pair[1])
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
                ns = graph_time_window.split('-')[2]
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
                        if int(begin_t) <= t:
                            index = i
                            break
                    return index


                graph_weight(begin_t, end_t, graph, base_dir)
                # graph dump
                graph_dump(graph, base_dir, begin_t + '-' + end_t)
                for t in t_index_time_window:
                    if int(begin_t) <= t <= int(end_t):
                        graph_index_time_map[t_index_time_window[t] - get_t(begin_t, t_index_time_window)] = t
                graphs_index_time_map[time_combine] = graph_index_time_map
                # 赋值异常时序索引
                anomalies_series_time_window = anomaly_time_series[time_window]
                a_t_index = []
                anomaly_time_series_index = {}
                for anomaly in anomalies_series_time_window:
                    anomaly_t_index = []
                    anomaly_series_time_window = anomalies_series_time_window[anomaly]
                    if max(anomaly_series_time_window) < int(begin_t) or min(anomaly_series_time_window) > int(end_t):
                        continue
                    for t in anomaly_series_time_window:
                        if int(begin_t) <= t <= int(end_t):
                            a_t_index.append(t_index_time_window[t] - get_t(begin_t, t_index_time_window))
                            anomaly_t_index.append(t_index_time_window[t] - get_t(begin_t, t_index_time_window))
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
