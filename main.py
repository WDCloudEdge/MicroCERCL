import os
import sys
import networkx as nx
from Config import Config
import MetricCollector
import time
from typing import Dict, List
from graph import combine_ns_graphs, graph_weight_ns, graph_weight, GraphIndex, graph_index, get_hg, \
    HeteroWithGraphIndex, graph_prune, combine_timestamps_graph
from model import train
from anomaly_detection import get_anomaly_by_df
from util.utils import *
from anomaly_detection import get_timestamp_index
import pandas as pd

if __name__ == "__main__":
    namespaces = ['bookinfo', 'hipster', 'cloud-sock-shop', 'horsecoder-test']
    config = Config()

    class Simple:
        def __init__(self, global_now_time, global_end_time, label, root_cause, dir):
            self.global_now_time = global_now_time
            self.global_end_time = global_end_time
            self.label = label
            self.root_cause = root_cause
            self.dir = dir

    def read_label_logs(namespace_path, label_service, simple_list: [Simple], minute):
        label_file_folder = os.path.join(namespace_path, label_service)
        dirs = []
        for item in os.listdir(label_file_folder):
            if os.path.isdir(os.path.join(label_file_folder, item)):
                dirs.append(item)
        if simple_list is None:
            simple_list = []
        file_path = os.path.join(label_file_folder, label_service + '_label.txt')
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return simple_list
                for line in lines:
                    if 'cpu_' in line or 'mem_' in line or 'net_' in line:
                        label_line = line.strip()
                        label_line_label = label_line.split('_')[1] + '_' + label_line.split('_')[3]
                        for dr in dirs:
                            dr_splits = dr.split('_')
                            if label_line_label == (dr_splits[len(dr_splits) - 2] + '_' + dr_splits[len(dr_splits) - 1]):
                                root_cause = dr[dr.rfind(label_service):dr.rfind(label_line_label) - 1]
                                dd = dr
                        if root_cause is None:
                            sys.exit(1)
                        simple = Simple(None, None, label_line, root_cause, dd)
                    elif 'start create' in line:
                        begin = line[:19]
                    elif 'finish delete' in line:
                        end = line[:19]
                        simple.global_now_time = time_string_2_timestamp_beijing(begin) - 30 * (minute - 3)
                        simple.global_end_time = time_string_2_timestamp_beijing(end) + 30 * (minute - 3)
                        simple_list.append(simple)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    top_k_services = []
    root_cause_services = []
    root_cause_namespace_dir = 'data/abnormal/' + config.dataset
    for item in os.listdir(root_cause_namespace_dir):
        if os.path.isdir(os.path.join(root_cause_namespace_dir, item)):
            root_cause_services.append(item)
    base_output_dir = os.path.join('data/abnormal', config.dataset)
    begin_tt = time.time()
    for root_cause_service in root_cause_services:
        top_k_service = []
        simples: List[Simple] = []
        read_label_logs(base_output_dir, root_cause_service, simples, config.time_window)
        for simple in simples:
            print(simple.label)
            global_now_time = simple.global_now_time
            global_end_time = simple.global_end_time
            now = int(time.time())
            if global_now_time > now:
                sys.exit("begin time is after now time")
            if global_end_time > now:
                global_end_time = now

            folder = '.'
            graphs_change_time: Dict[str, Dict[str, nx.DiGraph]] = {}
            base_dir = os.path.join(base_output_dir, root_cause_service, str(simple.dir))
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
                df = pd.read_csv(base_dir + '/hipster/metrics/' + 'latency.csv')
                df = df_time_limit(df, config.start, config.end)
                df_time_index, df_index_time = get_timestamp_index(df)
                time_pair_index[(config.start, config.end)] = df_time_index
            # Get a window of time when the topology has changed
            topology_change_time_window_list = []
            graphs_time_window_ns_all: Dict[str, Dict[str, nx.DiGraph]] = {}
            for time_pair in time_pair_list:
                config.start = time_pair[0]
                config.end = time_pair[1]
                for ns in namespaces:
                    config.namespace = ns
                    config.svcs.clear()
                    config.pods.clear()
                    count = 1
                    data_folder = base_dir + '/' + config.namespace + '/metrics'
                    print('Process data from ' +  config.namespace + ' microservice system for ' + str(count) + 'th time')
                    graphs_time_window_ns, graph_change_times_ns = MetricCollector.collect_and_build_graphs_change_time_ns(config, data_folder, config.collect)
                    if ns not in graphs_time_window_ns_all:
                        graphs_time_window_ns_all[ns] = {}
                    graphs_time_window_ns_all[ns].update(graphs_time_window_ns)
                    topology_change_time_window_list.extend([int(i) for i in graph_change_times_ns])
                    config.pods.clear()
                    count += 1
            topology_change_time_window_list.append(int(config.end))
            topology_change_time_window_list = sorted(list(set(topology_change_time_window_list)))
            topology_change_time_window_pair_list = []
            for i in range(len(topology_change_time_window_list)):
                if i != len(topology_change_time_window_list) - 1:
                    time_change_begin = topology_change_time_window_list[i]
                    time_change_end = topology_change_time_window_list[i + 1] - 1
                    graph_time_key = str(time_change_begin) + '-' + str(time_change_end)
                    topology_change_time_window_pair_list.append(graph_time_key)
                    for ns in namespaces:
                        graphs_ns_time_window = combine_timestamps_graph(graphs_time_window_ns_all[ns], ns, time_change_begin, time_change_end)
                        if graph_time_key not in graphs_change_time:
                            graphs_change_time[graph_time_key] = graphs_ns_time_window
                        else:
                            graphs_change_time[graph_time_key].update(graphs_ns_time_window)
            config.start = global_now_time
            config.end = global_end_time
            MetricCollector.collect_node(config, base_dir + '/node', config.collect)
            # application-level anomaly detection
            anomalies = {}
            anomalies_index = {}
            anomaly_time_series = {}
            for time_pair in time_pair_list:
                time_window = str(time_pair[0]) + '-' + str(time_pair[1])
                for ns in namespaces:
                    data_folder = base_dir + '/' + ns + '/metrics'
                    anomaly_list = anomalies.get(time_window, [])
                    anomalies_ns, anomaly_time_series_index = get_anomaly_by_df(config, base_output_dir, data_folder, simple.label, time_pair[0], time_pair[1])
                    anomaly_list.extend(anomalies_ns)
                    anomaly_list = list(set(anomaly_list))
                    anomalies[time_window] = anomaly_list
                    anomaly_time_series_list = anomaly_time_series.get(time_window, {})
                    anomaly_time_series_list = {**anomaly_time_series_list, **anomaly_time_series_index}
                    anomaly_time_series[time_window] = anomaly_time_series_list
                anomalies_index[time_window] = {a: i for i, a in enumerate(anomalies[time_window])}
                # Empowerment namespace subgraph
                anomaly_index = anomalies_index[time_window]
                t_index_time_window = time_pair_index[(int(time_window.split('-')[0]), int(time_window.split('-')[1]))]
                for graph_change_k in graphs_change_time:
                    begin_t = graph_change_k.split('-')[0]
                    end_t = graph_change_k.split('-')[1]
                    if int(begin_t) < time_pair[0] or int(end_t) > time_pair[1]:
                        continue
                    for graph_change_k_ns in graphs_change_time[graph_change_k]:
                        graph: nx.DiGraph = graphs_change_time[graph_change_k][graph_change_k_ns]
                        ns = graph_change_k_ns[graph_change_k_ns.index(end_t) + len(end_t) + 1:]
                        graph_weight_ns(begin_t, end_t, graph, base_dir, ns)
                # Consolidated hybrid deployment graphs
                graphs_combine = {}
                for topology_change_time in topology_change_time_window_pair_list:
                    graph_combine: Dict[str, nx.DiGraph] = combine_ns_graphs(graphs_change_time[topology_change_time])
                    graphs_combine.update(graph_combine)
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
                    graph_prune(graph, base_dir, begin_t + '-' + end_t)
                    for t in t_index_time_window:
                        if int(begin_t) <= time_string_2_timestamp(t) <= int(end_t):
                            graph_index_time_map[t_index_time_window[t] - get_t(begin_t, t_index_time_window)] = t
                    graphs_index_time_map[time_combine] = graph_index_time_map
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
                # construct graph neural network
                hetero_graphs_combine: Dict[str, HeteroWithGraphIndex] = get_hg(graphs_combine, graphs_combine_index,
                                                                                anomalies,
                                                                                graphs_anomaly_time_series_index,
                                                                                graphs_anomaly_time_series_index_map
                                                                                , graphs_index_time_map)
                top_k = train(config, simple.label, simple.root_cause, anomaly_index, hetero_graphs_combine, base_output_dir, config.train, rnn=config.rnn_type)
                top_k_service.append(top_k)
                top_k_services.append(top_k)
        print_pr(top_k_service, os.path.join(base_output_dir, root_cause_service + '-topk.txt'))
    print_pr(top_k_services, os.path.join(base_output_dir, 'topk.txt'))
    print('time cost:' + str(time.time() - begin_tt))