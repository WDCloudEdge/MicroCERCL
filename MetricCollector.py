import os
from Config import Config, Node
import pandas as pd
from util.utils import time_string_2_timestamp
from util.PrometheusClient import PrometheusClient
from util.KubernetesClient import KubernetesClient
import networkx as nx
from typing import Dict, List, Tuple
from graph import combine_timestamps_graph, NodeType, graph_load
import random


def collect_graph(config: Config, _dir: str, collect: bool) -> Dict[str, nx.DiGraph]:
    graphs_at_timestamp: Dict[str, nx.DiGraph] = {}
    graph_df = pd.DataFrame(columns=['source', 'destination'])
    svc_timestamp_map: Dict[str, List[Tuple[str, str]]] = {}
    pod_timestamp_map: Dict[str, List[Tuple[str, str]]] = {}
    path = os.path.join(_dir, 'graph.csv')
    k8s_nodes = [Node('izbp193ioajdcnpofhlr1hz', '172.26.146.178', 'izbp193ioajdcnpofhlr1hz', '', 'Ready', 'cloud'),
                 Node('izbp16opgy3xucvexwqp9dz', '172.23.182.14', 'izbp16opgy3xucvexwqp9dz', '', 'Ready', 'cloud'),
                 Node('izbp1gwb52uyj3g0wn52lez', '172.26.146.180', 'izbp1gwb52uyj3g0wn52lez', '', 'Ready', 'cloud'),
                 Node('izbp1gwb52uyj3g0wn52lfz', '172.26.146.179', 'izbp1gwb52uyj3g0wn52lfz', '', 'Ready', 'cloud'),
                 Node('server-1', '192.168.31.74', 'server-1', '', 'Ready', 'edge-1'),
                 Node('server-2', '192.168.31.85', 'server-2', '', 'Ready', 'edge-1'),
                 Node('server-3', '192.168.31.128', 'server-3', '', 'Ready', 'edge-2'),
                 Node('dell2018', '192.168.31.208', 'dell2018', '', 'Ready', 'edge-2')]
    if collect:
        prom_util = PrometheusClient(config)
        prom_sql = 'sum(istio_tcp_received_bytes_total{destination_workload_namespace=\"%s\"}) by (source_workload, destination_workload)' % config.namespace
        results = prom_util.execute_prom(config.prom_range_url, prom_sql)

        prom_sql = 'sum(istio_requests_total{destination_workload_namespace=\"%s\"}) by (source_workload, destination_workload)' % config.namespace
        results = results + prom_util.execute_prom(config.prom_range_url, prom_sql)
        for result in results:
            metric = result['metric']
            source = metric['source_workload']
            destination = metric['destination_workload']
            config.svcs.add(source)
            config.svcs.add(destination)
            values = result['values']
            values = list(zip(*values))
            for timestamp in values[0]:
                graph_df = graph_df.append({'source': source, 'destination': destination, 'timestamp': timestamp},
                                           ignore_index=True)
                t_list = svc_timestamp_map.get(str(timestamp), [])
                t_list.append((source, destination))
                svc_timestamp_map[str(timestamp)] = t_list

        prom_sql = 'sum(container_cpu_usage_seconds_total{namespace=\"%s\", container!~\'POD|istio-proxy\'}) by (instance, pod)' % config.namespace
        results = prom_util.execute_prom(config.prom_range_url_node, prom_sql)

        for result in results:
            metric = result['metric']
            if 'pod' in metric:
                source = metric['pod']
                config.pods.add(source)
                destination = metric['instance']
                # if node ip
                if ":" in destination:
                    destination = destination.split(":")[0]
                    for node in k8s_nodes:
                        if node.ip == destination:
                            destination = node.name
                            break
                values = result['values']
                values = list(zip(*values))
                for timestamp in values[0]:
                    graph_df = graph_df.append({'source': source, 'destination': destination, 'timestamp': timestamp},
                                               ignore_index=True)
                    t_list = pod_timestamp_map.get(str(timestamp), [])
                    t_list.append((source, destination))
                    pod_timestamp_map[str(timestamp)] = t_list

        graph_df['timestamp'] = graph_df['timestamp'].astype('datetime64[s]')
        graph_df = graph_df.sort_values(by='timestamp', ascending=True)
        graph_df.to_csv(path, index=False, mode='a')

    else:
        graph_df = pd.read_csv(path)
        # graph_df['timestamp'] = pd.to_datetime(graph_df['timestamp'])
        grouped = graph_df.groupby('timestamp')

        def is_pod_node(row, nodes: List[Node]):
            for n in nodes:
                if row['destination'] == n.node_name:
                    return True
            return False

        for group_name, group_data in grouped:
            timestamp_str = str(time_string_2_timestamp(str(group_name)))
            for idx, row in group_data.iterrows():
                if is_pod_node(row, k8s_nodes):
                    t_list = pod_timestamp_map.get(timestamp_str, [])
                    t_list.append((row['source'], row['destination']))
                    pod_timestamp_map[timestamp_str] = t_list
                    config.pods.add(row['source'])
                else:
                    t_list = svc_timestamp_map.get(timestamp_str, [])
                    t_list.append((row['source'], row['destination']))
                    svc_timestamp_map[timestamp_str] = t_list
                    config.svcs.add(row['source'])
                    config.svcs.add(row['destination'])

    combine_timestamp = pod_timestamp_map.copy()
    combine_timestamp.update(svc_timestamp_map.copy())
    k8s_client = KubernetesClient(config)
    node_center: Dict[str, str] = {'izbp16opgy3xucvexwqp9dz': 'cloud',
                                   'izbp193ioajdcnpofhlr1hz': 'cloud',
                                   'izbp1gwb52uyj3g0wn52lez': 'cloud',
                                   'izbp1gwb52uyj3g0wn52lfz': 'cloud',
                                   'server-1': 'edge-1',
                                   'server-2': 'edge-1',
                                   'server-3': 'edge-2',
                                   'dell2018': 'edge-2'}
    for timestamp in combine_timestamp:
        g = nx.DiGraph()
        svc_call_list = svc_timestamp_map.get(timestamp, None)
        svc_exist = []
        if svc_call_list:
            for svc_call in svc_call_list:
                source = svc_call[0]
                destination = svc_call[1]
                if source in config.masks or destination in config.masks:
                    continue
                g.add_edge(source, destination)
                g.nodes[source]['type'] = NodeType.SVC.value
                g.nodes[destination]['type'] = NodeType.SVC.value
                g.nodes[source]['center'] = ''
                g.nodes[destination]['center'] = ''
                svc_exist.append(source)
                svc_exist.append(destination)
        svc_exist = list(set(svc_exist))
        pod_list = pod_timestamp_map.get(timestamp, None)
        if pod_list:
            svc_pods_map = {}
            for pod in pod_list:
                source = pod[0]
                destination = pod[1]
                for svc in svc_exist:
                    if svc in source:
                        svc_pods = svc_pods_map.get(svc, [])
                        svc_pods.append(source)
                        svc_pods = list(set(svc_pods))
                        svc_pods_map[svc] = svc_pods
                center = node_center[destination]
                g.add_node(source)
                g.nodes[source]['type'] = NodeType.POD.value
                g.nodes[source]['center'] = center
                if source in config.masks or destination in config.masks:
                    continue
                # pod node 双向边
                g.add_edge(source, destination)
                g.add_edge(destination, source)
                g.nodes[destination]['type'] = NodeType.NODE.value
                g.nodes[destination]['center'] = center
            for svc in svc_pods_map:
                svc_pods = svc_pods_map[svc]
                for svc_pod in svc_pods:
                    g.add_edge(svc, svc_pod)
                    g.add_edge(svc_pod, svc)
        graphs_at_timestamp[timestamp] = g
    return graphs_at_timestamp


# Get the response time of the invocation edges
def collect_call_latency(config: Config, _dir: str):
    call_df = pd.DataFrame()

    prom_util = PrometheusClient(config)
    # P50，P90，P99
    prom_50_sql = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, source_workload, le))' % config.namespace
    prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, source_workload, le))' % config.namespace
    prom_99_sql = 'histogram_quantile(0.99, sum(irate(istio_request_duration_milliseconds_bucket{destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, source_workload, le))' % config.namespace
    responses_50 = prom_util.execute_prom(config.prom_range_url, prom_50_sql)
    responses_90 = prom_util.execute_prom(config.prom_range_url, prom_90_sql)
    responses_99 = prom_util.execute_prom(config.prom_range_url, prom_99_sql)

    def handle(result, call_df, type):
        name = result['metric']['source_workload'] + '_' + result['metric']['destination_workload']
        values = result['values']
        values = list(zip(*values))
        if 'timestamp' not in call_df:
            timestamp = values[0]
            call_df['timestamp'] = timestamp
            call_df['timestamp'] = call_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        key = name + '&' + type
        call_df[key] = pd.Series(metric)
        call_df[key] = call_df[key].fillna(0)
        call_df[key] = call_df[key].astype('float64')

    [handle(result, call_df, 'p50') for result in responses_50]
    [handle(result, call_df, 'p90') for result in responses_90]
    [handle(result, call_df, 'p99') for result in responses_99]

    path = os.path.join(_dir, 'call.csv')
    call_df.fillna(0).to_csv(path, index=False, mode='a')


# Get the response time for the microservices
def collect_svc_latency(config: Config, _dir: str):
    latency_df = pd.DataFrame()

    prom_util = PrometheusClient(config)
    # P50，P90，P99
    prom_50_sql = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, le))' % config.namespace
    prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, le))' % config.namespace
    prom_99_sql = 'histogram_quantile(0.99, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, le))' % config.namespace
    responses_50 = prom_util.execute_prom(config.prom_range_url, prom_50_sql)
    responses_90 = prom_util.execute_prom(config.prom_range_url, prom_90_sql)
    responses_99 = prom_util.execute_prom(config.prom_range_url, prom_99_sql)

    def handle(result, latency_df, type):
        name = result['metric']['destination_workload']
        values = result['values']
        values = list(zip(*values))
        if 'timestamp' not in latency_df:
            timestamp = values[0]
            latency_df['timestamp'] = timestamp
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        key = name + '&' + type
        latency_df[key] = pd.Series(metric)
        latency_df[key] = latency_df[key].fillna(0)
        latency_df[key] = latency_df[key].astype('float64')

    [handle(result, latency_df, 'p50') for result in responses_50]
    [handle(result, latency_df, 'p90') for result in responses_90]
    [handle(result, latency_df, 'p99') for result in responses_99]

    path = os.path.join(_dir, 'latency.csv')
    latency_df.to_csv(path, index=False, mode='a')


# 获取机器的vCPU和memory使用
def collect_resource_metric(config: Config, _dir: str):
    metric_df = pd.DataFrame()
    vCPU_sql = 'sum(rate(container_cpu_usage_seconds_total{image!="",namespace="%s"}[1m]))' % config.namespace
    mem_sql = 'sum(rate(container_memory_usage_bytes{image!="",namespace="%s"}[1m])) / (1024*1024)' % config.namespace
    prom_util = PrometheusClient(config)
    vCPU = prom_util.execute_prom(config.prom_range_url_node, vCPU_sql)
    mem = prom_util.execute_prom(config.prom_range_url_node, mem_sql)

    def handle(result, metric_df, col):
        values = result['values']
        values = list(zip(*values))
        if 'timestamp' not in metric_df:
            timestamp = values[0]
            metric_df['timestamp'] = timestamp
            metric_df['timestamp'] = metric_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        metric_df[col] = pd.Series(metric)
        metric_df[col] = metric_df[col].fillna(0)
        metric_df[col] = metric_df[col].astype('float64')

    [handle(result, metric_df, 'vCPU') for result in vCPU]
    [handle(result, metric_df, 'memory') for result in mem]

    path = os.path.join(_dir, 'resource.csv')
    metric_df.to_csv(path, index=False, mode='a')


# Get the number of pods for all microservices
def collect_pod_num(config: Config, _dir: str, coll: bool):
    path = os.path.join(_dir, 'instances_num.csv')
    if coll:
        instance_df = pd.DataFrame()
        prom_util = PrometheusClient(config)
        pod_info_sql = 'count(kube_pod_info{namespace="%s"}) by (created_by_name)' % config.namespace
        response = prom_util.execute_prom(config.prom_range_url_node, pod_info_sql)

        for result in response:
            if 'created_by_name' in result['metric']:
                name = result['metric']['created_by_name']
                values = result['values']
                values = list(zip(*values))
                deployment_df = pd.DataFrame()
                timestamp = values[0]
                deployment_df['timestamp'] = timestamp
                deployment_df['timestamp'] = deployment_df['timestamp'].astype('datetime64[s]')
                metric = pd.Series(values[1])
                deployment_df[name] = metric
                deployment_df = deployment_df.fillna(0)
                deployment_df[name] = deployment_df[name].astype('float64')
                if instance_df.empty:
                    instance_df = deployment_df
                else:
                    instance_df = pd.merge(instance_df, deployment_df, on='timestamp', how='outer')
                instance_df = instance_df.fillna(0)

        instance_num_df = pd.DataFrame()
        for column in instance_df.columns:
            if column == 'timestamp':
                continue
            name = column[:column.rfind('-')] + '&count'
            if name not in instance_num_df.columns:
                instance_num_df[name] = instance_df[column]
            else:
                instance_num_df[name] = instance_num_df[name] + instance_df[column]
        instance_num_df['timestamp'] = instance_df['timestamp'].astype('datetime64[s]')

        instance_num_df.to_csv(path, index=False, mode='a')
    else:
        instance_num_df = pd.read_csv(path)
    return instance_num_df


# get qps for microservice
def collect_svc_qps(config: Config, _dir: str):
    qps_df = pd.DataFrame()
    prom_util = PrometheusClient(config)
    qps_sql = 'sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[30s])) by (destination_workload)' % config.namespace
    response = prom_util.execute_prom(config.prom_range_url, qps_sql)

    def handle(result, qps_df):
        name = result['metric']['destination_workload']
        values = result['values']
        values = list(zip(*values))
        if 'timestamp' not in qps_df:
            timestamp = values[0]
            qps_df['timestamp'] = timestamp
            qps_df['timestamp'] = qps_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        qps_df[name] = pd.Series(metric)
        qps_df[name] = qps_df[name].fillna(0)
        qps_df[name] = qps_df[name].astype('float64')

    [handle(result, qps_df) for result in response]

    path = os.path.join(_dir, 'svc_qps.csv')
    qps_df.to_csv(path, index=False, mode='a')


# Get metric for microservices
def collect_svc_metric(config: Config, _dir: str):
    prom_util = PrometheusClient(config)
    final_df = prom_util.get_svc_metric_range()
    path = os.path.join(_dir, 'svc_metric.csv')
    final_df.to_csv(path, index=False, mode='a')


# 收集容器CPU, memory, network
def collect_ctn_metric(config: Config, _dir: str):
    pod_df = pd.DataFrame()
    prom_util = PrometheusClient(config)

    pod_info_sql = 'kube_pod_info{namespace=\'%s\'}' % config.namespace
    response = prom_util.execute_prom(config.prom_range_url_node, pod_info_sql)
    for result in response:
        pod_name = result['metric']['pod']
        values = result['values']
        values = list(zip(*values))
        container_df = pd.DataFrame()
        timestamp = values[0]
        metric = pd.Series(values[1])
        container_df['timestamp'] = timestamp
        container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
        container_df[pod_name] = metric
        container_df = container_df.fillna(0)
        container_df[pod_name] = container_df[pod_name].astype('float64')
        if pod_df.empty:
            pod_df = container_df
        elif pod_name in pod_df.columns:
            pod_df = pod_df.set_index('timestamp').combine_first(container_df.set_index('timestamp')).reset_index()
            for i in range(len(container_df[pod_name])):
                pod_df_index = pod_df.loc[pod_df['timestamp'] == container_df['timestamp'][i]].index[0]
                pod_df.at[pod_df_index, pod_name] = 1 if pod_df[pod_name][pod_df_index] == 0 and container_df[pod_name][
                    i] == 1 else pod_df[pod_name][pod_df_index]
        else:
            pod_df = pd.merge(pod_df, container_df, on='timestamp', how='outer')
        pod_df = pod_df.fillna(0)

    prom_cpu_sql = 'sum(rate(container_cpu_usage_seconds_total{namespace=\'%s\',container!~\'POD|istio-proxy|\',pod!~\'jaeger.*\'}[1m])* 1000)  by (pod, instance, container)' % config.namespace
    prom_memory_sql = 'sum(container_memory_working_set_bytes{namespace=\'%s\',container!~\'POD|istio-proxy|\',pod!~\'jaeger.*\'}) by(pod, instance, container)  / 1000000' % (
        config.namespace)
    response = prom_util.execute_prom(config.prom_range_url_node, prom_cpu_sql)
    cpu_rename = {}
    cpu_df = pd.DataFrame()
    for result in response:
        pod_name = result['metric']['pod']
        config.pods.add(pod_name)
        values = result['values']
        values = list(zip(*values))
        container_df = pd.DataFrame()
        timestamp = values[0]
        container_df['timestamp'] = timestamp
        container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = pod_name + '_cpu'
        cpu_rename[pod_name] = col_name
        container_df[pod_name] = metric
        container_df = container_df.fillna(0)
        container_df[pod_name] = container_df[pod_name].astype('float64')
        if cpu_df.empty:
            cpu_df = container_df
        else:
            cpu_df = pd.merge(cpu_df, container_df, on='timestamp', how='outer')
        cpu_df = cpu_df.fillna(0)
    cpu_df = cpu_df.mask((cpu_df == 0) & (pod_df == 0), -1)
    # mask = np.logical_and(cpu_df.values == 0, pod_df.values == 0)
    # cpu_df[mask] = -1
    cpu_df.rename(columns=cpu_rename, inplace=True)

    mem_rename = {}
    mem_df = pd.DataFrame()
    response = prom_util.execute_prom(config.prom_range_url_node, prom_memory_sql)
    for result in response:
        pod_name = result['metric']['pod']
        config.pods.add(pod_name)
        container_df = pd.DataFrame()
        values = result['values']
        values = list(zip(*values))
        timestamp = values[0]
        container_df['timestamp'] = timestamp
        container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = pod_name + '_memory'
        mem_rename[pod_name] = col_name
        container_df[pod_name] = metric
        container_df = container_df.fillna(0)
        container_df[pod_name] = container_df[pod_name].astype('float64')
        if mem_df.empty:
            mem_df = container_df
        else:
            mem_df = pd.merge(mem_df, container_df, on='timestamp', how='outer')
        mem_df = mem_df.fillna(0)
    mem_df = mem_df.mask((mem_df == 0) & (pod_df == 0), -1)
    mem_df.rename(columns=mem_rename, inplace=True)

    net_rename = {}
    net_df = pd.DataFrame()
    for pod_name in config.pods:
        prom_network_sql = 'sum(rate(container_network_transmit_packets_total{namespace=\"%s\", pod="%s"}[1m])) * sum(rate(container_network_transmit_packets_total{namespace=\"%s\", pod="%s"}[1m]))' % (
            config.namespace, pod_name, config.namespace, pod_name)
        response = prom_util.execute_prom(config.prom_range_url_node, prom_network_sql)
        container_df = pd.DataFrame()
        values = response[0]['values']
        values = list(zip(*values))
        timestamp = values[0]
        container_df['timestamp'] = timestamp
        container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = pod_name + '_network'
        net_rename[pod_name] = col_name
        container_df[pod_name] = metric
        container_df = container_df.fillna(0)
        container_df[pod_name] = container_df[pod_name].astype('float64')
        if net_df.empty:
            net_df = container_df
        else:
            net_df = pd.merge(net_df, container_df, on='timestamp', how='outer')
        net_df = net_df.fillna(0)
    net_df = net_df.mask((net_df == 0) & (pod_df == 0), -1)
    net_df.rename(columns=net_rename, inplace=True)

    df = pd.merge(cpu_df, mem_df, on='timestamp', how='outer')
    df = pd.merge(df, net_df, on='timestamp', how='outer')
    path = os.path.join(_dir, 'instance.csv')
    df.to_csv(path, index=False, mode='a')
    return df


# Get the success rate for microservices
def collect_succeess_rate(config: Config, _dir: str):
    success_df = pd.DataFrame()
    prom_util = PrometheusClient(config)
    success_rate_sql = '(sum(rate(istio_requests_total{reporter="destination", response_code!~"5.*",namespace="%s"}[1m])) by (destination_workload, destination_workload_namespace) / sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m])) by (destination_workload, destination_workload_namespace))' % (
        config.namespace, config.namespace)
    response = prom_util.execute_prom(config.prom_range_url, success_rate_sql)

    def handle(result, success_df):
        name = result['metric']['destination_workload']
        values = result['values']
        values = list(zip(*values))
        if 'timestamp' not in success_df:
            timestamp = values[0]
            success_df['timestamp'] = timestamp
            success_df['timestamp'] = success_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        success_df[name] = pd.Series(metric)
        success_df[name] = success_df[name].astype('float64')

    [handle(result, success_df) for result in response]

    path = os.path.join(_dir, 'success_rate.csv')
    success_df.to_csv(path, index=False, mode='a')


def collect_node_metric(config: Config, _dir: str):
    df = pd.DataFrame()
    prom_util = PrometheusClient(config)
    for node in KubernetesClient(config).get_nodes():
        prom_sql = 'rate(node_network_transmit_packets_total{device="cni0", instance="%s"}[1m]) / 1000' % node.node_name
        response = prom_util.execute_prom(config.prom_range_url_node, prom_sql)
        if response == []:
            return
        values = response[0]['values']
        values = list(zip(*values))
        timestamp = values[0]
        node_df = pd.DataFrame()
        node_df['timestamp'] = timestamp
        node_df['timestamp'] = node_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = '(node)' + node.name + '_network'
        node_df[col_name] = metric
        node_df[col_name] = node_df[col_name].astype('float64')
        node_df = node_df.fillna(0)
        if df.empty:
            df = node_df
        else:
            df = pd.merge(df, node_df, on='timestamp', how='outer')
        df = df.fillna(0)

        prom_sql = 'rate(node_network_transmit_packets_total{device="raven0", instance="%s"}[3m]) / 1000' % node.node_name
        response = prom_util.execute_prom(config.prom_range_url_node, prom_sql)
        values = response[0]['values']
        values = list(zip(*values))
        node_df = pd.DataFrame()
        node_df['timestamp'] = timestamp
        node_df['timestamp'] = node_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = '(node)' + node.name + '_edge_network'
        node_df[col_name] = metric
        node_df[col_name] = node_df[col_name].astype('float64')
        node_df = node_df.fillna(0)
        if df.empty:
            df = node_df
        else:
            df = pd.merge(df, node_df, on='timestamp', how='outer')
        df = df.fillna(0)

        prom_sql = '1-(sum(increase(node_cpu_seconds_total{instance="%s",mode="idle"}[1m]))/sum(increase(node_cpu_seconds_total{instance="%s"}[1m])))' % (
            node.node_name, node.node_name)
        response = prom_util.execute_prom(config.prom_range_url_node, prom_sql)
        values = response[0]['values']
        values = list(zip(*values))
        node_df = pd.DataFrame()
        node_df['timestamp'] = timestamp
        node_df['timestamp'] = node_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = '(node)' + node.name + '_cpu'
        node_df[col_name] = metric
        node_df[col_name] = node_df[col_name].astype('float64')
        node_df = node_df.fillna(0)
        if df.empty:
            df = node_df
        else:
            df = pd.merge(df, node_df, on='timestamp', how='outer')
        df = df.fillna(0)

        prom_sql = '(node_memory_MemTotal_bytes{instance="%s"}-(node_memory_MemFree_bytes{instance="%s"}+ node_memory_Cached_bytes{instance="%s"} + node_memory_Buffers_bytes{instance="%s"})) / node_memory_MemTotal_bytes{instance="%s"}' % (
            node.node_name, node.node_name, node.node_name, node.node_name, node.node_name)
        response = prom_util.execute_prom(config.prom_range_url_node, prom_sql)
        values = response[0]['values']
        values = list(zip(*values))
        node_df = pd.DataFrame()
        node_df['timestamp'] = timestamp
        node_df['timestamp'] = node_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = '(node)' + node.name + '_memory'
        node_df[col_name] = metric
        node_df[col_name] = node_df[col_name].astype('float64')
        node_df = node_df.fillna(0)
        if df.empty:
            df = node_df
        else:
            df = pd.merge(df, node_df, on='timestamp', how='outer')
        df = df.fillna(0)

    path = os.path.join(_dir, 'node.csv')
    df.to_csv(path, index=False, mode='a')

    return df


def collect(config: Config, _dir: str, collect: bool) -> Dict[str, nx.DiGraph]:
    print('collect metrics')
    # 建立文件夹
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    # 收集各种数据
    graphs: Dict[str, nx.DiGraph] = collect_graph(config, _dir, collect)
    if collect:
        collect_call_latency(config, _dir)
        collect_svc_latency(config, _dir)
        collect_resource_metric(config, _dir)
        collect_succeess_rate(config, _dir)
        collect_svc_qps(config, _dir)
        collect_svc_metric(config, _dir)
        # collect_pod_num(config, _dir)
        collect_ctn_metric(config, _dir)
    return graphs


def collect_from_file(config: Config, _dir: str) -> Dict[str, nx.DiGraph]:
    print('collect graph from file')
    graph = graph_load(_dir, 'graph.json')
    graphs: Dict[str, nx.DiGraph] = {t: graph for t in range(config.start, config.end + config.step, config.step)}
    return graphs


def collect_node(config: Config, _dir: str, coll: bool):
    print('collect node metrics')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    if coll:
        collect_node_metric(config, _dir)


def collect_graph_single(config: Config, _dir: str):
    print('collect graph')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    collect_graph(config, _dir, True)


def collect_and_build_graphs(config: Config, _dir: str, topology_change_time_window_list, window_size, coll: bool) -> \
        Dict[
            str, nx.DiGraph]:
    print('collect graphs')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    graphs: Dict[str, nx.DiGraph] = collect(config, _dir, coll)
    # for graph_key in graphs:
    #     graph = graphs[graph_key]
    #     graph_svcs_calls = []
    #     graph_svcs = []
    #     for edge in graph.edges:
    #         source = edge[0]
    #         target = edge[1]
    #         if graph.nodes[source]['type'] == 'svc' and graph.nodes[target]['type'] == 'svc':
    #             graph_svcs_calls.append((source, target))
    #             graph_svcs.append(source)
    #             graph_svcs.append(target)
    #     graph_svcs = list(set(graph_svcs))
    #     for svc_call in graph_svcs_calls:
    #         s = svc_call[0]
    #         t = svc_call[1]
    #         for sn in graph.nodes:
    #             if s in sn and not s == sn:
    #                 for tn in graph.nodes:
    #                     if t in tn and not t == tn:
    #                         # pass
    #                         random_number = random.random()
    #                         if random_number > 0.5:
    #                             graph.add_edge(sn, tn)
    graphs_time_window = combine_timestamps_graph(graphs, config.namespace, topology_change_time_window_list,
                                                  window_size)
    return graphs_time_window


def collect_pod_num_and_build_graph_change_windows(config: Config, _dir: str, coll: bool) -> List[
    str]:
    print('collect pod numbers and build graph change windows')
    time_change = []
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    instance_num_df = collect_pod_num(config, _dir, coll)
    for index, row in instance_num_df.iterrows():
        if index == 0 or (
                index >= 1 and not row.drop('timestamp').equals(instance_num_df.iloc[index - 1].drop('timestamp'))
                and (time_string_2_timestamp(str(row['timestamp'])) - time_string_2_timestamp(
            time_change[-1])) >= config.graph_min_gap
        ):
            time_change.append(str(row['timestamp']))
    last_timestamp = str(instance_num_df.iloc[instance_num_df.shape[0] - 1]['timestamp'])
    if last_timestamp not in time_change and (
            time_string_2_timestamp(last_timestamp) - time_string_2_timestamp(time_change[-1])) >= config.graph_min_gap:
        time_change.append(last_timestamp)
    return time_change
