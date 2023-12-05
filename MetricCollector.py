import os
from Config import Config
import pandas as pd
from util.PrometheusClient import PrometheusClient
from util.KubernetesClient import KubernetesClient
import networkx as nx
from typing import Dict, List, Tuple
from graph import combine_timestamps_graph, NodeType


# todo 异常pod包含未能成功上报指标数据的pod
def collect_graph(config: Config, _dir: str, output: bool) -> Dict[str, nx.DiGraph]:
    graphs_at_timestamp: Dict[str, nx.DiGraph] = {}
    graph_df = pd.DataFrame(columns=['source', 'destination'])
    prom_util = PrometheusClient(config)
    prom_sql = 'sum(istio_tcp_received_bytes_total{destination_workload_namespace=\"%s\"}) by (source_workload, destination_workload)' % config.namespace
    results = prom_util.execute_prom(config.prom_range_url, prom_sql)

    prom_sql = 'sum(istio_requests_total{destination_workload_namespace=\"%s\"}) by (source_workload, destination_workload)' % config.namespace
    results = results + prom_util.execute_prom(config.prom_range_url, prom_sql)
    svc_timestamp_map: Dict[str, List[Tuple[str, str]]] = {}
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

    pod_timestamp_map: Dict[str, List[Tuple[str, str]]] = {}
    for result in results:
        metric = result['metric']
        if 'pod' in metric:
            source = metric['pod']
            config.pods.add(source)
            destination = metric['instance']
            # if node ip
            if ":" in destination:
                destination = destination.split(":")[0]
                for node in KubernetesClient(config).get_nodes():
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

    if output:
        graph_df['timestamp'] = graph_df['timestamp'].astype('datetime64[s]')
        graph_df = graph_df.sort_values(by='timestamp', ascending=True)
        path = os.path.join(_dir, 'graph.csv')
        graph_df.to_csv(path, index=False, mode='a')

    combine_timestamp = pod_timestamp_map.copy()
    combine_timestamp.update(svc_timestamp_map.copy())
    k8s_client = KubernetesClient(config)
    node_center: Dict[str, str] = k8s_client.get_node_center()
    for timestamp in combine_timestamp:
        g = nx.DiGraph()
        svc_call_list = svc_timestamp_map.get(timestamp, None)
        if svc_call_list:
            for svc_call in svc_call_list:
                source = svc_call[0]
                if 'ingress' in source:
                    continue
                destination = svc_call[1]
                g.add_edge(source, destination)
                g.nodes[source]['type'] = NodeType.SVC.value
                g.nodes[destination]['type'] = NodeType.SVC.value
                g.nodes[source]['center'] = ''
                g.nodes[destination]['center'] = ''
        pod_list = pod_timestamp_map.get(timestamp, None)
        if pod_list:
            for pod in pod_list:
                source = pod[0]
                destination = pod[1]
                # pod node 双向边
                g.add_edge(source, destination)
                g.add_edge(destination, source)
                center = node_center[destination]
                g.nodes[source]['type'] = NodeType.POD.value
                g.nodes[source]['center'] = center
                g.nodes[destination]['type'] = NodeType.NODE.value
                g.nodes[destination]['center'] = center
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
    call_df.to_csv(path, index=False, mode='a')


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
def collect_pod_num(config: Config, _dir: str):
    instance_df = pd.DataFrame()
    prom_util = PrometheusClient(config)
    # qps_sql = 'count(container_cpu_usage_seconds_total{namespace="%s", container!~"POD|istio-proxy"}) by (container)' % (config.yaml.namespace)
    # def handle(result, instance_df):
    #     if 'container' in result['metric']:
    #         name = result['metric']['container'] + '&count'
    #         values = result['values']
    #         values = list(zip(*values))
    #         if 'timestamp' not in instance_df:
    #             timestamp = values[0]
    #             instance_df['timestamp'] = timestamp
    #             instance_df['timestamp'] = instance_df['timestamp'].astype('datetime64[s]')
    #         metric = values[1]
    #         instance_df[name] = pd.Series(metric)
    #         instance_df[name] = instance_df[name].astype('float64')
    qps_sql = 'count(kube_pod_info{namespace="%s"}) by (created_by_name)' % config.namespace
    response = prom_util.execute_prom(config.prom_range_url_node, qps_sql)

    def handle(result, instance_df):
        if 'created_by_name' in result['metric']:
            name = result['metric']['created_by_name'][:result['metric']['created_by_name'].rfind('-')] + '&count'
            values = result['values']
            values = list(zip(*values))
            if 'timestamp' not in instance_df:
                timestamp = values[0]
                instance_df['timestamp'] = timestamp
                instance_df['timestamp'] = instance_df['timestamp'].astype('datetime64[s]')
            metric = values[1]
            instance_df[name] = pd.Series(metric)
            instance_df[name] = instance_df[name].fillna(0)
            instance_df[name] = instance_df[name].astype('float64')

    [handle(result, instance_df) for result in response]

    path = os.path.join(_dir, 'instances_num.csv')
    instance_df.to_csv(path, index=False, mode='a')


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
    df = pd.DataFrame()
    prom_util = PrometheusClient(config)
    prom_cpu_sql = 'sum(rate(container_cpu_usage_seconds_total{namespace=\'%s\',container!~\'POD|istio-proxy|\',pod!~\'jaeger.*\'}[1m])* 1000)  by (pod, instance,container)' % config.namespace

    response = prom_util.execute_prom(config.prom_range_url_node, prom_cpu_sql)
    for result in response:
        pod_name = result['metric']['pod']
        prom_memory_sql = 'sum(container_memory_working_set_bytes{namespace=\'%s\',pod="%s"}) by(pod, instance)  / 1000000' % (
            config.namespace, pod_name)
        prom_network_sql = 'sum(rate(container_network_transmit_packets_total{namespace=\"%s\", pod="%s"}[1m])) * sum(rate(container_network_transmit_packets_total{namespace=\"%s\", pod="%s"}[1m]))' % (
            config.namespace, pod_name, config.namespace, pod_name)

        config.pods.add(pod_name)
        values = result['values']
        values = list(zip(*values))
        container_df = pd.DataFrame()
        timestamp = values[0]
        container_df['timestamp'] = timestamp
        container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = pod_name + '_cpu'
        container_df[col_name] = metric
        container_df = container_df.fillna(0)
        container_df[col_name] = container_df[col_name].astype('float64')
        if df.empty:
            df = container_df
        else:
            df = pd.merge(df, container_df, on='timestamp', how='outer')
        df = df.fillna(0)

        container_df = pd.DataFrame()
        response = prom_util.execute_prom(config.prom_range_url_node, prom_memory_sql)
        values = response[0]['values']
        values = list(zip(*values))
        timestamp = values[0]
        container_df['timestamp'] = timestamp
        container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = pod_name + '_memory'
        container_df[col_name] = metric
        container_df = container_df.fillna(0)
        container_df[col_name] = container_df[col_name].astype('float64')
        if df.empty:
            df = container_df
        else:
            df = pd.merge(df, container_df, on='timestamp', how='outer')
        df = df.fillna(0)

        container_df = pd.DataFrame()
        response = prom_util.execute_prom(config.prom_range_url_node, prom_network_sql)
        values = response[0]['values']
        values = list(zip(*values))
        timestamp = values[0]
        container_df['timestamp'] = timestamp
        container_df['timestamp'] = container_df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        col_name = pod_name + '_network'
        container_df[col_name] = metric
        container_df = container_df.fillna(0)
        container_df[col_name] = container_df[col_name].astype('float64')
        if df.empty:
            df = container_df
        else:
            df = pd.merge(df, container_df, on='timestamp', how='outer')
        df = df.fillna(0)

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
        # 改动
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
        collect_pod_num(config, _dir)
        collect_ctn_metric(config, _dir)
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


def collect_and_build_graphs(config: Config, _dir: str, window_size, begin_timestamp, coll: bool) -> Dict[
    str, nx.DiGraph]:
    print('collect graphs')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    graphs: Dict[str, nx.DiGraph] = collect(config, _dir, coll)
    graphs_time_window = combine_timestamps_graph(graphs, begin_timestamp, config.namespace, window_size)
    return graphs_time_window
