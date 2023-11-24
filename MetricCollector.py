import os
import Config
import pandas as pd
from util.PrometheusClient import PrometheusClient


def collect_graph(config: Config, _dir: str):
    graph_df = pd.DataFrame(columns=['source', 'destination'])
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
                for node in config.nodes:
                    if node.ip == destination:
                        destination = node.name
                        break
            values = result['values']
            values = list(zip(*values))
            for timestamp in values[0]:
                graph_df = graph_df.append({'source': source, 'destination': destination, 'timestamp': timestamp},
                                           ignore_index=True)

    graph_df['timestamp'] = graph_df['timestamp'].astype('datetime64[s]')
    graph_df = graph_df.sort_values(by='timestamp', ascending=True)
    path = os.path.join(_dir, 'graph.csv')
    graph_df.to_csv(path, index=False, mode='a')


def collect(config: Config, _dir: str):
    print('collect metrics')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    collect_graph(config, _dir)