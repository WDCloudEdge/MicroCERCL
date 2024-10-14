import sys

import requests
from Config import Config
import pickle
import os
from PrometheusClient import PrometheusClient
from KubernetesClient import KubernetesClient


def collect(config, _dir, condition, start_timestamp, duration):
    pod_url = build_log_urls(config)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    for pod in pod_url.keys():
        res = execute(pod_url[pod])
        pipe_path = os.path.join(_dir, '%s_%s.pkl' % (pod, condition))
        # print(pipe_path)
        with open(pipe_path, 'ab') as fw:
            pickle.dump(res, fw)


def build_log_urls(config):
    pod_url = {}
    for pod in config.pods:
        if config.namespace == 'bookinfo':
            url = config.kiali_url + '/namespaces/{}/pods/{}/logs?container={}&sinceTime={}&duration={}s'.format(
                config.namespace,
                pod,
                pod.split('-')[0],
                int(config.start),
                config.interval)
        else:
            url = config.kiali_url + '/namespaces/{}/pods/{}/logs?container=server&sinceTime={}&duration={}s'.format(
                config.namespace,
                pod,
                int(config.start),
                config.interval)
        pod_url[pod] = url
    return pod_url


def execute(url):
    try:
        response = requests.get(url)
        # print(response.json()['entries'][0])
        return response.json()['entries']
    except:
        return None


def get_pod(config, namespace):
    prom_util = PrometheusClient(config)
    prom_sql = 'sum(container_cpu_usage_seconds_total{namespace=\"%s\", container!~\'POD|istio-proxy\'}) by (instance, pod)' % namespace
    results = prom_util.execute_prom(config.prom_range_url_node, prom_sql)

    for result in results:
        metric = result['metric']
        if 'pod' in metric:
            source = metric['pod']
            config.pods.add(source)


if __name__ == '__main__':
    namespaces = ['bookinfo', 'hipster', 'cloud-sock-shop', 'horsecoder-test']
    config = Config()
    data_folder = sys.argv[1]
    condition = sys.argv[2]
    config.start = int(sys.argv[3])
    config.end = int(sys.argv[4])
    duration = config.end - config.start
    current_directory = os.path.dirname(os.path.abspath(__file__))  
    for namespace in namespaces:
        config.namespace = namespace
        config.svcs.clear()
        config.pods.clear()
        get_pod(config, namespace)
        collect(config, os.path.join(current_directory ,'data', data_folder, namespace), condition, config.start, duration)




