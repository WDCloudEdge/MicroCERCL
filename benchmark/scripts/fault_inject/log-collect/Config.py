import time

class Config:
    def __init__(self):
        self.namespace = 'bookinfo'
        self.nodes = None
        self.svcs = set()
        self.pods = set()

        self.interval = 10 * 60  # 每次收集数据的时间（10min）
        # duration related to interval
        self.duration = self.interval
        self.start = int(round((time.time() - self.duration)))
        self.end = int(round(time.time()))

        # prometheus
        self.prom_range_url = "http://47.99.240.112:31444/api/v1/query_range"  # istio支持
        self.prom_range_url_node = "http://47.99.240.112:31222/api/v1/query_range"  # 原生Prometheus
        self.prom_no_range_url_node = "http://47.99.240.112:31222/api/v1/query"
        self.prom_no_range_url = "http://47.99.240.112:31444/api/v1/query"
        self.step = 5

        # jaeger
        self.jaeger_url = 'http://47.99.200.176:16686/api/traces?'
        self.lookBack = str(int(self.duration / 60)) + 'm'
        self.limit = 100000

        # kiali
        self.kiali_url = 'http://47.99.240.112:32001/kiali/api'

        # kubernetes
        self.k8s_config = 'config.yaml'  # kubernetes配置文件地址

        # concurrency set
        self.user = 'test-hipster'


class Node:
    def __init__(self, name, ip, node_name, cni_ip, status):
        self.name = name
        self.ip = ip
        self.node_name = node_name
        self.cni_ip = cni_ip
        self.status = status


class Pod:
    def __init__(self, node, namespace, host_ip, ip, name):
        self.node = node
        self.namespace = namespace
        self.host_ip = host_ip
        self.ip = ip
        self.name = name
