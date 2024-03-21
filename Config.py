import time
from enum import Enum


class RnnType(Enum):
    LSTM = 'lstm'
    GRU = 'gru'


class TrainType(Enum):
    TRAIN = 0
    EVAL = 1
    TRAIN_CHECKPOINT = 2


class Config:
    def __init__(self):
        # base
        self.train = TrainType.TRAIN
        self.collect = False
        self.rnn_type = RnnType.LSTM
        self.attention = False
        self.namespace = 'hipster'
        self.nodes = None
        self.svcs = set()
        self.pods = set()

        self.interval = 15 * 60
        # duration related to interval
        self.duration = self.interval
        self.window_size = 60
        self.start = int(round((time.time() - self.duration)))
        self.end = int(round(time.time()))

        # prometheus
        self.prom_range_url = "http://47.99.240.112:31444/api/v1/query_range"
        self.prom_range_url_node = "http://47.99.240.112:31222/api/v1/query_range"
        self.prom_no_range_url = "http://47.99.240.112:31444/api/v1/query"
        self.step = 5

        # jarger
        self.jaeger_url = 'http://47.99.200.176:16686/api/traces?'
        self.lookBack = str(int(self.duration / 60)) + 'm'
        self.limit = 100000

        # kubernetes
        self.k8s_config = 'local-config'

        # graph
        self.graph_min_gap = 12 * self.step

        self.masks = ['izbp16opgy3xucvexwqp9dz', 'istio-ingressgateway', 'izbp1gwb52uyj3g0wn52lfz', 'adservice-edge', 'productcatalogservice-edge', 'izbp1gwb52uyj3g0wn52lez', 'dell2018', 'unknown']


class Node:
    def __init__(self, name, ip, node_name, cni_ip, status, center):
        self.name = name
        self.ip = ip
        self.node_name = node_name
        self.cni_ip = cni_ip
        self.status = status
        self.center = center


class Pod:
    def __init__(self, node, namespace, host_ip, ip, name, center):
        self.node = node
        self.namespace = namespace
        self.host_ip = host_ip
        self.ip = ip
        self.name = name
        self.center = center
