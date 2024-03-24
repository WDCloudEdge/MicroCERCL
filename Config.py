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

        self.time_window = 15
        self.interval = self.time_window * 60
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

        # anomaly threshold
        self.anomaly_threshold = 0.03
        self.delta = 1e-5
        self.min_epoch = 200
        self.patience = 5

        self.masks = {
            'adservice': ['izbp16opgy3xucvexwqp9dz', 'istio-ingressgateway', 'izbp1gwb52uyj3g0wn52lfz', 'adservice-edge', 'productcatalogservice-edge', 'izbp1gwb52uyj3g0wn52lez', 'dell2018', 'unknown'],
            'adservice-edge': ['izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'istio-ingressgateway', 'productcatalogservice-edge', 'izbp1gwb52uyj3g0wn52lez', 'adservice', 'dell2018', 'unknown', 'cartservice-edge'],
            'cartservice': ['izbp16opgy3xucvexwqp9dz', 'istio-ingressgateway', 'currencyservice-edge', 'adservice-edge', 'productcatalogservice-edge', 'cartservice-edge', 'izbp1gwb52uyj3g0wn52lfz', 'unknown', 'dell2018'],
            'checkoutservice': ['izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'istio-ingressgateway', 'productcatalogservice-edge', 'adservice-edge', 'izbp1gwb52uyj3g0wn52lez', 'dell2018', 'unknown', 'cartservice-edge'],
            'currencyservice': ['izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'istio-ingressgateway', 'productcatalogservice-edge', 'adservice-edge', 'izbp1gwb52uyj3g0wn52lez', 'dell2018', 'unknown', 'cartservice-edge'],
            'emailservice': ['izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'istio-ingressgateway', 'productcatalogservice-edge', 'adservice', 'izbp1gwb52uyj3g0wn52lez', 'dell2018', 'unknown', 'cartservice-edge'],
            'emailservice-edge': ['izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'istio-ingressgateway', 'productcatalogservice-edge', 'dell2018', 'unknown', 'adservice', 'izbp1gwb52uyj3g0wn52lez', 'cartservice-edge'],
            'frontend': ['istio-ingressgateway', 'izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'adservice-edge', 'productcatalogservice-edge', 'izbp1gwb52uyj3g0wn52lez', 'dell2018', 'unknown'],
            'paymentservice': ['izbp16opgy3xucvexwqp9dz', 'istio-ingressgateway', 'izbp1gwb52uyj3g0wn52lfz', 'adservice', 'productcatalogservice-edge', 'izbp1gwb52uyj3g0wn52lez', 'dell2018', 'unknown', 'cartservice-edge'],
            'paymentservice-edge': ['izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'istio-ingressgateway', 'productcatalogservice-edge', 'adservice-edge', 'checkoutservice-6f9b7759b4-2x9sk', 'izbp1gwb52uyj3g0wn52lez', 'unknown', 'dell2018', 'cartservice-edge'],
            'productcatalogservice': ['izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'istio-ingressgateway', 'unknown', 'productcatalogservice-edge', 'adservice', 'izbp1gwb52uyj3g0wn52lez', 'dell2018', 'cartservice-edge'],
            'productcatalogservice-edge': ['istio-ingressgateway', 'izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'unknown', 'adservice-edge', 'izbp1gwb52uyj3g0wn52lez', 'dell2018', 'cartservice-edge'],
            'recommendationservice': ['istio-ingressgateway', 'izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'unknown', 'productcatalogservice-edge', 'adservice-edge', 'izbp1gwb52uyj3g0wn52lez', 'dell2018'],
            'shippingservice': ['izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'istio-ingressgateway', 'izbp1gwb52uyj3g0wn52lez', 'productcatalogservice-edge', 'adservice', 'unknown', 'dell2018', 'cartservice-edge'],
            'shippingservice-edge': ['izbp16opgy3xucvexwqp9dz', 'izbp1gwb52uyj3g0wn52lfz', 'istio-ingressgateway', 'productcatalogservice-edge', 'adservice-edge', 'izbp1gwb52uyj3g0wn52lez', 'dell2018', 'unknown', 'cartservice-edge'],
        }


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
