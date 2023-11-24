# SPDX-License-Identifier: MIT

import json
import logging
import sys
import os
from os.path import dirname
from util.KubernetesClient import KubernetesClient
from Config import Config
from util.utils import *
from datetime import datetime

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# persistence_type = "NONE"
# persistence_type = "REDIS"
# persistence_type = "KAFKA"
persistence_type = "FILE"

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

if persistence_type == "KAFKA":
    from drain3.kafka_persistence import KafkaPersistence

    persistence = KafkaPersistence("drain3_state", bootstrap_servers="localhost:9092")

elif persistence_type == "FILE":
    from drain3.file_persistence import FilePersistence

    persistence = FilePersistence("data/normal/20231120/tcpdump_logs/drain3_state.bin")

elif persistence_type == "REDIS":
    from drain3.redis_persistence import RedisPersistence

    persistence = RedisPersistence(redis_host='',
                                   redis_port=25061,
                                   redis_db=0,
                                   redis_pass='',
                                   is_ssl=True,
                                   redis_key="drain3_state_key")
else:
    persistence = None


class Pair:
    def __init__(self, caller: str, callee: str, protocol: str):
        self.caller = caller
        self.callee = callee
        self.protocol = protocol
        self.caller_ip = caller[:caller.rfind('.')]
        self.callee_ip = callee[:callee.rfind('.')]
        self.is_pair = False

    def base_key(self):
        return self.caller + '-' + self.callee + '-' + self.protocol

    def pair(self):
        self.is_pair = True

    def get_pair(self):
        return self.is_pair


class Node_Pair(Pair):
    def __init__(self, caller: str, callee: str):
        super().__init__(caller, callee, 'node')

    def base_key(self):
        return self.caller + '-' + self.callee


class TCP_Pair(Pair):
    def __init__(self, caller: str, callee: str, protocol: str, is_seq_or_ack: bool, key_num: int, length: int = 0):
        super().__init__(caller, callee, protocol)
        self.is_seq_or_ack = is_seq_or_ack
        self.key_num = key_num
        self.length = length

    def base_key(self):
        return super().base_key() + '-' + self.key_num

    def reverse_key(self):
        return self.callee + '-' + self.caller + '-' + self.protocol + '-' + self.key_num


class UDP_Pair(Pair):
    def __init__(self, caller: str, callee: str, protocol: str, length: int, time):
        super().__init__(caller, callee, protocol)
        self.length = length
        self.timestamp = datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f").timestamp()

    def reverse_key(self):
        return self.callee + '-' + self.caller + '-' + self.protocol

    def within(self, timestamp):
        return abs(int(timestamp - self.timestamp)) <= 10


config = TemplateMinerConfig()
config.load(f"{dirname(__file__)}/drain3.ini")
config.profiling_enabled = False

template_miner = TemplateMiner(persistence, config)
print(f"Drain3 started with '{persistence_type}' persistence")
print(f"{len(config.masking_instructions)} masking instructions are in use")
print(f"Starting training mode. Reading from std-in ('q' to finish)")

folder_path = 'data/normal/test/tcpdump_logs'


# folder_path = 'data/normal/20231120/tcpdump_logs'


def get_dir_files(folder_path):
    files = os.listdir(folder_path)
    file_paths = []
    # 遍历文件列表
    for file_name in files:
        # 拼接文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        # 判断是否为文件（而非文件夹）
        if os.path.isfile(file_path) and '.log' in file_name:
            file_paths.append(file_path)
        elif os.path.isdir(file_path):
            file_paths_dir = get_dir_files(file_path)
            for path in file_paths_dir:
                file_paths.append(path)
    return file_paths


# 获取文件夹中所有文件的列表
file_paths = get_dir_files(folder_path)

pair_candidate = {}
udp_pair_candidate = {}
length_pair_candidate = {}
# 打开文件，'r' 表示只读模式
for file_path in file_paths:
    with open(file_path, 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines()
        # 遍历每一行
        for log_line in lines:
            # 处理每一行的内容
            # print(log_line.strip())  # strip() 用于移除行末尾的换行符
            if log_line == 'q':
                break
            result = template_miner.add_log_message(log_line)
            result_json = json.dumps(result)
            # print(result_json)
            template = result["template_mined"]
            params = template_miner.extract_parameters(template, log_line)
            seq_ack_num = None
            ack_num = None
            length = 0
            caller_ip = None
            callee_ip = None
            timestamp = None
            if 'TIMESTAMP' in template and ('TCP' in template or 'UDP' in template):
                for param in params:
                    if 'PAIR' == param.mask_name:
                        value = param.value
                        ip_pair = value.split('>')
                        if len(ip_pair) > 1:
                            caller_ip = ip_pair[0].strip()
                            callee_ip = ip_pair[1].strip()
                    elif 'TCP' == param.mask_name:
                        protocol = 'TCP'
                        value = param.value
                        if 'seq' in value:
                            seq_string = value.split(',')[0].split(' ')[1]
                            seq_ack_num = seq_string.split(':')[1]
                        else:
                            ack_num = value.split(',')[0].split(' ')[1]
                        length = value[value.rfind(',') + 1:].strip().split(' ')[1]
                    elif 'UDP' == param.mask_name:
                        protocol = 'UDP'
                        value = param.value
                        length = value.split(',')[1].strip().split(' ')[1]
                    elif 'TIMESTAMP' == param.mask_name:
                        timestamp = param.value
                if caller_ip and callee_ip and protocol:
                    if seq_ack_num:
                        tcp_pair = TCP_Pair(caller_ip, callee_ip, protocol, True, seq_ack_num, length)
                        pair_candidate[tcp_pair.base_key()] = tcp_pair
                    if ack_num:
                        tcp_pair = TCP_Pair(caller_ip, callee_ip, protocol, False, ack_num, length)
                        seq_tcp_pair = pair_candidate.get(tcp_pair.reverse_key(), None)
                        if not seq_tcp_pair:
                            continue
                        seq_tcp_pair.pair()
                        pair_length = length_pair_candidate.get(seq_tcp_pair.base_key(), 0)
                        length_pair_candidate[seq_tcp_pair.base_key()] = pair_length + int(seq_tcp_pair.length)
                    if length and protocol == 'UDP':
                        udp_pair = UDP_Pair(caller_ip, callee_ip, protocol, length, '2023-11-20 ' + timestamp)
                        seq_udp_pair = pair_candidate.get(udp_pair.reverse_key(), None)
                        # todo 乱序成对UDP权重有误，例如：
                        # 13:11:50.162743 IP 10.244.13.59.50679 > 10.244.12.152.45414: UDP, length 674
                        # 13:11:50.178542 IP 10.244.13.59.50679 > 10.244.12.152.45414: UDP, length 342
                        # 13:11:50.184329 IP 10.244.12.152.45414 > 10.244.13.59.50679: UDP, length 66
                        # 13:11:50.199678 IP 10.244.12.152.45414 > 10.244.13.59.50679: UDP, length 66
                        # udp没有匹配成功
                        if not seq_udp_pair or not seq_udp_pair.within(udp_pair.timestamp):
                            old_udp_pair = pair_candidate.get(udp_pair.base_key(), None)
                            # 覆盖之前相同key的pair，需要统计上一次是否匹配成功
                            if old_udp_pair:
                                pair_count = udp_pair_candidate.get(old_udp_pair.base_key(), 1)
                                if not old_udp_pair.get_pair():
                                    udp_pair_candidate[old_udp_pair.base_key()] = pair_count - 1
                            pair_candidate[udp_pair.base_key()] = udp_pair
                        else:
                            seq_udp_pair.pair()
                            new_pair_count = udp_pair_candidate.get(seq_udp_pair.base_key(), 0)
                            udp_pair_candidate[seq_udp_pair.base_key()] = new_pair_count + 1
                            pair_length = length_pair_candidate.get(seq_udp_pair.base_key(), 0)
                            length_pair_candidate[seq_udp_pair.base_key()] = pair_length + int(length)

print("Training done. Mined clusters:")
for cluster in template_miner.drain.clusters:
    with open(folder_path + '/template.result', "a") as output_file:
        print(cluster, file=output_file)

nodes = KubernetesClient(Config()).get_nodes()
node_pair = {}
length_node_pair = {}
for v in pair_candidate:
    with open(folder_path + '/network_pair.result', "a") as output_file:
        if udp_pair_candidate.get(v, 0) > 1:
            pair_candidate[v].pair()
        print(v + ':' + str(pair_candidate[v].get_pair()) + ':' + str(length_pair_candidate.get(v, 0)),
              file=output_file)
    pair = pair_candidate[v]
    if pair.get_pair():
        caller_node = None
        callee_node = None
        for node in nodes:
            if pair.caller_ip == node.ip or \
                    ip_2_subnet(pair.caller_ip, node.cni_ip[node.cni_ip.rfind('/') + 1:]) == \
                    node.cni_ip[:node.cni_ip.rfind('/')]:
                caller_node = node.node_name
            if pair.callee_ip == node.ip or \
                    ip_2_subnet(pair.callee_ip, node.cni_ip[node.cni_ip.rfind('/') + 1:]) == \
                    node.cni_ip[:node.cni_ip.rfind('/')]:
                callee_node = node.node_name
        new_pair = Node_Pair(caller_node, callee_node)
        pair_count = node_pair.get(new_pair.base_key(), 0)
        node_pair[new_pair.base_key()] = pair_count + 1
        pair_length = length_node_pair.get(new_pair.base_key(), 0)
        length_node_pair[new_pair.base_key()] = pair_length + length_pair_candidate.get(v, 0)
for p in node_pair:
    with open(folder_path + '/topology_pair.result', "a") as output_file:
        print(p + ':' + str(node_pair[p]) + ':' + str(length_node_pair.get(p, 0)), file=output_file)

# print(f"Starting inference mode, matching to pre-trained clusters. Input log lines or 'q' to finish")
# while True:
#     log_line = input("> ")
#     if log_line == 'q':
#         break
#     cluster = template_miner.match(log_line)
#     if cluster is None:
#         print(f"No match found")
#     else:
#         template = cluster.get_template()
#         print(f"Matched template #{cluster.cluster_id}: {template}")
#         print(f"Parameters: {template_miner.get_parameter_list(template, log_line)}")
