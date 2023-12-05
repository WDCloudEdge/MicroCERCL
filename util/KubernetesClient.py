from kubernetes import client, config

from Config import Config
from Config import Node
from Config import Pod
from typing import Dict, List
import time
import json
import schedule
from datetime import datetime, timezone, timedelta


class KubernetesClient:
    def __init__(self, project_config: Config):
        self.namespace = project_config.namespace
        # self.k8s_yaml = project_config.k8s_yaml
        config.kube_config.load_kube_config(config_file=project_config.k8s_config)
        self.core_api = client.CoreV1Api()  # namespace,pod,service,pv,pvc
        self.apps_api = client.AppsV1Api()  # deployment

    # Get all nodes
    def get_nodes(self):
        ret = self.core_api.list_node()
        nodes = []
        for i in ret.items:
            status = 'NotReady'
            for cond in i.status.conditions:
                if cond.type == 'Ready':
                    if cond.status == 'True':
                        status = 'Ready'
            nodes.append(
                Node(i.metadata.name, i.metadata.annotations['flannel.alpha.coreos.com/public-ip'], i.metadata.name,
                     i.spec.pod_cidr, status, i.metadata.labels['apps.openyurt.io/nodepool']
                     ))
        return nodes

    def get_node_center(self) -> Dict[str, str]:
        nodes = self.get_nodes()
        return {node.node_name: node.center for node in nodes}

    def get_node_ns_pods(self, node, namespaces):
        converted_pods = []
        pods = []
        for namespace in namespaces:
            field_selector = f"spec.nodeName={node.node_name},metadata.namespace={namespace}"
            pods.append(self.core_api.list_pod_for_all_namespaces(field_selector=field_selector))

        # 打印 Pod 信息
        for pod_item in pods:
            for pod in pod_item.items:
                converted_pods.append(
                    Pod(pod.spec.node_name, pod.metadata.namespace, pod.status.host_ip, pod.status.pod_ip,
                        pod.metadata.name, node.center))
        return converted_pods

    def get_pod_node(self, namespaces: List) -> Dict[str, Pod]:
        pod_node: Dict[str, Pod] = {}
        nodes = self.get_nodes()
        for node in nodes:
            converted_pods: [Pod] = self.get_node_ns_pods(node, namespaces)
            pod_node.update({pod.name: pod.node for pod in converted_pods})
        return pod_node

    # Get all microservices
    def get_svcs(self):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        svcs = [i.metadata.name for i in ret.items if i.metadata.name != 'loadgenerator']
        svcs.sort()
        return svcs

    # Get stateless microservices（exclude redis，mq，mongo，db）
    def get_svcs_without_state(self):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)

        def judge_state_svc(svc):
            state_svcs = ['redis', 'rabbitmq', 'mongo', 'mysql']
            for state_svc in state_svcs:
                if state_svc in svc:
                    return True
            return False

        svcs = [i.metadata.name for i in ret.items if not judge_state_svc(i.metadata.name)]
        svcs.sort()
        return svcs

    def get_svcs_counts(self):
        dic = {}
        pod_ret = self.core_api.list_namespaced_pod(self.namespace, watch=False)
        svcs = self.get_svcs()
        for svc in svcs:
            dic[svc] = 0
            for i in pod_ret.items:
                if i.metadata.name.find(svc) != -1:
                    dic[svc] = dic[svc] + 1
        return dic

    def get_svc_count(self, svc):
        ret_deployment = self.apps_api.read_namespaced_deployment_scale(svc, self.namespace)
        return ret_deployment.spec.replicas

    def all_avaliable(self):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        for item in ret.items:
            if item.status.ready_replicas != item.spec.replicas:
                return False
        return True

    # Determine the status of the service (avaliable?)
    def svcs_avaliable(self, svcs):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        items = [item for item in ret.items if item.metadata.name == 'svc']
        for item in ret.items:
            if item.metadata.name in svcs and item.status.ready_replicas != item.spec.replicas:
                return False
        return True

    def patch_scale(self, svc, count):
        body = {'spec': {'replicas': count}}
        self.apps_api.patch_namespaced_deployment_scale(svc, self.namespace, body)

    # def update_yaml(self):
    #     os.system('kubectl apply -f %s > temp.log' % self.k8s_yaml)


def collect_pod_topology(begin_timestamp, dir, client: KubernetesClient, namespaces, interval=5):
    now = int(time.time())
    time_schedule: int
    if begin_timestamp > now:
        time_schedule = begin_timestamp
    elif begin_timestamp == now:
        time_schedule = begin_timestamp + interval
    elif begin_timestamp < now:
        time_schedule = now + (interval - ((now - begin_timestamp) % interval))
    date = datetime.fromtimestamp(time_schedule, tz=timezone(timedelta(hours=8)))
    date_string = date.strftime('%Y-%m-%d %H:%M:%S')

    def job():
        print("Pod Topology Job is running...")
        with open(dir + '/pod_topology.result', "a") as output_file:
            print(datetime.now().strptime() + ':\n' + json.dumps(client.get_pod_node(namespaces)), file=output_file)

    schedule.every(interval).seconds.at(date_string).do(job).tag('Pod Topology Job')


if __name__ == '__main__':
    config_local = Config()
    config_local.k8s_config = '../local-config'
    client = KubernetesClient(config_local)
    print(client.get_nodes())
    # print(client.get_node_ns_pods("izbp16opgy3xucvexwqp9dz", ["bookinfo"]))
