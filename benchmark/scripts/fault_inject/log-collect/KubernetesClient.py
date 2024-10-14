import os
from kubernetes import client, config

import Config
from Config import Node
from Config import Pod


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
                     i.spec.pod_cidr, status))
        return nodes

    def get_ns_pods(self, namespaces):
        converted_pods = []
        pods = []
        for namespace in namespaces:
            field_selector = f"metadata.namespace={namespace}"
            pods.append(self.core_api.list_pod_for_all_namespaces(field_selector=field_selector))

        # 打印 Pod 信息
        for pod_item in pods:
            for pod in pod_item.items:
                converted_pods.append(
                    Pod(pod.spec.node_name, pod.metadata.namespace, pod.status.host_ip, pod.status.pod_ip,
                        pod.metadata.name))
        return converted_pods

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

    def get_svc_list_name(self):
        result = []
        responses = self.core_api.list_namespaced_service(self.namespace)
        for i in responses.items:
            result.append(i.metadata.name)
        return result

    def get_all_svc(self):
        namespaces = ['bookinfo', 'hipster', 'hipster2', 'cloud-sock-shop', 'horsecoder-test', 'trainticket']
        namespace_svc_dict = {}
        for namespace in namespaces:
            responses = self.core_api.list_namespaced_service(namespace)
            svc_list = []
            for i in responses.items:
                svc_list.append(i.metadata.name)
            namespace_svc_dict[namespace] = svc_list

        return namespace_svc_dict

    def pod_exist(self, namespaces, pod):
        pods = self.get_ns_pods(namespaces)
        for pod in pods:
            if pod.name == pod:
                return True
        return False

    # def update_yaml(self):
    #     os.system('kubectl apply -f %s > temp.log' % self.k8s_yaml)
