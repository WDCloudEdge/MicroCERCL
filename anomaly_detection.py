import numpy as np
from sklearn.cluster import Birch
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
from util.utils import df_time_limit_normalization, df_time_limit, normalize_series, time_string_2_timestamp


def get_anomaly_by_df(file_dir, begin_timestamp, end_timestamp):
    anomalies = []
    # todo 统计时间戳的严重程度
    anomaly_time_series = {}
    # read call latency data
    call_data = pd.read_csv(file_dir + '/' + 'call.csv')
    anomaly_svc_calls, anomaly_call_time_series_index = birch_ad_with_smoothing(
        df_time_limit_normalization(call_data, begin_timestamp, end_timestamp))
    anomaly_time_series = {**anomaly_time_series, **anomaly_call_time_series_index}
    a_svc_calls = [a[:a.rfind('&')].split('_') for a in anomaly_svc_calls]
    for a_svc in a_svc_calls:
        anomalies.extend(a_svc)
    # read svc latency data
    latency_data = pd.read_csv(file_dir + '/' + 'latency.csv')
    anomaly_svcs, anomaly_svc_time_series_index = birch_ad_with_smoothing(
        df_time_limit_normalization(latency_data, begin_timestamp, end_timestamp))
    anomaly_time_series = {**anomaly_time_series, **anomaly_svc_time_series_index}
    anomalies.extend([a_svc[:a_svc.rfind('&')] for a_svc in anomaly_svcs])
    # qps data
    qps_file_name = file_dir + '/' + 'svc_qps.csv'
    qps_source_data = pd.read_csv(qps_file_name)
    anomaly_qps, anomaly_qps_time_series_index = birch_ad_with_smoothing(
        df_time_limit_normalization(qps_source_data, begin_timestamp, end_timestamp))
    anomaly_time_series = {**anomaly_time_series, **anomaly_qps_time_series_index}
    anomalies.extend([a_svc for a_svc in anomaly_qps])
    # success rate data
    success_rate_file_name = file_dir + '/' + 'success_rate.csv'
    success_rate_source_data = pd.read_csv(success_rate_file_name)
    anomaly_success_rate, anomaly_success_rate_time_series_index = birch_ad_with_smoothing(
        df_time_limit_normalization(success_rate_source_data, begin_timestamp, end_timestamp))
    anomaly_time_series = {**anomaly_time_series, **anomaly_success_rate_time_series_index}
    anomalies.extend([a_svc for a_svc in anomaly_success_rate])
    # instances data
    instance_file_name = file_dir + '/' + 'instance.csv'
    instance_source_data = pd.read_csv(instance_file_name)
    anomalies_index = df_time_limit_normalization_ctn_anomalies_with_index(instance_source_data, begin_timestamp,
                                                                           end_timestamp)
    anomalies.extend(a_instance[:a_instance.rfind('_')] for a_instance in anomalies_index.keys())
    anomalies = list(set(anomalies))
    if 'istio-ingressgateway' in anomalies:
        anomalies.remove('istio-ingressgateway')
    a_instance_time_series = {}
    for a_instance in anomalies_index:
        a_i = a_instance_time_series.get(a_instance[:a_instance.rfind('_')], [])
        a_i.extend(anomalies_index[a_instance])
        a_instance_time_series[a_instance[:a_instance.rfind('_')]] = list(set(a_i))
    anomaly_time_series = {**anomaly_time_series, **a_instance_time_series}
    return anomalies, anomaly_time_series


def birch_ad_with_smoothing(df, threshold=0.1, smoothing_window=12, n=5):
    # anomaly detection on response time of service invocation.
    # input: response times of service invocations, threshold for birch clustering
    # output: anomalous service invocation
    df_time_index, df_index_time = get_timestamp_index(df)
    anomalies = []
    anomaly_time_series_index = {}
    for node, metrics in df.iteritems():
        # No anomaly detection in db
        # if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
        if node != 'timestamp' and 'Unnamed' not in node and 'node' not in node and 'tcp' not in node:

            metrics = metrics.rolling(
                window=smoothing_window, min_periods=1).mean()
            x = np.array(metrics)
            x = np.where(np.isnan(x), 0, x)
            normalized_x = preprocessing.normalize([x])

            X = normalized_x.reshape(-1, 1)
            brc = Birch(branching_factor=50, n_clusters=None,
                        threshold=threshold, compute_labels=True)
            brc.fit(X)
            brc.predict(X)

            labels = brc.labels_
            n_clusters = np.unique(labels).size
            if n_clusters > 1:
                anomalies.append(node)
                # time_label_index = {}
                # for idx, label in enumerate(labels):
                #     time_labels = time_label_index.get(label, [])
                #     time_labels.append(idx)
                #     time_label_index[label] = time_labels
                mean_vector = np.mean(X, axis=0)
                # 计算每个样本到整体平均的距离
                distances_to_cluster_centers = pairwise_distances(X, [mean_vector]).flatten()
                # distances_to_cluster_centers = pairwise_distances_argmin_min(X, brc.subcluster_centers_)[1]
                # 找到最离群的前n个点的索引
                if len(labels) < n:
                    n_outlying_indices = [i for i in range(len(labels))]
                else:
                    n_outlying_indices = np.argsort(distances_to_cluster_centers)[-n:]
                # time_label_index = {}
                # for idx, label in enumerate(labels):
                #     time_labels = time_label_index.get(label, [])
                #     time_labels.append(idx)
                #     time_label_index[label] = time_labels
                # anomaly_time_series_index.extend(time_label_index[len(brc.subcluster_centers_) - 1])
                # anomaly_time_series_index[node] = df_index_time[time_label_index[len(brc.subcluster_centers_) - 1]]
                anomaly_time_series_index[node] = df_index_time[n_outlying_indices]
    return anomalies, anomaly_time_series_index


def birch_ad_with_smoothing_series(series, threshold=0.1, smoothing_window=12, n=5):
    anomaly_time_series_index = []
    metrics = normalize_series(series)
    metrics = metrics.rolling(
        window=smoothing_window, min_periods=1).mean()
    x = np.array(metrics)
    x = np.where(np.isnan(x), 0, x)
    normalized_x = preprocessing.normalize([x])

    X = normalized_x.reshape(-1, 1)
    brc = Birch(branching_factor=50, n_clusters=None,
                threshold=threshold, compute_labels=True)
    brc.fit(X)
    brc.predict(X)

    labels = brc.labels_
    n_clusters = np.unique(labels).size
    is_anomaly = False
    if n_clusters > 1:
        is_anomaly = True
        mean_vector = np.mean(X, axis=0)
        # todo 实例不健康时，是否需要别的处理策略
        # 计算每个样本到整体平均的距离
        distances_to_cluster_centers = pairwise_distances(X, [mean_vector]).flatten()
        # distances_to_cluster_centers = pairwise_distances_argmin_min(X, brc.subcluster_centers_)[1]
        # 找到最离群的前n个点的索引
        if len(labels) < n:
            n_outlying_indices = [i for i in range(len(labels))]
        else:
            n_outlying_indices = np.argsort(distances_to_cluster_centers)[-n:]
        # time_label_index = {}
        # for idx, label in enumerate(labels):
        #     time_labels = time_label_index.get(label, [])
        #     time_labels.append(idx)
        #     time_label_index[label] = time_labels
        # anomaly_time_series_index.extend(time_label_index[len(brc.subcluster_centers_) - 1])
        anomaly_time_series_index.extend(n_outlying_indices)
    return is_anomaly, list(set(anomaly_time_series_index))


def df_time_limit_normalization_ctn_anomalies_with_index(df, begin_timestamp, end_timestamp):
    df = df_time_limit(df, begin_timestamp, end_timestamp)
    df_time_index, df_index_time = get_timestamp_index(df)
    anomalies_index = {}
    for node, metrics in df.iteritems():
        if not node == 'timestamp':
            series_list = []
            boundary_index = (metrics != -1)
            if_exist = False
            index = []
            for i, a in enumerate(boundary_index):
                if not if_exist and a:
                    if_exist = True
                    index.append(i)
                elif if_exist and not a:
                    if_exist = False
                    index.append(i - 1)
            if if_exist:
                index.append(len(boundary_index) - 1)
            if len(index) == 2 and index[0] == 0 and index[1] == len(boundary_index) - 1:
                prune_series = metrics
            else:
                for i in range(0, len(index) - 1, 2):
                    series_list.append(metrics[index[i]:index[i + 1] + 1])
                prune_series = pd.concat(series_list, axis=0)
            is_anomaly, anomaly_time_series_index = birch_ad_with_smoothing_series(prune_series)
            if is_anomaly:
                anomalies_index[node] = [df_index_time[idx + index[0]] for idx in anomaly_time_series_index]
    return anomalies_index


def get_timestamp_index(df):
    df_time_index = {}
    df_index_time = {}
    for node, metrics in df.iteritems():
        if node == 'timestamp':
            for t_index, t in enumerate(metrics):
                df_time_index[t] = t_index
                df_index_time[t_index] = t
            break
    return df_time_index, df_index_time
