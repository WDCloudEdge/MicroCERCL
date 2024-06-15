# MicroCERCL

## 简介

**MicroCERCL**从内核级日志中提取有效内容，优先定位内核级故障根因。此外，**MicroCERCL**构建了异构动态拓扑堆栈，基于指标Metrics数据训练图神经网络模型，从而在不依赖历史数据的情况下准确定位微服务应用程序级的故障根因。

## 快速开始

### 前置需要

+ 推荐使用Python3.7，其他Python3也可以兼容

+ Git

### 配置

```shell
git clone https://github.com/WDCloudEdge/MicroCERCL.git
cd MicroCERCL
python3.7 -m pip install -r requirements.txt
```

### 启动MicroCERCL

#### 参数配置

Change the dataset and other configs in `Config.py`

#### 执行

```shell
python3.7 ./main.py
```

## 数据集

### 下载链接

[Dropbox](https://www.dropbox.com/scl/fi/lw4xlw9b2rlhaa1ds0bju/abnormal_20240615.zip?rlkey=yuomecjids9qa54029755bjzo&st=qsrs7wfc&dl=0)

### 数据集描述

包含三个文件夹，分别对应混合部署场景中故障根因所在的 Bookinfo、Hipster 和 SockShop系统。根据不同的微服务（或其实例）故障原因，每个文件夹被进一步拆分为二级文件夹。每个根因服务文件夹都包含注入的所有故障的标签信息（xxx_label.txt）。在每个服务中，根据标签文件将其拆分为三级文件夹，形成故障样本。每个故障样本包含所有混合部署的微服务系统数据，构成第四级文件夹。每个混合部署的微服务系统文件夹包含三种类型的监控数据：指标、跟踪和日志（ Bookinfo中的每个故障样本不含日志数据）。
如图所示:

<img width="310" alt="image" src="https://github.com/WDCloudEdge/MicroCERCL/assets/48899336/461ec9a0-80c9-4fb1-a989-566cb14661e6">

### 故障样本

当发生故障时，一个故障样本包含混合部署的微服务系统的所有监控数据。
如图所示:

<img width="324" alt="image" src="https://github.com/WDCloudEdge/MicroCERCL/assets/48899336/346c2b81-371b-41ca-92de-ce99df51509e">

### 数据细节

#### Metrics

<img width="193" alt="image" src="https://github.com/WDCloudEdge/MicroCERCL/assets/48899336/6b7e5e22-0d5d-4629-9dbe-ca09c5894766">

| File             | Description                                                                         |
| ---------------- | ----------------------------------------------------------------------------------- |
| call.csv         | 微服务之间的时间序列调用延迟，包含P99、P95 和 P90，分别代表延迟数据的第 99、95 和 90 百分位数。                          |
| graph.csv        | 时间序列拓扑包含实例、实例所在的服务器和服务调用关系。                                                         |
| instance.csv     | 每个实例的时间序列指标，包括 CPU 使用量、内存使用量和网络传输包。                                                 |
| latency.csv      | 微服务的时序延迟，包含P99、P95 和 P90，分别代表延迟数据的第 99、95 和 90 百分位数。                                |
| resource.csv     | 特定命名空间内实例的时间序列指标数据，包括 CPU 使用总量和内存使用总量                                               |
| success_rate.csv | 微服务成功率时序数据                                                                          |
| svc_metric.csv   | 微服务的时间序列指标数据（其实例的平均值），包含 CPU 使用率、CPU 限制、内存使用率、内存限制、FS 写入、FS 读取、FS 使用率、网络接收、网络发送数据包。 |
| svc_qps.csv      | 微服务qps时序数据                                                                          |

#### Traces

<img width="217" alt="image" src="https://github.com/WDCloudEdge/MicroCERCL/assets/48899336/6ab3177b-3502-413c-b77f-4251387a3d20">

| File                  | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| abnormal.pkl          | 记录结构缺失、状态码异常、有报错信息的数据，不包含延时信息异常的数据 |
| abnormal_half.pkl     | 针对文件所在的命名空间，基于 abnormal.pkl 消除Trace数据中其他命名空间服务信息（仅包含本命名空间服务信息）后的数据 |
| inbound.pkl           | 针对文件所在的命名空间，记录包含其他命名空间服务调用本命名空间服务的Trace数据 |
| inbound_half.pkl      | 针对文件所在的命名空间，基于 inbound.pkl 消除Trace数据中其他命名空间服务信息（仅包含本命名空间服务信息）后的数据 |
| normal.pkl            | 记录结构完整、状态码正常的数据，包含延时信息异常的数据       |
| outbound.pkl          | 针对文件所在的命名空间，记录包含本命名空间服务调用其他命名空间服务的Trace数据 |
| outbound_half.pkl     | 针对文件所在的命名空间，基于 outbound.pkl 消除Trace数据中其他命名空间服务信息（仅包含本命名空间服务信息）后的数据 |
| trace_net_latency.pkl | 统计一对服务调用之间的请求延时数据和响应延时数据             |
| trace_pod_latency.pkl | 统计一对服务调用之间调用服务从发送请求到接收响应的延时数据   |

#### Logs

每个实例（容器）都有一个 .pkl 文件，其中包含容器的所有业务日志。

<img width="301" alt="image" src="https://github.com/WDCloudEdge/MicroCERCL/assets/48899336/e3bddbcf-8b6e-4b02-8f9c-1cbd6945cb43">

## 项目结构

```textile
McroCERCL/
│├── .gitignore
│├── Config.py
│├── MetricCollector.py
│├── README.md
│├── anomaly_detection.py
│├── graph.py
│├── log_parser.py
│├── main.py
│├── model.py
│├── model_aggregate.py
│├── requirements.txt
│└── util/
│└── │├── KubernetesClient.py
│└── │├── PrometheusClient.py
│└── │└── utils.py
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
