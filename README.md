# MicroCERCL

## Description

**MicroCERCL** extracts valid contents from kernel-level logs to prioritize localizing the kernel-level root cause. Moreover, **MicroCERCL** constructs a heterogeneous dynamic topology stack and train a graph neural network model to accurately localize the application-level root cause without relying on historical data.

## Quick Start

### Requirement

+ Python3.7 is recommended. Otherwise, any python3 version should be fine.

+ Git

### Setup

```shell
git clone https://github.com/WDCloudEdge/MicroCERCL.git
cd MicroCERCL
python3.7 -m pip install -r requirements.txt
```

### Running MicroCERCL

#### Config

Change the dataset and other configs in `Config.py`

#### Execute

```shell
python3.7 ./main.py
```

## Dataset

### Download

[Dropbox](https://www.dropbox.com/scl/fi/s6gugabhlfd4ar46vu3nf/abnormal.zip?rlkey=iztl9kqkorakqt6dxocmlv3k7&st=jsbbcozk&dl=0)

### Description

It contains three folders corresponding to Bookinfo, Hipster, and SockShop, where the root cause is located within a hybrid deployment scenario. Each folder is further split into secondary folders based on the root cause of the microservice (or its instances). Each root cause service folder contains label information (xxx_label.txt) for all failures injected. Within each service, it is split into third-level folders according to the label file to form a failure sample. Each failure sample contains all hybrid-deployed microservice systems that form the fourth-level folders. Each hybrid-deployed microservice system folder contains three types of monitoring data: metrics, traces, and logs. (Bookinfo without logs in each failure sample)

### Failure Sample

### Details

#### Metrics



#### Traces

#### Logs

## Project Structure

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