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

[Dropbox](https://www.dropbox.com/scl/fi/s6gugabhlfd4ar46vu3nf/abnormal.zip?rlkey=iztl9kqkorakqt6dxocmlv3k7&st=jsbbcozk&dl=0)

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