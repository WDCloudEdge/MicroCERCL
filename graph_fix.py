import csv
import datetime

data = """
frontend	adservice
checkoutservice	cartservice
frontend	cartservice
frontend	checkoutservice
checkoutservice	currencyservice
frontend	currencyservice
checkoutservice	emailservice
checkoutservice	paymentservice
checkoutservice	productcatalogservice
frontend	productcatalogservice
recommendationservice	productcatalogservice
frontend	recommendationservice
checkoutservice	shippingservice
frontend	shippingservice
adservice-7659d48d84-xdcfs	192.168.31.101:9100
cartservice-75d494679c-qdl5c	192.168.31.101:9100
cartservice-75d494679c-vfldf	192.168.31.101:9100
checkoutservice-7d8cb45794-hc2rj	192.168.31.101:9100
checkoutservice-7d8cb45794-vfnmh	192.168.31.101:9100
currencyservice-588fc9584d-c9hrl	192.168.31.101:9100
currencyservice-588fc9584d-cl9dm	192.168.31.101:9100
currencyservice-588fc9584d-l8vxn	192.168.31.101:9100
emailservice-8848674-b7vr4	192.168.31.101:9100
emailservice-8848674-j9hwb	192.168.31.101:9100
frontend-875b86bb8-lq7z8	192.168.31.101:9100
frontend-875b86bb8-wvlvq	192.168.31.101:9100
paymentservice-6879f6c8c4-8pbjc	192.168.31.101:9100
productcatalogservice-5ff5f57dc8-8z7xf	192.168.31.101:9100
productcatalogservice-5ff5f57dc8-9zfh2	192.168.31.101:9100
productcatalogservice-5ff5f57dc8-mpw5r	192.168.31.101:9100
productcatalogservice-5ff5f57dc8-sckzn	192.168.31.101:9100
productcatalogservice-5ff5f57dc8-sv4x2	192.168.31.101:9100
recommendationservice-689548cfbd-ml8h2	192.168.31.101:9100
shippingservice-589dc45c5d-h7pnk	192.168.31.101:9100
shippingservice-589dc45c5d-sfzzg	192.168.31.101:9100
shippingservice-589dc45c5d-xdx28	192.168.31.101:9100
"""

# 将数据按行切分
lines = data.strip().split('\n')

# 设置起始时间和结束时间
start_time = datetime.datetime(2022, 7, 22, 2, 47, 51)
end_time = datetime.datetime(2022, 7, 22, 2, 57, 51)

# 设置时间间隔
time_interval = 5  # 5秒

# 打开 CSV 文件，准备写入
csv_filename = 'output.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['source', 'destination', 'timestamp']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    # 从起始时间开始每隔5秒写入一行
    current_time = start_time
    while current_time <= end_time:
        # 格式化时间字符串
        formatted_time = current_time.strftime("%Y/%m/%d %H:%M:%S")

        # 写入 CSV 文件
        for line in lines:
            print(f"{line}\t{formatted_time}")
            writer.writerow({'source': line.split('\t')[0], 'destination': line.split('\t')[1], 'timestamp': formatted_time})

        # 输出到控制台
        print(formatted_time)

        # 增加时间间隔
        current_time += datetime.timedelta(seconds=time_interval)
