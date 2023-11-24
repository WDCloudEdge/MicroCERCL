import sys
from Config import Config
import MetricCollector
import time

if __name__ == "__main__":
    # namespaces = ['bookinfo', 'hipster', 'hipster2', 'sock-shop', 'horsecoder-test', 'horsecoder-minio']
    namespaces = ['bookinfo']
    config = Config()

    global_now_time = 1700289600
    global_end_time = 1700290200
    now = int(time.time())
    if global_now_time > now:
        sys.exit("begin time is after now time")
    if global_end_time > now:
        global_end_time = now

    folder = '.'
    for n in namespaces:
        config.namespace = n
        config.svcs.clear()
        config.pods.clear()
        count = 1
        now_time = global_now_time
        end_time = global_end_time
        data_folder = './data/' + str(config.user) + '/' + config.namespace
        while now_time < end_time:
            config.start = int(round(now_time))
            config.end = int(round(now_time + config.duration))
            if config.end > end_time:
                config.end = end_time
            print('第' + str(count) + '次获取 [' + config.namespace + '] 数据')
            MetricCollector.collect(config, data_folder)
            now_time += config.duration + 1
            config.pods.clear()
            count += 1
    data_folder = './data/' + str(config.user) + '/node'
    count = 1
    now_time = global_now_time
    end_time = global_end_time
    while now_time < end_time:
        config.start = int(round(now_time))
        config.end = int(round(now_time + config.duration))
        if config.end > end_time:
            config.end = end_time
        print('第' + str(count) + '次获取 [' + 'node' + '] 数据')
        MetricCollector.collect_node(config, data_folder)
        now_time += config.duration + 1
        count += 1
