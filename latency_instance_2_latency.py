import pandas as pd

df = pd.read_csv('data/MicroIRC/hipster/latency_instance.csv')

# 遍历列，找出前缀相同的列，并计算平均值
result = {}

for col in df.columns:
    if 'timestamp' == col: continue
    prefix = col.split('&')[0].split('-')[0]  # 假设前缀是通过下划线分隔的
    if prefix not in result:
        result[prefix] = df.filter(like=prefix).mean(axis=1)

# 将结果合并为一个新的 DataFrame
result_df = pd.DataFrame(result)
result_df['timestamp'] = df['timestamp']
result_df = result_df.set_index('timestamp')
result_df.to_csv('data/MicroIRC/hipster/latency.csv')

# 打印结果
print(result_df)
