from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler, StandardScaler
import pandas as pd
from datetime import datetime
import pytz


def ip_2_subnet(ip: str, net_mask: int):
    mask_zero = 32 - int(net_mask)
    count = -1
    ip_split = ip.split('.')
    while mask_zero >= 1:
        if mask_zero > 8:
            mask_zero_turn = 8
        else:
            mask_zero_turn = mask_zero
        ip_split[count] = str(int(ip_split[count]) & int(bin(0)[2:].zfill(mask_zero_turn), 2))
        mask_zero -= 8
    return ".".join(ip_split)


def time_string_2_timestamp(time_string):
    dt_object = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc)
    # 使用 timestamp() 将 datetime 对象转换为时间戳
    return int(dt_object.timestamp())


def timestamp_2_time_string(timestamp):
    dt_object = datetime.utcfromtimestamp(timestamp)
    return dt_object.strftime("%Y-%m-%d %H:%M:%S")


def timestamp_2_time_string_beijing(timestamp):
    # 设置北京时区
    beijing_tz = pytz.timezone('Asia/Shanghai')

    # 将时间戳转换为 datetime 对象
    dt_object = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.utc)

    # 将 datetime 对象转换为北京时间
    dt_object = dt_object.astimezone(beijing_tz)

    return dt_object.strftime("%Y-%m-%d %H:%M:%S")


def time_string_2_timestamp_beijing(time_string):
    # 设置北京时区
    beijing_tz = pytz.timezone('Asia/Shanghai')

    # 将时间字符串转换为 datetime 对象
    dt_object = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')

    # 将 datetime 对象转换为北京时间
    dt_object = beijing_tz.localize(dt_object)

    # 使用 timestamp() 将 datetime 对象转换为时间戳
    return int(dt_object.timestamp())


def normalize_dataframe(data):
    # 获取 DataFrame 的列名
    data_without_time = data.drop(['timestamp'], axis=1)

    # 对每一列进行归一化操作
    normalized_data = preprocessing.normalize(data_without_time.values, axis=0)

    # 创建新的 DataFrame，使用原始列名
    normalized_df = pd.DataFrame(normalized_data, columns=data_without_time.columns)
    normalized_df['timestamp'] = data['timestamp']

    return normalized_df


def normalize_series(data):
    # scaler = StandardScaler()
    # normalized_data = scaler.fit_transform(data.values.reshape(-1, 1))
    normalized_data = preprocessing.normalize(data.fillna(0).values.reshape(-1, 1), axis=0)
    normalized_series = pd.Series(normalized_data.flatten())
    return normalized_series


def df_time_limit(df, begin_timestamp, end_timestamp):
    begin_index = 0
    end_index = 1

    max_timestamp = time_string_2_timestamp(df['timestamp'][df.shape[0] - 1])
    for index, row in df.iterrows():
        if time_string_2_timestamp(row['timestamp']) >= int(begin_timestamp):
            begin_index = index
            break
    for index, row in df.iterrows():
        if index > begin_index and time_string_2_timestamp(row['timestamp']) >= int(end_timestamp):
            end_index = index
            break
    if max_timestamp < int(end_timestamp):
        end_index = df.shape[0] - 1
    if time_string_2_timestamp(df.loc[end_index]['timestamp']) == int(end_timestamp):
        end_index += 1
    df = df.loc[begin_index:end_index]
    df = df.reset_index(drop=True)
    return df


def df_time_limit_normalization(df, begin_timestamp, end_timestamp):
    return normalize_dataframe(df_time_limit(df, begin_timestamp, end_timestamp).fillna(0))


def top_k_node(sorted_dict_node, root_cause, output_file):
    top_k = 0
    is_top_k = False
    for key, value in list(sorted_dict_node.items()):
        if not is_top_k:
            top_k += 1
        print(f"{key}: {value}", file=output_file)
        if ('-' in root_cause and key in root_cause) or root_cause in key:
            is_top_k = True
    print(f"top_k: {top_k}", file=output_file)
    print(f"root_cause: {root_cause}, top_k: {top_k}")


def top_k_node_time_series(sorted_dict_node_time_series, root_cause, output_file):
    top_k = 0
    is_top_k = False
    for key, value in list(sorted_dict_node_time_series.items()):
        if not is_top_k:
            top_k += 1
        print(f"{key}: {value}", file=output_file)
        if ('-' in root_cause and key[:key.rfind('-')] in root_cause) or root_cause in key[:key.rfind('-')]:
            is_top_k = True
    print(f"top_k: {top_k}", file=output_file)
