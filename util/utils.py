from sklearn import preprocessing
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
    return int(dt_object.timestamp())


def timestamp_2_time_string(timestamp):
    dt_object = datetime.utcfromtimestamp(timestamp)
    return dt_object.strftime("%Y-%m-%d %H:%M:%S")


def timestamp_2_time_string_beijing(timestamp):
    beijing_tz = pytz.timezone('Asia/Shanghai')
    dt_object = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.utc)
    dt_object = dt_object.astimezone(beijing_tz)
    return dt_object.strftime("%Y-%m-%d %H:%M:%S")


def time_string_2_timestamp_beijing(time_string):
    beijing_tz = pytz.timezone('Asia/Shanghai')
    dt_object = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
    dt_object = beijing_tz.localize(dt_object)
    return int(dt_object.timestamp())


def normalize_dataframe(data):
    data_without_time = data.drop(['timestamp'], axis=1)
    normalized_data = preprocessing.normalize(data_without_time.values, axis=0)
    normalized_df = pd.DataFrame(normalized_data, columns=data_without_time.columns)
    normalized_df['timestamp'] = data['timestamp']
    return normalized_df


def normalize_series(data):
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
            end_index = index - 1
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
        if ('edge' in root_cause and 'edge' in key and key in root_cause) or ('edge' not in root_cause and 'edge' not in key and (key in root_cause or root_cause in key)):
            is_top_k = True
    print(f"top_k: {top_k}", file=output_file)
    print(f"root_cause: {root_cause}, top_k: {top_k}")
    return top_k


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


def print_pr_rank(nums, rank, file):
    pr = 0
    fill_nums = []
    for num in nums:
        if num != 0:
            fill_nums.append(num)
    for num in fill_nums:
        if num <= rank:
            pr += 1
    pr_ = round(pr / len(fill_nums), 3)
    with open(file, "a") as output_file:
        print('PR@' + str(rank) + ': ' + str(pr_), file=output_file)
    return pr_


def print_pr(nums, file):
    pr_1 = print_pr_rank(nums, 1, file)
    pr_2 = print_pr_rank(nums, 2, file)
    pr_3 = print_pr_rank(nums, 3, file)
    pr_4 = print_pr_rank(nums, 4, file)
    pr_5 = print_pr_rank(nums, 5, file)
    pr_6 = print_pr_rank(nums, 6, file)
    pr_7 = print_pr_rank(nums, 7, file)
    pr_8 = print_pr_rank(nums, 8, file)
    pr_9 = print_pr_rank(nums, 9, file)
    pr_10 = print_pr_rank(nums, 10, file)
    avg_1 = pr_1
    avg_3 = round((pr_1 + pr_2 + pr_3) / 3, 3)
    avg_5 = round((pr_1 + pr_2 + pr_3 + pr_4 + pr_5) / 5, 3)
    avg_10 = round((pr_1 + pr_2 + pr_3 + pr_4 + pr_5 + pr_6 + pr_7 + pr_8 + pr_9 + pr_10) / 10, 3)
    with open(file, "a") as output_file:
        print('AVG@1:' + str(avg_1), file=output_file)
        print('AVG@3:' + str(avg_3), file=output_file)
        print('AVG@5:' + str(avg_5), file=output_file)
        print('AVG@10:' + str(avg_10), file=output_file)
    return pr_1, pr_3, pr_5, pr_10, avg_1, avg_3, avg_5, avg_10
