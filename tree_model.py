import pandas as pd
import numpy as np
from itertools import groupby
from tqdm import tqdm
import seaborn as sns
import time
import gc
# from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

input_dir = "wsdm_train_data/"
output_dir = "wsdm_model_data/"

launch = pd.read_csv(input_dir + "app_launch_logs.csv")

launch['user_count'] = launch['user_id'].map(launch['user_id'].value_counts())
launch['date_count'] = launch['date'].map(launch['date'].value_counts())
launch_grp = launch.groupby("user_id").agg(
    launch_date=("date", list),
    launch_type=("launch_type", list),
    user_count=("user_count", np.max),
    launch_data_count=("date_count", list)
).reset_index()
del launch
gc.collect()

launch_grp['date_len'] = launch_grp.launch_date.apply(lambda x: max(x)-min(x))

def choose_end_date(launch_date):
    n1, n2 = min(launch_date), max(launch_date)
    if n1 < n2 - 7:
        end_date = np.random.randint(n1, n2 - 7)
    else:
        end_date = np.random.randint(100, 222 - 7)
    return end_date
launch_grp["end_date"] = launch_grp.launch_date.apply(choose_end_date)

def get_label(row):
    launch_list = row.launch_date
    end = row.end_date
    label = sum([1 for x in set(launch_list) if end < x < end+8])
    return label
launch_grp["label"] = launch_grp.apply(get_label, axis=1)

train = launch_grp[["user_id","end_date","label"]]
test = pd.read_csv("test-a.csv")
test["label"] = -1
data = pd.concat([train, test], ignore_index=True)

# append test data to launch_grp
launch_grp = launch_grp.append(
    test.merge(launch_grp[["user_id", "launch_type", "launch_date","user_count","launch_data_count","date_len"]], how="left", on="user_id")
)
# launch_grp = reduce_mem(launch_grp)

# get latest 32 days([end_date-31, end_date]) launch type sequence
# 0 for not launch, 1 for launch_type=0, and 2 for launch_type=1
def gen_launch_seq(row):
    seq_sort = sorted(zip(row.launch_type, row.launch_date), key=lambda x: x[1])
    seq_map = {k: max(g)[0] + 1 for k, g in groupby(seq_sort, lambda x: x[1])}
    end = row.end_date
    seq = [seq_map.get(x, 0) for x in range(end-31, end+1)]
    return seq

#日活用户数量序列
def gen_launch_count_seq(row):
    seq_map = dict(zip(row.launch_date, row.launch_data_count))
    end = row.end_date
    seq = [seq_map.get(x, 0) for x in range(end-31, end+1)]
    return seq
launch_grp["launch_seq"] = launch_grp.apply(gen_launch_seq, axis=1)
launch_grp["launch_count_seq"] = launch_grp.apply(gen_launch_count_seq, axis=1)

#计算end_date前x天的用户登陆天数
#这里可以构造更多的序列，比如用户每日观看视频时长序列，观看视频完播率序列，每日观看视频个数序列等等序列
#这里可以操作的空间还有很多
x_list = [3,7,11,15,19,23,27,31]
for x in tqdm(x_list):
    for fea in ['launch_seq','launch_count_seq']:
        launch_grp[f'{x}_before_{fea}_sum'] = launch_grp[fea].apply(lambda seq: np.sum(seq[-x:]))
        launch_grp[f'{x}_before_{fea}_mean'] = launch_grp[fea].apply(lambda seq: np.mean(seq[-x:]))
        launch_grp[f'{x}_before_{fea}_std'] = launch_grp[fea].apply(lambda seq: np.std(seq[-x:]))
# 加差分、均值、标准差

x_feature_list = [col for col in launch_grp.columns if 'before' in col]
data = data.merge(
    launch_grp[["user_id", "end_date", "label","user_count"]+x_feature_list],
    on=["user_id", "end_date", "label"],
    how="left"
)

# finally
data.loc[data.label >= 0].to_csv(output_dir + "train_data_tree.txt", sep="\t", index=False)
data.loc[data.label < 0].to_csv(output_dir + "test_data_tree.txt", sep="\t", index=False)