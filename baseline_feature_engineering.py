import pandas as pd
import numpy as np
from itertools import groupby

input_dir = "wsdm_train_data/"
output_dir = "wsdm_model_data/"

launch = pd.read_csv(input_dir + "app_launch_logs.csv")
launch.date.min(), launch.date.max()
# 每个用户计数、每天计数
launch['user_count'] = launch['user_id'].map(launch['user_id'].value_counts())
launch['date_count'] = launch['date'].map(launch['date'].value_counts())

# 对用户聚合得到日期序列和launch序列
launch_grp = launch.groupby("user_id").agg(
    launch_date=("date", list),
    launch_type=("launch_type", list),
    user_count=("user_count", np.max),
    launch_date_count=("date_count", list)
).reset_index()
launch_grp['date_len'] = launch_grp.launch_date.apply(lambda x: max(x)-min(x))
# 定义最后一天
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

train = launch_grp[["user_id", "end_date", "label"]]
test = pd.read_csv("test-a.csv")
test["label"] = -1
data = pd.concat([train, test], ignore_index=True)

# append test data to launch_grp
launch_grp = launch_grp.append(
    test.merge(launch_grp[["user_id", "launch_type", "launch_date","user_count","launch_date_count","date_len"]], how="left", on="user_id")
)
# 以最近32天作为窗口生产序列
# get latest 32 days([end_date-31, end_date]) launch type sequence
# 0 for not launch, 1 for launch_type=0, and 2 for launch_type=1
def gen_launch_seq(row):
    seq_sort = sorted(zip(row.launch_type, row.launch_date), key=lambda x: x[1])
    seq_map = {k: max(g)[0] + 1 for k, g in groupby(seq_sort, lambda x: x[1])}
    end = row.end_date
    seq = [seq_map.get(x, 0) for x in range(end-31, end+1)]
    return seq
launch_grp["launch_seq"] = launch_grp.apply(gen_launch_seq, axis=1)

#日活用户数量序列
def gen_launch_count_seq(row):
    seq_map = dict(zip(row.launch_date, row.launch_date_count))
    end = row.end_date
    seq = [seq_map.get(x, 0) for x in range(end-31, end+1)]
    return seq
launch_grp["launch_count_seq"] = launch_grp.apply(gen_launch_count_seq, axis=1)



data = data.merge(
    launch_grp[["user_id", "end_date", "label", "user_count", "launch_seq", "launch_count_seq"]],
    on=["user_id", "end_date", "label"],
    how="left"
)
# (615001, 6)


# choose playback data in [end_date-31, end_date]
playback = pd.read_csv(input_dir + "user_playback_data.csv", dtype={"item_id": str})
playback = playback.merge(data, how="inner", on="user_id")
playback = playback.loc[(playback.date >= playback.end_date-31) & (playback.date <= playback.end_date)]

# add video info to playback data
video_data = pd.read_csv(input_dir + "video_related_data.csv", dtype=str)
playback = playback.merge(video_data[video_data.item_id.notna()], how="left", on="item_id")
# (22528183, 13)


# # using target encoding
# # Tutorial: https://www.kaggle.com/ryanholbrook/target-encoding
# def target_encoding(name, df, m=1):
#     df[name] = df[name].str.split(";")
#     df = df.explode(name)
#     overall = df["label"].mean()
#     df = df.groupby(name).agg(
#         freq=("label", "count"),
#         in_category=("label", np.mean)
#     ).reset_index()
#     df["weight"] = df["freq"] / (df["freq"] + m)
#     df["score"] = df["weight"] * df["in_category"] + (1 - df["weight"]) * overall
#     return df
#
# # father_id target encoding
# df = playback.loc[(playback.label >= 0) & (playback.father_id.notna()), ["father_id", "label"]]
# father_id_score = target_encoding("father_id", df)
# # tag_id target encoding
# df = playback.loc[(playback.label >= 0) & (playback.tag_list.notna()), ["tag_list", "label"]]
# tag_id_score = target_encoding("tag_list", df)
# tag_id_score.rename({"tag_list": "tag_id"}, axis=1, inplace=True)
# # cast_id target encoding
# df = playback.loc[(playback.label >= 0) & (playback.cast.notna()), ["cast", "label"]]
# cast_id_score = target_encoding("cast", df)
# cast_id_score.rename({"cast": "cast_id"}, axis=1, inplace=True)

playback_grp = playback.groupby(["user_id", "end_date", "label"]).agg(
    playtime_list=("playtime", list),
    date_list=("date", list),
    duration_list=("duration", lambda x: ";".join(map(str, x))),
    # father_id_list=("father_id", lambda x: ";".join(map(str, x))),
    # tag_list=("tag_list", lambda x: ";".join(map(str, x))),
    # cast_list=("cast", lambda x: ";".join(map(str, x)))
).reset_index()

# generate latest 32 days([end_date-31, end_date]) playtime sequence
# playtime_norm = 1/(1 + exp(3 - playtime/450)). when playtime=3600s it's preference score is almost equal to 1
def get_playtime_seq(row):
    seq_sort = sorted(zip(row.playtime_list, row.date_list), key=lambda x: x[1])
    seq_map = {k: sum(x[0] for x in g) for k, g in groupby(seq_sort, key=lambda x: x[1])}
    seq_norm = {k: 1/(1+np.exp(3-v/450)) for k, v in seq_map.items()}
    seq = [round(seq_norm.get(i, 0), 4) for i in range(row.end_date-31, row.end_date+1)]
    return seq
playback_grp["playtime_seq"] = playback_grp.apply(get_playtime_seq, axis=1)
# (424312, 7)

# 每天看了几个视频
def get_playcount_seq(row):
    seq = [row.date_list.count(i) for i in range(row.end_date-31, row.end_date+1)]
    return seq
playback_grp["playcount_seq"] = playback_grp.apply(get_playtime_seq, axis=1)

# 完播率



drn_desc = video_data.loc[video_data.duration.notna(), "duration"].astype(int)
# duration preference is a 16-dimentional prefer vector
# for a user, count the frequency of each duration
# prefer_score = freq / max(freq)
# if the user's duration_list is all null, then return null
# null duration_prefer will later be filled with zero vector
def get_duration_prefer(duration_list):
    drn_list = sorted(duration_list.split(";"))
    drn_map = {k: sum(1 for _ in g) for k, g in groupby(drn_list) if k != "nan"}
    if drn_map:
        max_ = max(drn_map.values())
        res = [round(drn_map.get(str(i), 0)/max_, 4) for i in range(1, 17)]
        return res
    else:
        return np.nan
playback_grp["duration_prefer"] = playback_grp.duration_list.apply(get_duration_prefer)

# # add all target encoding scores into a dict
# id_score = dict()
# id_score.update({x[1]: x[5] for x in father_id_score.itertuples()})
# id_score.update({x[1]: x[5] for x in tag_id_score.itertuples()})
# id_score.update({x[1]: x[5] for x in cast_id_score.itertuples()})
#
# # check if features ids are duplicated
# father_id_score.shape[0]+tag_id_score.shape[0]+cast_id_score.shape[0] == len(id_score)

# for these 3 features: father_id_score, cast_score, tag_score,
# choose top 3 preferences
# e.g top_3_id = [(id, freq), (id, freq), (id, freq)]
# score = weight_avg(top_3_id), which values are id_score, weights are frequency
# if the id_list is all null, then return null
# def get_id_score(id_list):
#     x = sorted(id_list.split(";"))
#     x_count = {k: sum(1 for _ in g) for k, g in groupby(x) if k != "nan"}
#     if x_count:
#         x_sort = sorted(x_count.items(), key=lambda k: -k[1])
#         top_x = x_sort[:3]
#         res = [(n, id_score.get(k, 0)) for k, n in top_x]
#         res = sum(n*v for n, v in res) / sum(n for n, v in res)
#         return res
#     else:
#         return np.nan
#
# playback_grp["father_id_score"] = playback_grp.father_id_list.apply(get_id_score)
# playback_grp["cast_id_score"] = playback_grp.cast_list.apply(get_id_score)
# playback_grp["tag_score"] = playback_grp.tag_list.apply(get_id_score)

data = data.merge(
    playback_grp[["user_id", "end_date", "label", "playtime_seq", "playcount_seq", "duration_prefer"]],
    on=["user_id", "end_date", "label"],
    how="left"
)
# data = data.merge(
#     playback_grp[["user_id", "end_date", "label", "playtime_seq", "duration_prefer"]],
#     on=["user_id", "end_date", "label"],
#     how="left"
# )

portrait = pd.read_csv(input_dir + "user_portrait_data.csv", dtype={"territory_code": str})
portrait = pd.merge(data[["user_id", "label"]], portrait, how="left", on="user_id")
# for territory_code, using target encoding again
# df = portrait.loc[(portrait.label >= 0) & (portrait.territory_code.notna()), ["territory_code", "label"]]
# territory_score = target_encoding("territory_code", df)
# add territory_code score into id_score
# n1 = len(id_score)
# id_score.update({x[1]: x[5] for x in territory_score.itertuples()})
# n1 + territory_score.shape[0] == len(id_score)
# get territory score, retain null value
# portrait["territory_score"] = portrait.territory_code.apply(lambda x: id_score.get(x, 0) if isinstance(x, str) else np.nan)

# for multi values of device_ram and device_rom, choose the first one
portrait["device_ram"] = portrait.device_ram.apply(lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)
portrait["device_rom"] = portrait.device_rom.apply(lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)
# add portrait features into data
data = data.merge(portrait.drop("territory_code", axis=1), how="left", on=["user_id", "label"])

# only use interact_type preference
# use all interaction data to calculate interact_type preference
interact = pd.read_csv(input_dir + "user_interaction_data.csv")
interact.interact_type.min(), interact.interact_type.max()

interact_grp = interact.groupby("user_id").agg(
    interact_type=("interact_type", list)
).reset_index()

# similar to duration preference, the interact_type preference could be a 11-dimentional vector
def get_interact_prefer(interact_type):
    x = sorted(interact_type)
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x)}
    x_max = max(x_count.values())
    res = [round(x_count.get(i, 0)/x_max, 4) for i in range(1, 12)]
    return res
interact_grp["interact_prefer"] = interact_grp.interact_type.apply(get_interact_prefer)

data = data.merge(interact_grp[["user_id", "interact_prefer"]], on="user_id", how="left")


# the following features should be normalized
# method: x = (x - mean(x)) / std(x)
norm_cols = ["device_ram", "device_rom","age"]
# norm_cols = ["father_id_score", "cast_id_score", "tag_score",
#             "device_type", "device_ram", "device_rom", "sex",
#             "age", "education", "occupation_status", "territory_score"]
for col in norm_cols:
    mean = data[col].mean()
    std = data[col].std()
    data[col] = (data[col] - mean) / std

# filling null vector features with zero-vectors
data.fillna({
    "playtime_seq": str([0]*32),
    "playcount_seq": str([0]*32),
    "duration_prefer": str([0]*16),
    "interact_prefer": str([0]*11)
}, inplace=True)

# filling null numeric features with 0
data.fillna(0, inplace=True)
# finally
data.loc[data.label >= 0].to_csv(output_dir + "train_data.txt", sep="\t", index=False)
data.loc[data.label < 0].to_csv(output_dir + "test_data.txt", sep="\t", index=False)

