import copy
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from model import AQYModel
from model_tools import AQYDataset, fit, predict, validate
import json

warnings.filterwarnings('ignore')


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random_seed(2022)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_dir = "./wsdm_model_data/"
# train data
data = pd.read_csv(data_dir + "train_data.txt", sep="\t")
data["launch_seq"] = data.launch_seq.apply(lambda x: json.loads(x))
data["launch_count_seq"] = data.launch_count_seq.apply(lambda x: json.loads(x))
data["playtime_seq"] = data.playtime_seq.apply(lambda x: json.loads(x))
data["playcount_seq"] = data.playcount_seq.apply(lambda x: json.loads(x))
data["duration_prefer"] = data.duration_prefer.apply(lambda x: json.loads(x))
data["interact_prefer"] = data.interact_prefer.apply(lambda x: json.loads(x))
# shuffle data
data = data.sample(frac=1).reset_index(drop=True)

train_dataset = AQYDataset(data.iloc[:-6000])
# train_dataset = AQYDataset(data)
val_dataset = AQYDataset(data.iloc[-6000:])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

test_data = pd.read_csv(data_dir + "test_data.txt", sep="\t")
test_data["launch_seq"] = test_data.launch_seq.apply(lambda x: json.loads(x))
test_data["launch_count_seq"] = test_data.launch_count_seq.apply(lambda x: json.loads(x))
test_data["playtime_seq"] = test_data.playtime_seq.apply(lambda x: json.loads(x))
test_data["playcount_seq"] = test_data.playcount_seq.apply(lambda x: json.loads(x))
test_data["duration_prefer"] = test_data.duration_prefer.apply(lambda x: json.loads(x))
test_data["interact_prefer"] = test_data.interact_prefer.apply(lambda x: json.loads(x))
test_data['label'] = 0

test_dataset = AQYDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
test_data.shape


model = AQYModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_score = float('-inf')
last_improve = 0
best_model = None

for epoch in range(50):
    train_score = fit(model, train_loader, optimizer, criterion, device)
    val_score = validate(model, val_loader, device)

    if val_score > best_val_score:
        best_val_score = val_score
        best_model = copy.deepcopy(model)
        last_improve = epoch
        improve = '*'
    else:
        improve = ''

    if epoch - last_improve > 10:
        break

    print(
        f'Epoch: {epoch} Train Score: {train_score}, Valid Score: {val_score} '
    )

model = best_model

# valid['pred'] = predict(model, val_loader, device)
# valid['diff'] = valid['label'] - valid['pred']
# valid['diff'] = abs(valid['diff']) / 7
# score = 100 * (1 - valid['diff'].mean())
# print(f'Valid Score: {score}')

os.makedirs('sub', exist_ok=True)

# test_data['pred'] = predict(model, test_loader, device)
# test = test_data[['user_id', 'pred']]
# test['user_id'] = test['user_id'] - 1
# test['user_id'] = user_lbe.inverse_transform(test['user_id'])
#
# test.to_csv(f'sub/{score}.csv', index=False, header=False, float_format="%.2f")


test_pred = predict(model, test_loader,device)
test_pred = np.vstack(test_pred)

test_data["prediction"] = test_pred[:, 0]
test_data = test_data[["user_id", "prediction"]]
# can clip outputs to [0, 7] or use other tricks
test_data.to_csv("sub/baseline_submission.csv", index=False, header=False, float_format="%.2f")