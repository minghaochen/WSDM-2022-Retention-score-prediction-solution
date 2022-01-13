import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def cal_score(pred, label):
    pred = np.array(pred)
    label = np.array(label)

    diff = (pred - label) / 7
    diff = np.abs(diff)

    score = 100 * (1 - np.mean(diff))
    return score

class AQYDataset(Dataset):
    def __init__(self, df):
        self.df = df
        # self.feat_col = list(set(self.df.columns) - set(['user_id', 'end_date', 'label',
        #                                                  'launch_seq', 'playtime_seq',
        #                                                  'duration_prefer', 'interact_prefer']))
        self.feat_col = list(set(self.df.columns) - set(['user_id', 'end_date', 'label', 'user_count',
                                                         'launch_seq', 'launch_count_seq',
                                                         'playtime_seq','playcount_seq',
                                                         'duration_prefer', 'interact_prefer',
                                                         'device_type','sex','education','occupation_status']))
        self.df_feat = self.df[self.feat_col]


    def __getitem__(self, index):
        launch_seq = self.df['launch_seq'].iloc[index]
        launch_count_seq = self.df['launch_count_seq'].iloc[index]
        playtime_seq = self.df['playtime_seq'].iloc[index]
        playcount_seq = self.df['playcount_seq'].iloc[index]
        duration_prefer = self.df['duration_prefer'].iloc[index]
        interact_prefer = self.df['interact_prefer'].iloc[index]
        feat = self.df_feat.iloc[index].values.astype(np.float32)
        device_type = self.df['device_type'].iloc[index]
        sex = self.df['sex'].iloc[index]
        education = self.df['education'].iloc[index]
        occupation_status = self.df['occupation_status'].iloc[index]
        label = self.df['label'].iloc[index]

        # # 统计特征
        # days = []
        # for x in [3,7,15,31]:
        #     days.append(np.sum(launch_seq[-x:]))
        # days.append(days[1] - days[0])
        # days.append(days[2] - days[1])
        # days.append(days[3] - days[2])
        # feat = np.concatenate([feat, np.array(days)]).astype(np.float32)

        launch_seq = torch.tensor(launch_seq).long()
        playtime_seq = torch.tensor(playtime_seq).float()
        duration_prefer = torch.tensor(duration_prefer).float()
        interact_prefer = torch.tensor(interact_prefer).float()
        feat = torch.tensor(feat).float()
        label = torch.tensor(label).float()

        launch_count_seq, playcount_seq, device_type, sex, \
        education, occupation_status = torch.tensor(launch_count_seq).float(), torch.tensor(playcount_seq).float(), \
                                       torch.tensor(device_type).long(), torch.tensor(sex).long(),\
                                       torch.tensor(education).long(),torch.tensor(occupation_status).long()


        return launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label,\
               launch_count_seq, playcount_seq, device_type, sex, education, occupation_status

    def __len__(self):
        return len(self.df)

# class AQYDataset(Dataset):
#     def __init__(self, df, device, fea_col):
#         self.user_id_list = df['user_id'].values
#
#         self.launch_seq_list = df['launch_seq'].values
#
#         self.label_list = df['label'].values
#
#         self.fea = df[fea_col].values
#
#     def __getitem__(self, index):
#         user_id = self.user_id_list[index]
#
#         launch_seq = np.array(self.launch_seq_list[index])
#
#         label = self.label_list[index]
#
#         fea = self.fea[index]
#
#         return user_id, launch_seq, label, fea
#
#     def __len__(self):
#         return len(self.user_id_list)


def fit(model, train_loader, optimizer, criterion, device):
    model.train()

    pred_list = []
    label_list = []

    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label,\
               launch_count_seq, playcount_seq, device_type, sex, education, occupation_status in tqdm(train_loader):

        launch_seq = launch_seq.to(device)
        playtime_seq = playtime_seq.to(device)
        duration_prefer = duration_prefer.to(device)
        interact_prefer = interact_prefer.to(device)
        feat = feat.to(device)
        label = label.to(device)

        launch_count_seq, playcount_seq, device_type, sex, education, \
        occupation_status = launch_count_seq.to(device), \
                            playcount_seq.to(device), device_type.to(device), sex.to(device), \
                            education.to(device), occupation_status.to(device)

        pred = model(launch_seq, playtime_seq, duration_prefer,
                     interact_prefer, feat, launch_count_seq,
                     playcount_seq, device_type, sex,
                     education, occupation_status)

        loss = criterion(pred.squeeze(), label)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score


def validate(model, val_loader, device):
    model.eval()

    pred_list = []
    label_list = []

    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label,\
               launch_count_seq, playcount_seq, device_type, sex, education, occupation_status in tqdm(val_loader):
        launch_seq = launch_seq.to(device)
        playtime_seq = playtime_seq.to(device)
        duration_prefer = duration_prefer.to(device)
        interact_prefer = interact_prefer.to(device)
        feat = feat.to(device)
        label = label.to(device)

        launch_count_seq, playcount_seq, device_type, sex, education, \
        occupation_status = launch_count_seq.to(device), \
                            playcount_seq.to(device), device_type.to(device), sex.to(device), \
                            education.to(device), occupation_status.to(device)

        pred = model(launch_seq, playtime_seq, duration_prefer,
                     interact_prefer, feat, launch_count_seq,
                     playcount_seq, device_type, sex,
                     education, occupation_status)

        # pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat)

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score


def predict(model, test_loader, device):
    model.eval()
    test_pred = []
    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label,\
               launch_count_seq, playcount_seq, device_type, sex, education, occupation_status in tqdm(test_loader):
        launch_seq = launch_seq.to(device)
        playtime_seq = playtime_seq.to(device)
        duration_prefer = duration_prefer.to(device)
        interact_prefer = interact_prefer.to(device)
        feat = feat.to(device)
        label = label.to(device)

        launch_count_seq, playcount_seq, device_type, sex, education, \
        occupation_status = launch_count_seq.to(device), \
                            playcount_seq.to(device), device_type.to(device), sex.to(device), \
                            education.to(device), occupation_status.to(device)

        pred = model(launch_seq, playtime_seq, duration_prefer,
                     interact_prefer, feat, launch_count_seq,
                     playcount_seq, device_type, sex,
                     education, occupation_status)

        # pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat).squeeze()
        test_pred.extend(pred.cpu().detach().numpy())

    return test_pred