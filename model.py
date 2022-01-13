import torch
import torch.nn as nn


class AQYModel(nn.Module):
    def __init__(self):
        super(AQYModel, self).__init__()

        # self.user_id_embedding = nn.Embedding(600000 + 1, 8)
        self.launch_type_embedding = nn.Embedding(2 + 1, 8)
        self.device_type_embedding = nn.Embedding(5, 8)
        self.sex_embedding = nn.Embedding(3, 8)
        self.education_embedding = nn.Embedding(3, 8)
        self.occupation_status_embedding = nn.Embedding(2, 8)

        self.seq_input = nn.Linear(11,32)
        self.encoder = nn.TransformerEncoderLayer(d_model=32,nhead=2,dim_feedforward=128)
        self.encoder_out = nn.Linear(32*32,64)

        # self.launch_seq_gru = nn.GRU(input_size=11,
        #                              hidden_size=32,
        #                              batch_first=True)
        # self.playtime_seq_gru = nn.GRU(input_size=1,
        #                              hidden_size=16,
        #                              batch_first=True)


        self.fc1 = nn.Linear(126, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, launch_seq, playtime_seq, duration_prefer,
                interact_prefer, feat, launch_count_seq,
                playcount_seq, device_type, sex,
                education, occupation_status):


        launch_seq = self.launch_type_embedding(launch_seq)
        playtime_seq = playtime_seq.reshape((-1, 32, 1))
        launch_count_seq = launch_count_seq.reshape((-1, 32, 1))
        playcount_seq = playcount_seq.reshape((-1, 32, 1))

        device_type = self.device_type_embedding(device_type)
        sex = self.sex_embedding(sex)
        education = self.education_embedding(education)
        occupation_status = self.occupation_status_embedding(occupation_status)

        sequence_input = torch.cat([launch_seq,playtime_seq,launch_count_seq,playcount_seq],dim=2)

        sequence_input = self.seq_input(sequence_input)
        sequence_input = sequence_input.transpose(0,1)

        launch_seq = self.encoder(sequence_input)

        launch_seq = launch_seq.transpose(0,1)

        launch_seq = self.encoder_out(torch.flatten(launch_seq,start_dim=1))

        # launch_seq, h_n = self.launch_seq_gru(sequence_input)

        # playtime_seq = playtime_seq.reshape((-1, 32, 1))
        # playtime_seq, _ = self.playtime_seq_gru(playtime_seq)

        # launch_seq_mean = torch.mean(launch_seq, dim=1)
        # launch_seq_max,_ = torch.max(launch_seq, dim=1)

        # playtime_seq_mean = torch.mean(playtime_seq, dim=1)
        # playtime_seq_max, _ = torch.max(playtime_seq, dim=1)

        fc_input = torch.cat([launch_seq, duration_prefer, interact_prefer, feat,
                              device_type, sex, education, occupation_status], 1)

        pred = self.fc2(self.fc1(fc_input))

        return pred