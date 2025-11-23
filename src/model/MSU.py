import torch
import torch.nn as nn
from model.TE import TE_1D


class MSU(nn.Module):
    def __init__(self, in_features, window_len, hidden_dim, transformer_msu_bool, temporal_attention_bool=True):
        super(MSU, self).__init__()
        self.in_features = in_features
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.transformer_msu_bool = transformer_msu_bool
        self.temporal_attention_bool = temporal_attention_bool

        if self.transformer_msu_bool:
            self.TE_1D = TE_1D(window_len=13, dim=128, depth=2, heads=4, mlp_dim=32, channels=in_features, dim_head=4, dropout=0.1, emb_dropout=0.1)
            if self.temporal_attention_bool:
                self.attn1 = nn.Linear(hidden_dim, hidden_dim)
                self.attn2 = nn.Linear(hidden_dim, 1)
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, 2)

        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim)
            if self.temporal_attention_bool:
                self.attn1 = nn.Linear(2 * hidden_dim, hidden_dim)
                self.attn2 = nn.Linear(hidden_dim, 1)
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, 2)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X):
        """
        :X: [batch_size(B), window_len(L), in_features(I)]
        :return: Parameters: [batch, 2]
        """
        X = X.permute(1, 0, 2)

        # Backward compatibility: default to True if attribute doesn't exist in old models
        temporal_attention_bool = getattr(self, 'temporal_attention_bool', True)

        if self.transformer_msu_bool:
            outputs = self.TE_1D(X)
            if temporal_attention_bool:
                scores = self.attn2(torch.tanh(self.attn1(outputs)))
                scores = scores.squeeze(2).transpose(1, 0)

        else:
            outputs, (h_n, c_n) = self.lstm(X)  # lstm version
            if temporal_attention_bool:
                H_n = h_n.repeat((self.window_len, 1, 1))
                scores = self.attn2(torch.tanh(self.attn1(torch.cat([outputs, H_n], dim=2))))  # [L, B*N, 1]
                scores = scores.squeeze(2).transpose(1, 0)  # [B*N, L]

        if temporal_attention_bool:
            attn_weights = torch.softmax(scores, dim=1)
            outputs = outputs.permute(1, 0, 2)  # [B*N, L, H]
            attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        else:
            # Without attention, use mean pooling over all timesteps
            outputs = outputs.permute(1, 0, 2)  # [B*N, L, H]
            attn_embed = outputs.mean(dim=1)  # Mean pooling [B*N, H]

        embed = torch.relu(self.bn1(self.linear1(attn_embed)))
        #embed = self.dropout(embed)
        parameters = self.linear2(embed)
        # return parameters[:, 0], parameters[:, 1]   # mu, sigma
        return parameters.squeeze(-1)

if __name__ == '__main__':
    a = torch.randn((16, 20, 3))
    net = MSU(3, 20, 128)
    b = net(a)
    print(b)
