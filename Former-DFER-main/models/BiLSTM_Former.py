import torch
import torch.nn as nn

class TBiLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_size=640, num_layers=1, dropout=0.0):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.proj = nn.Linear(hidden_size * 2, 512)   # ← 512 out, NOT 7 # output 512 to match the fc head

    def forward(self, x):
        x = x.contiguous().view(-1, 16, 512)
        self.bilstm.flatten_parameters()
        lstm_out, _ = self.bilstm(x)
        feat = lstm_out.mean(dim=1)        # mean-pooling (your idea)
        feat = self.norm(feat)             # LayerNorm (your idea)
        feat = self.proj(feat)             # → 512 (my idea, keeps pipeline intact)
        return feat

def temporal_transformer():
    return TBiLSTM()


if __name__ == '__main__':
    x = torch.randn((8, 16, 512))
    model = temporal_transformer()
    print(model(x).shape)   # must print torch.Size([8, 512])


# import torch
# import torch.nn as nn


# class TBiLSTM(nn.Module):
#     def __init__(self, input_dim=512, hidden_size=256, num_layers=3, dropout=0.2):
#         super().__init__()
#         self.bilstm = nn.LSTM(input_size=input_dim,
#                               hidden_size=hidden_size,
#                               num_layers=num_layers,
#                               batch_first=True,
#                               bidirectional=True,
#                               dropout=dropout if num_layers > 1 else 0.0)
#         self.norm = nn.LayerNorm(hidden_size * 2)      # 256*2 = 512
#         self.proj = nn.Linear(hidden_size * 2, 512)    # 512 -> 512

#     def forward(self, x):
#         x = x.contiguous().view(-1, 16, 512)
#         self.bilstm.flatten_parameters()
#         lstm_out, _ = self.bilstm(x)                   # (b, 16, 512)
#         feat = lstm_out.mean(dim=1)                    # (b, 512)
#         feat = self.norm(feat)
#         feat = self.proj(feat)                         # (b, 512)
#         return feat


# def temporal_transformer():
#     return TBiLSTM()


# if __name__ == '__main__':
#     x = torch.randn((8, 16, 512))
#     model = temporal_transformer()
#     print(model(x).shape)   # must print torch.Size([8, 512])    