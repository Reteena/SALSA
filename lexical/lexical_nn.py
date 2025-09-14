import torch.nn as nn

class LexicalHead(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, out_dim)
        )
    def forward(self, x):
        return self.net(x)