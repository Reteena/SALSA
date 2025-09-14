import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__ (self, input_size, hidden, output, dropout=0.3):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output)
        )

    def forward(self, x):
        return self.classifier(x)