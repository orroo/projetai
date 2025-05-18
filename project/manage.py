#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import torch.nn as nn

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=2, num_layers=1):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.dropout1 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)  # output: (batch_size, sequence_length, hidden_size)
        out = self.dropout(out)
        # out = self.fc1(out)
        # out = self.dropout1(out)
        out = self.fc(out)
        return out

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
