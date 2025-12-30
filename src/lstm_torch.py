import torch
import torch.nn as nn
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMPredictor:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = LSTMModel().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict(self, window):
        w = np.asarray(window).reshape(-1, 1)   # 🔒 TEK KURAL
        x = torch.tensor(w, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(x).item())
