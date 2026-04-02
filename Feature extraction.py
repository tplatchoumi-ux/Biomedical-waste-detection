# ==========================================================
# FMRCNN-Based Feature Extraction for Biomedical Waste
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# 1. Convolutional Feature Extractor
# -------------------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # Spatial features
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x  # Feature map

# -------------------------------
# 2. ConvLSTM Cell (Spatial + Temporal)
# -------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)

        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_output, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

# -------------------------------
# 3. GRU for Temporal Learning
# -------------------------------
class TemporalGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TemporalGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1, :]  # last time step

# -------------------------------
# 4. Simplified RPN (Region Proposal)
# -------------------------------
class SimpleRPN(nn.Module):
    def __init__(self, in_channels):
        super(SimpleRPN, self).__init__()

        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(256, 9 * 2, kernel_size=1)  # objectness
        self.reg_layer = nn.Conv2d(256, 9 * 4, kernel_size=1)  # bbox

    def forward(self, x):
        x = F.relu(self.conv(x))
        cls = self.cls_layer(x)
        reg = self.reg_layer(x)
        return cls, reg

# -------------------------------
# 5. FMRCNN Model
# -------------------------------
class FMRCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(FMRCNN, self).__init__()

        self.cnn = CNNFeatureExtractor()

        self.convlstm = ConvLSTMCell(input_dim=128, hidden_dim=128, kernel_size=3)

        self.rpn = SimpleRPN(128)

        self.gru = TemporalGRU(input_size=128*28*28, hidden_size=256)

        self.fc = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x_seq):
        """
        x_seq shape: (batch, time_steps, channels, H, W)
        """

        batch_size, time_steps, C, H, W = x_seq.size()

        h, c = None, None
        features_seq = []

        for t in range(time_steps):
            x = x_seq[:, t]

            # CNN Feature Extraction
            feat = self.cnn(x)

            # Initialize ConvLSTM states
            if h is None:
                h = torch.zeros_like(feat)
                c = torch.zeros_like(feat)

            # ConvLSTM
            h, c = self.convlstm(feat, h, c)

            # RPN (region proposals)
            cls_map, reg_map = self.rpn(h)

            # Flatten features for GRU
            f_flat = h.view(batch_size, -1)
            features_seq.append(f_flat)

        features_seq = torch.stack(features_seq, dim=1)

        # GRU temporal modeling
        temporal_feat = self.gru(features_seq)

        # Fully connected layers
        x = F.relu(self.fc(temporal_feat))
        out = self.out(x)

        return out, cls_map, reg_map


# -------------------------------
# 6. Example Run
# -------------------------------
if __name__ == "__main__":

    model = FMRCNN(num_classes=3)

    # Example input: batch=2, time_steps=5, 3 channels, 224x224
    dummy_input = torch.randn(2, 5, 3, 224, 224)

    output, cls_map, reg_map = model(dummy_input)

    print("Final Output:", output.shape)
    print("RPN Classification Map:", cls_map.shape)
    print("RPN Regression Map:", reg_map.shape)