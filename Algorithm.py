# ==========================================================
# XAI-FMRCNN Algorithm for Biomedical Waste Detection
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# -------------------------------
# Step 1: Data Preprocessing
# -------------------------------

def preprocess_image(image_path, size=(224,224)):
    img = cv2.imread(image_path)

    # Resize (Eq. 20)
    img = cv2.resize(img, size)

    # Gaussian Blur (Eq. 22)
    img = cv2.GaussianBlur(img, (5,5), sigmaX=1)

    # Normalization (Eq. 21)
    img = img / 255.0

    img = torch.tensor(img).permute(2,0,1).float().unsqueeze(0)
    return img


# -------------------------------
# Step 2: FMRCNN Model
# -------------------------------

class FMRCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(FMRCNN, self).__init__()

        # Convolution (Eq. 23)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        # RPN layers (Eq. 24)
        self.rpn_cls = nn.Linear(64*56*56, 2)
        self.rpn_reg = nn.Linear(64*56*56, 4)

        # Mask prediction (Eq. 25)
        self.mask_layer = nn.Linear(64*56*56, 1)

        # Output layer (Eq. 28)
        self.fc = nn.Linear(64*56*56, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):

        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x_flat = x.view(x.size(0), -1)

        # RPN
        rpn_score = self.rpn_cls(x_flat)
        rpn_bbox = self.rpn_reg(x_flat)

        # Mask prediction
        mask = torch.sigmoid(self.mask_layer(x_flat))

        # Classification
        feat = F.relu(self.fc(x_flat))
        output = self.out(feat)

        return output, rpn_score, rpn_bbox, mask


# -------------------------------
# Step 3: XAI Integration
# -------------------------------

# Attention Map (Eq. 26)
def attention_map(model, img):
    img.requires_grad = True

    output, _, _, _ = model(img)
    score = output.max()

    score.backward()

    grad = img.grad.data[0].cpu().numpy()
    attn = np.max(np.abs(grad), axis=0)

    attn = (attn - attn.min()) / (attn.max() + 1e-8)
    return attn


# SHAP (Eq. 27)
def shap_explanation(model, img):
    try:
        import shap

        def f(x):
            x = torch.tensor(x).permute(0,3,1,2).float()
            out, _, _, _ = model(x)
            return out.detach().numpy()

        explainer = shap.Explainer(f, img.permute(0,2,3,1).numpy())
        shap_values = explainer(img.permute(0,2,3,1).numpy())

        return shap_values
    except:
        return None


# -------------------------------
# Step 4: Real-Time Inference
# -------------------------------

def predict(model, img):
    output, rpn_score, rpn_bbox, mask = model(img)

    # Prediction (Eq. 28)
    pred_class = torch.argmax(output, dim=1).item()

    return pred_class, rpn_score, rpn_bbox, mask


# -------------------------------
# Step 5: Alert System
# -------------------------------

def generate_alert(pred_class):
    # Example class mapping
    classes = ["Organic", "Non-Organic", "Bio-Waste"]

    label = classes[pred_class]

    # Alert condition (Eq. 29)
    if label == "Bio-Waste":
        print("🚨 ALERT: Biomedical Waste Detected!")
    else:
        print(f"Detected: {label}")


# -------------------------------
# Step 6: Run Full Pipeline
# -------------------------------

if __name__ == "__main__":

    model = FMRCNN(num_classes=3)
    model.eval()

    # Input image
    img = preprocess_image("sample.jpg")

    # Prediction
    pred_class, rpn_score, rpn_bbox, mask = predict(model, img)

    # Alert
    generate_alert(pred_class)

    # XAI
    attn = attention_map(model, img.clone())

    print("Prediction Class:", pred_class)