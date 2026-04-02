# ==========================================================
# XAI Techniques for Biomedical Waste Detection (Section 2.5.1)
# ==========================================================

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Model (Example)
# -------------------------------
# Replace with your trained FMRCNN model
model = torch.hub.load('pytorch/vision:v0.10.0', 'fasterrcnn_resnet50_fpn', pretrained=True)
model.eval()

# -------------------------------
# 2. Load Image
# -------------------------------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_tensor = torch.tensor(img / 255.0).permute(2, 0, 1).float()
    return img, img_tensor.unsqueeze(0)

# -------------------------------
# 3. Attention Map (Eq. 16)
# A(i,j) = ∂L / ∂X(i,j)
# -------------------------------
def attention_map(model, img_tensor):
    img_tensor.requires_grad = True

    output = model(img_tensor)[0]

    # Take highest scoring object
    score = output['scores'][0]

    score.backward()

    grad = img_tensor.grad.data[0].cpu().numpy()
    attn = np.max(np.abs(grad), axis=0)

    attn = (attn - attn.min()) / (attn.max() + 1e-8)

    return attn

# -------------------------------
# 4. Saliency Map (Eq. 17)
# S(i,j) = ∂Output / ∂X(i,j)
# -------------------------------
def saliency_map(model, img_tensor):
    img_tensor.requires_grad = True

    output = model(img_tensor)[0]
    score = output['scores'][0]

    score.backward()

    grad = img_tensor.grad.data[0].cpu().numpy()
    saliency = np.max(np.abs(grad), axis=0)

    saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)

    return saliency

# -------------------------------
# 5. Activation Map (Eq. 19)
# A_xy = ReLU(Σ X * W + b)
# -------------------------------
def get_activation_map(model, img_tensor):
    activation = []

    def hook_fn(module, input, output):
        activation.append(output.detach())

    # Hook a convolutional layer
    layer = list(model.backbone.body.children())[0]
    hook = layer.register_forward_hook(hook_fn)

    model(img_tensor)

    hook.remove()

    act_map = activation[0][0].cpu().numpy()
    act_map = np.mean(act_map, axis=0)

    act_map = (act_map - act_map.min()) / (act_map.max() + 1e-8)

    return act_map

# -------------------------------
# 6. SHAP Feature Attribution (Eq. 18)
# -------------------------------
def shap_explanation(model, img_tensor):
    try:
        import shap

        # Simplified wrapper
        def f(x):
            x = torch.tensor(x).permute(0,3,1,2).float()
            out = model(x)[0]
            return out['scores'].detach().cpu().numpy()

        explainer = shap.Explainer(f, img_tensor.permute(0,2,3,1).numpy())
        shap_values = explainer(img_tensor.permute(0,2,3,1).numpy())

        return shap_values

    except ImportError:
        print("Install SHAP: pip install shap")
        return None

# -------------------------------
# 7. Visualization
# -------------------------------
def visualize(original, maps, titles):
    plt.figure(figsize=(12,4))
    for i in range(len(maps)):
        plt.subplot(1, len(maps), i+1)
        plt.imshow(original)
        plt.imshow(maps[i], cmap='jet', alpha=0.5)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# -------------------------------
# 8. Run Example
# -------------------------------
if __name__ == "__main__":
    image_path = "sample.jpg"  # replace with your image

    original, img_tensor = load_image(image_path)

    attn = attention_map(model, img_tensor.clone())
    sal = saliency_map(model, img_tensor.clone())
    act = get_activation_map(model, img_tensor.clone())

    visualize(original, [attn, sal, act],
              ["Attention Map", "Saliency Map", "Activation Map"])

    # SHAP (optional)
    shap_values = shap_explanation(model, img_tensor.clone())