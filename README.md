
# Mowito Assignment

## Prerequisites

Before you begin, ensure the following:

- Python installed (version >= 3.7).
- PyTorch and torchvision installed.
- Required libraries: Pillow, numpy, matplotlib.

Install missing libraries using:
```bash
pip install torch torchvision pillow numpy matplotlib
```

### Trained Models
- `scratch_detector_4.pth` (for anomaly detection classification).
- `segmentation_model_6.pth` (for segmentation).

---

## 1. Anomaly Detection (Classification)

### Description
The anomaly detection model classifies images into two categories:
- **Good (0):** No scratch detected.
- **Bad (1):** Scratch detected.

### Steps

#### Place the Trained Model
Ensure `scratch_detector.pth` is in the working directory.

#### Run the Prediction Script
Use the following Python code to evaluate new images:
```python
import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: Good and Bad
model.load_state_dict(torch.load('scratch_detector_4.pth'))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_anomaly(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    class_name = "Good" if predicted.item() == 0 else "Bad"
    print(f"Prediction for {image_path}: {class_name}")

predict_anomaly('path_to_test_image')
```

### Output
The script will print whether the image is **Good** or **Bad** based on the prediction.

---

## 2. Segmentation

### Description
The segmentation model predicts a binary mask highlighting areas with scratches in the input image.

### Steps

#### Place the Trained Model
Ensure `segmentation_model.pth` is in the working directory.

#### Run the Prediction Script
Use the following Python code to evaluate and visualize the segmentation results:
```python
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = models.resnet34(pretrained=False)
        self.encoder_layers = list(self.encoder.children())[:-2]
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load trained model
model = UNet().to(DEVICE)
model.load_state_dict(torch.load('segmentation_model_5.pth'))
model.eval()

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0) 
    image = image.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        output = output.squeeze(0) 
        output = output.cpu().numpy() 
        output = (output > 0.5).astype(np.uint8) 

    
    output_image = Image.fromarray(output[0] * 255)  
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(0).cpu().permute(1, 2, 0)) 
    plt.title("Input Image")
    plt.axis("off")

    
    plt.subplot(1, 2, 2)
    plt.imshow(output_image, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()


input_image_path = "path of test image" 
predict(model, input_image_path, image_transform, DEVICE)
```

### Output
The script will display:
1. The input image.
2. The predicted segmentation mask highlighting scratches.
