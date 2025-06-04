import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.onnx

# === 設定 ===
DATA_DIR = 'training_data'
BATCH_SIZE = 16
EPOCHS = 30
IMAGE_SIZE = (20, 32)  # 高さ, 幅 ← あなたの画像サイズに合わせる
ONNX_OUTPUT = 'model.onnx'

# === データ前処理 ===
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomRotation(5),       # 傾きのばらつき
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 平行移動
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 濃さのばらつき
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3), 
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === シンプルなCNNモデル ===
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 入力チャンネル1（グレースケール）
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_SIZE[0] // 4) * (IMAGE_SIZE[1] // 4), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN(num_classes)

# === 学習準備 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 学習ループ ===
print("Training...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}: Loss = {running_loss:.4f}")

# === ONNXエクスポート ===
print(f"Exporting to {ONNX_OUTPUT} ...")
dummy_input = torch.randn(1, 1, *IMAGE_SIZE).to(device)
torch.onnx.export(model, dummy_input, ONNX_OUTPUT,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})

print("Done! Model exported as:", ONNX_OUTPUT)
print("Classes:", class_names)

# === 保存用にクラスラベルも出力 ===
with open("labels.txt", "w") as f:
    for cls in class_names:
        f.write(cls + "\n")