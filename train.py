import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd, os
import matplotlib.pyplot as plt

# ===================== DATASET =====================
class FairFaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['file'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        gender = torch.tensor(1.0 if row['gender'] == 'Male' else 0.0)

        age_str = str(row['age'])
        if '-' in age_str:
            low, high = age_str.split('-')
            age_val = (float(low) + float(high)) / 2
        elif age_str == 'more than 70': # Handle 'more than 70' explicitly
            age_val = 70.0  # Assign a numerical value, e.g., 70
        else:
            age_val = float(age_str)
        age = torch.tensor(age_val)

        return img, gender, age

# ===================== MODEL =====================
class GenderAgeModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ Fixed deprecation warning
        vgg = models.vgg16(weights=None)
        vgg.load_state_dict(torch.load(
            '/content/vgg16-397923af.pth',
            map_location='cpu',
            weights_only=False
        ))
        print("✅ VGG16 weights loaded!")

        for param in vgg.features.parameters():
            param.requires_grad = False

        self.features = vgg.features
        self.avgpool  = nn.AdaptiveAvgPool2d((7, 7))

        self.shared = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5)
        )

        self.gender_head = nn.Linear(1024, 1)
        self.age_head    = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.shared(x)
        gender = torch.sigmoid(self.gender_head(x))
        age    = self.age_head(x)
        return gender, age

# ===================== CHECK FILES FIRST =====================
print("📁 Checking files...")
print("CSV exists:", os.path.exists('/content/FairFace/fairface_label_train.csv'))
print("VGG exists:", os.path.exists('/content/vgg16-397923af.pth'))
print("Train folder exists:", os.path.exists('/content/FairFace/train'))

# ===================== TRAINING SETUP =====================
CSV_FILE = '/content/FairFace/fairface_label_train.csv'
IMG_DIR  = '/content/FairFace'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset      = FairFaceDataset(CSV_FILE, IMG_DIR, transform)
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0   # ✅ Fixed: 0 works best in Colab
)

print(f"✅ Dataset loaded! Total images: {len(dataset)}")

model  = GenderAgeModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

gender_loss_fn = torch.nn.BCELoss()
age_loss_fn    = torch.nn.L1Loss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)

# ===================== TRAINING LOOP =====================
total_losses  = []
gender_losses = []
age_losses    = []

for epoch in range(10):
    epoch_loss  = 0
    epoch_gloss = 0
    epoch_aloss = 0

    for batch_idx, (imgs, genders, ages) in enumerate(train_loader):
        imgs, genders, ages = imgs.to(device), genders.to(device), ages.to(device)
        optimizer.zero_grad()
        pred_gender, pred_age = model(imgs)

        g_loss = gender_loss_fn(pred_gender.squeeze(), genders)
        a_loss = age_loss_fn(pred_age.squeeze(), ages)
        loss   = g_loss + a_loss

        loss.backward()
        optimizer.step()

        epoch_loss  += loss.item()
        epoch_gloss += g_loss.item()
        epoch_aloss += a_loss.item()

        # ✅ Show progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    total_losses.append(epoch_loss)
    gender_losses.append(epoch_gloss)
    age_losses.append(epoch_aloss)
    print(f"✅ Epoch {epoch+1}/10 | Total: {epoch_loss:.4f} | Gender: {epoch_gloss:.4f} | Age: {epoch_aloss:.4f}")

# ===================== SAVE MODEL =====================
# ✅ Save locally first (fast)
torch.save(model.state_dict(), '/content/model.pth')
print("✅ model.pth saved locally!")

# ✅ Save graph locally
plt.figure(figsize=(10,5))
plt.plot(total_losses,  label='Total Loss')
plt.plot(gender_losses, label='Gender Loss')
plt.plot(age_losses,    label='Age Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('/content/training_graph.png')
plt.show()
print("✅ training_graph.png saved!")

# ✅ Download both files to your PC immediately
from google.colab import files
files.download('/content/model.pth')
files.download('/content/training_graph.png')
print("✅ Files downloaded to your PC!")