import os
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm  # EfficientNet and others
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ================== Config ==================
DATASET_DIR = "dataset_faces"
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_EPOCHS = 10
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

# ================= Transforms ==============
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
# ===========================================

# =============== Load Data =================
def get_loader(split):
    path = os.path.join(DATASET_DIR, split)

    # Force correct label mapping: fake=0, real=1
    class_to_idx = {"fake": 0, "real": 1}

    dataset = datasets.ImageFolder(path, transform=transform)
    dataset.class_to_idx = class_to_idx  # override the default mapping

    print(f"[{split}] class_to_idx: {dataset.class_to_idx}")  # debug check

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == 'train'), num_workers=NUM_WORKERS)
    return loader, dataset

train_loader, train_data = get_loader("train")
val_loader, val_data = get_loader("val")
test_loader, test_data = get_loader("test")
# ===========================================

# ============ Model Definition =============
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
model = model.to(DEVICE)
# ===========================================

# ============== Loss and Optimizer =========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# ===========================================

# ============== Training Loop ==============
def train():
    best_val_acc = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0
        all_preds, all_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{NUM_EPOCHS}] Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        print(f"Train Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")

        # ===== Validation =====
        val_acc = evaluate(val_loader, split="Val")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved!")

# ============== Evaluation =================
def evaluate(loader, split="Test"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"[{split}] Evaluating"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"{split} Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"{split} Confusion Matrix:\n{cm}")
    return acc

# ===========================================

if __name__ == "__main__":
    print(f"🖥️ Using device: {DEVICE}")
    train()
    print("🎯 Testing best model...")
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate(test_loader)
