import random

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import sys
import io
import numpy as np
console_buffer = io.StringIO()
sys.stdout = console_buffer  # 将所有print输出重定向到内存中
def set_seed(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 已固定随机种子: {seed}")

set_seed(42)


import torch
import torch.nn as nn
from torchvision.models import resnet18

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNet18_SE(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(ResNet18_SE, self).__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # 在 layer2、3、4 后加入 SEBlock
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x); x = self.se2(x)
        x = self.backbone.layer3(x); x = self.se3(x)
        x = self.backbone.layer4(x); x = self.se4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------- 自定义 Dataset ----------------------
class TN5000ClassificationDataset(Dataset):
    def __init__(self, txt_path, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform

        with open(txt_path, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        ann_path = os.path.join(self.ann_dir, img_id + ".xml")

        image = Image.open(img_path).convert("RGB")

        tree = ET.parse(ann_path)
        root = tree.getroot()
        obj = root.find("object")
        label = int(obj.find("name").text.strip())  # 类别编号

        if self.transform:
            image = self.transform(image)

        return image, label


# ======================== 配置区域 =========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
num_classes = 2
num_epochs = 20
k_folds = 5
batch_size = 16
learning_rate = 1e-4
# patience = 2  # 早停耐心值
mod='se_attention'
txt_all = r"C:\Users\a1551\Downloads\TN5000_forReview\TN5000_forReview\ImageSets\Main\trainval.txt"
img_dir = r"C:\Users\a1551\Downloads\TN5000_forReview\TN5000_forReview\JPEGImages"
ann_dir = r"C:\Users\a1551\Downloads\TN5000_forReview\TN5000_forReview\Annotations"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
# ## ======================== Dataset & Transform =========================
train_transform = transforms.Compose([
    transforms.Resize((919, 646)),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
#
dataset_all = TN5000ClassificationDataset(txt_all, img_dir, ann_dir, train_transform)

# ======================== K 折划分 =========================
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_all)):
    print(f"\n================ Fold {fold+1}/{k_folds} ================")

    train_sub = Subset(dataset_all, train_idx)
    val_sub = Subset(dataset_all, val_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

    model = ResNet18_SE(num_classes=num_classes, pretrained=True, dropout=0.5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器：当验证集loss在若干epoch内未降低时自动减半学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_val_acc = 0
    best_state = None
    best_state_loss=[]
    best_state_acc=[]
    vcc=0
    loss=1
    for epoch in range(num_epochs):
        # ---------- Train ----------
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"[Fold {fold+1}] Epoch {epoch+1}/{num_epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"[Fold {fold+1}] Epoch {epoch+1}/{num_epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_loader)
        acc = correct / total
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Fold {fold+1}] Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {acc:.4f} | LR: {current_lr:.6f}")

        # ---------- 学习率调整 ----------
        scheduler.step(val_loss)

        # ---------- 保存最优模型 ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_loss = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': acc,
                'learning_rate': current_lr
            }
            torch.save(best_state_loss, f"ReduceLROnPlateau_result/{mod}resnet18_fold{fold + 1}_bestLoss.pth")
            print(f"✅ 最优Loss模型更新：Val Loss={val_loss:.4f}, Acc={acc:.4f}, LR={current_lr:.6f}")

        # （新增部分）单独保存最优Acc模型
        if acc > best_val_acc:
            best_val_acc = acc
            best_state_acc = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': acc,
                'learning_rate': current_lr
            }
            torch.save(best_state_acc, f"ReduceLROnPlateau_result/{mod}resnet18_fold{fold + 1}_bestAcc.pth")
            print(f"🏆 最优Acc模型更新：Val Acc={acc:.4f}, Loss={val_loss:.4f}, LR={current_lr:.6f}")

    # ---------- Fold 结果 ----------
    if best_state_loss is None and best_state_acc is None:
        print(f"⚠️ Fold {fold + 1} 未检测到任何最优模型，使用最后一轮结果。")
        best_state_loss = best_state_acc = {
            'val_acc': acc,
            'val_loss': val_loss
        }

    # 输出最优Loss模型和最优Acc模型的结果
    print(f"\n📊 Fold {fold + 1} 结果：")
    print(f"  ▶ 最优Loss模型：Val Loss={best_state_loss['val_loss']:.4f} | Val Acc={best_state_loss['val_acc']:.4f}")
    print(f"  ▶ 最优Acc模型：Val Loss={best_state_acc['val_loss']:.4f} | Val Acc={best_state_acc['val_acc']:.4f}")

    # 同时记录两种指标
    fold_results.append({
        'fold': fold + 1,
        'best_loss': best_state_loss['val_loss'],
        'best_acc_for_loss': best_state_loss['val_acc'],
        'best_acc': best_state_acc['val_acc'],
        'best_loss_for_acc': best_state_acc['val_loss']
    })

    # ---------- 绘制每折曲线 ----------
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold+1} Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ======================== 汇总结果 =========================
print("\n===== K 折验证结果 =====")
for result in fold_results:
    print(f"Fold {result['fold']}: "
          f"BestLoss={result['best_loss']:.4f} (Acc={result['best_acc_for_loss']:.4f}) | "
          f"BestAcc={result['best_acc']:.4f} (Loss={result['best_loss_for_acc']:.4f})")

# 平均值
avg_acc_for_loss = sum(r['best_acc_for_loss'] for r in fold_results) / len(fold_results)
avg_acc = sum(r['best_acc'] for r in fold_results) / len(fold_results)
avg_loss = sum(r['best_loss'] for r in fold_results) / len(fold_results)
print(f"\n平均准确率(基于Loss最优)：{avg_acc_for_loss:.4f}")
print(f"平均准确率(基于Acc最优)： {avg_acc:.4f}")
print(f"平均Loss(基于Loss最优)：{avg_loss:.4f}")




# # ======================== 测试集推理与评估 =========================
# ======================== 测试集推理与评估 =========================
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
import numpy as np
import seaborn as sns
#
# ---------- 测试集路径 ----------
txt_test = r"C:\Users\a1551\Downloads\TN5000_forReview\TN5000_forReview\ImageSets\Main\test.txt"

# 使用与训练相同的图像预处理，但去掉随机翻转
test_transform = transforms.Compose([
    transforms.Resize((919, 646)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 只创建一个完整测试集
test_dataset = TN5000ClassificationDataset(txt_test, img_dir, ann_dir, test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_labels = []
all_probs = []  # 每个 fold 的概率预测
num_folds = k_folds  # 例如5折

# 获取真实标签
for _, labels in test_loader:
    all_labels.extend(labels.numpy())
y_true = np.array(all_labels)

# 五折模型依次推理
fold_probs = []
for fold in range(num_folds):
    model_path = rf"D:\program\py\mpaper_decetion_10_25\ReduceLROnPlateau_result\{mod}resnet18_fold{fold+1}_bestAcc.pth"
    if not os.path.exists(model_path):
        print(f"⚠️ 未找到 {model_path}，跳过此 fold")
        continue

    model = ResNet18_SE(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    probs_list = []

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc=f"[Fold {fold+1}] Testing"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 取 malignant 概率
            probs_list.extend(probs)

    fold_probs.append(np.array(probs_list))

# ------------------ 集成预测 ------------------
# 每个样本五折模型概率取平均
y_prob_final = np.mean(fold_probs, axis=0)
# 阈值0.5转为最终预测
y_pred_final = (y_prob_final >= 0.5).astype(int)

# ------------------ 计算混淆矩阵和指标 ------------------
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

cm = confusion_matrix(y_true, y_pred_final)
TP = cm[1, 1]
FP = cm[0, 1]
TN = cm[0, 0]
FN = cm[1, 0]

accuracy = accuracy_score(y_true, y_pred_final)
precision = precision_score(y_true, y_pred_final)
recall = recall_score(y_true, y_pred_final)
f1 = f1_score(y_true, y_pred_final)
roc_auc = roc_auc_score(y_true, y_prob_final)


print(f"Confusion Matrix:\n{cm}")
print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={roc_auc:.3f}")
# ------------------ 绘制混淆矩阵 ------------------
labels_names = ['Benign (0)', 'Malignant (1)']
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_names, yticklabels=labels_names,
            cbar=False, linewidths=0.5, square=True)
plt.title('Confusion Matrix (Ensemble of 5 Folds)', fontsize=14, pad=15)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

# ------------------ 绘制 ROC/AUC 曲线 ------------------
fpr, tpr, thresholds = roc_curve(y_true, y_prob_final)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

sys.stdout = sys.__stdout__

# 保存捕获到的控制台内容
with open(r"D:\program\py\mpaper_decetion_10_25\resnet18_SBE_unblock\console.txt", "w", encoding="utf-8") as f:
    f.write(console_buffer.getvalue())

print("✅ 控制台输出已保存为 console_output.txt")
'''===== K 折验证结果 =====
Fold 1: Val Acc = 0.8538
Fold 2: Val Acc = 0.8087
Fold 3: Val Acc = 0.8400
Fold 4: Val Acc = 0.8387
Fold 5: Val Acc = 0.8337
平均准确率: 0.8350
'''