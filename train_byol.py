import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import copy
import random
import numpy as np
from tqdm import tqdm
import os
import statistics
import json


# ----------------------------
#  ФИКСИРОВАНИЕ СИДОВ
# ----------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------------------
#  TwoTransform (два вида изображения)
# ----------------------------
class TwoTransform:
    def __init__(self, base_transform):
        self.base = base_transform

    def __call__(self, x):
        return self.base(x), self.base(x)


# ----------------------------
#  Аугментации
# ----------------------------
mnist_aug = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

linear_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ----------------------------
#  ЭНКОДЕР CNN
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, rep_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 14×14
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 7×7
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(128, rep_dim)

    def forward(self, x):
        return self.fc(self.net(x))


# ----------------------------
#  MLP — проектор и предиктор
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
#  BYOL
# ----------------------------
class BYOL:
    def __init__(self, encoder, projector_dim, pred_hidden, tau):
        self.student_encoder = encoder
        self.student_projector = MLP(encoder.fc.out_features, pred_hidden, projector_dim)
        self.student_predictor = MLP(projector_dim, pred_hidden, projector_dim)

        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher_projector = copy.deepcopy(self.student_projector)

        # freeze teacher
        self._set_requires_grad(self.teacher_encoder, False)
        self._set_requires_grad(self.teacher_projector, False)

        self.teacher_encoder.eval()
        self.teacher_projector.eval()

        self.tau = tau

    @staticmethod
    def _set_requires_grad(model, req):
        for p in model.parameters():
            p.requires_grad = req

    def to(self, device):
        self.student_encoder.to(device)
        self.student_projector.to(device)
        self.student_predictor.to(device)
        self.teacher_encoder.to(device)
        self.teacher_projector.to(device)

    def student_forward(self, x):
        y = self.student_encoder(x)
        z = self.student_projector(y)
        p = self.student_predictor(z)
        return y, z, p

    @torch.no_grad()
    def teacher_forward(self, x):
        y = self.teacher_encoder(x)
        z = self.teacher_projector(y)
        return y, z

    @torch.no_grad()
    def update_teacher(self):
        for param_q, param_k in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            param_k.mul_(self.tau).add_(param_q, alpha=1 - self.tau)

        for param_q, param_k in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            param_k.mul_(self.tau).add_(param_q, alpha=1 - self.tau)


# ----------------------------
#  BYOL LOSS
# ----------------------------
def byol_loss(p, z_target):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z_target, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1)


# ----------------------------
#  Коллапс-чек
# ----------------------------
def make_collapse_loader(n_samples=512):
    ds = datasets.MNIST(root='./data', train=True, transform=linear_aug, download=True)
    subset_idx = torch.randperm(len(ds))[:n_samples]
    subset = torch.utils.data.Subset(ds, subset_idx)
    loader = DataLoader(subset, batch_size=256, shuffle=False)
    return loader


@torch.no_grad()
def compute_trace_cov(encoder, loader, device):
    encoder.eval()
    feats = []

    for x, _ in loader:
        x = x.to(device)
        y = encoder(x)
        feats.append(y.cpu())

    feats = torch.cat(feats, dim=0)
    mu = feats.mean(dim=0, keepdim=True)
    X = feats - mu
    cov = (X.T @ X) / (X.size(0) - 1)
    return torch.trace(cov).item()


# ----------------------------
# TRAIN BYOL
# ----------------------------
def train_byol(byol, dataloader, optimizer, device, epochs):
    collapse_loader = make_collapse_loader()
    loss_list = []
    trace_list = []

    for epoch in range(1, epochs + 1):

        byol.student_encoder.train()
        byol.student_projector.train()
        byol.student_predictor.train()

        current_epoch = 0.0

        for (x1, x2), _ in tqdm(dataloader, desc=f"BYOL epoch {epoch}/{epochs}"):
            x1, x2 = x1.to(device), x2.to(device)

            _, z1, p1 = byol.student_forward(x1)
            _, z2, p2 = byol.student_forward(x2)

            with torch.no_grad():
                _, z1_t = byol.teacher_forward(x1)
                _, z2_t = byol.teacher_forward(x2)

            loss = (byol_loss(p1, z2_t).mean() + byol_loss(p2, z1_t).mean()) * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            byol.update_teacher()

            current_epoch += loss.item() * x1.size(0)

        avg_loss = current_epoch / len(dataloader.dataset)
        trace_val = compute_trace_cov(byol.student_encoder, collapse_loader, device)

        loss_list.append(avg_loss)
        trace_list.append(trace_val)

        print(f"Epoch {epoch}: BYOL loss = {avg_loss:.4f}, trace(cov) = {trace_val:.2f}")

    return loss_list, trace_list


# ----------------------------
#  ЛИНЕЙНАЯ ГОЛОВА
# ----------------------------
def train_linear(encoder, train_loader, test_loader, device, epochs, lr):
    for p in encoder.parameters():
        p.requires_grad = False

    feat_dim = encoder.fc.out_features
    linear = nn.Linear(feat_dim, 10).to(device)
    opt = torch.optim.Adam(linear.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        linear.train()
        current = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = encoder(x)
            logits = linear(feats)

            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            current += loss.item() * x.size(0)

        acc = evaluate_encoder_linear(encoder, linear, test_loader, device)
        print(f"Linear epoch {epoch}: loss={current / len(train_loader.dataset):.4f}, acc={acc:.2f}%")

    return linear


@torch.no_grad()
def evaluate_encoder_linear(encoder, linear, test_loader, device):
    encoder.eval()
    linear.eval()
    correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        feats = encoder(x)
        preds = linear(feats).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return 100 * correct / total


# ----------------------------
# Датасеты
# ----------------------------
class MNISTTwoView(datasets.MNIST):
    def __init__(self, root, train, transform, download):
        super().__init__(root=root, train=train, transform=None, download=download)
        self.twotransform = transform

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        v1, v2 = self.twotransform(img)
        return (v1, v2), target


def make_dataloaders(batch_size_pretrain, batch_size_eval):
    g = torch.Generator().manual_seed(42)

    pretrain_ds = MNISTTwoView(root='./data', train=True,
                               transform=TwoTransform(mnist_aug), download=True)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=batch_size_pretrain,
                                 shuffle=True, num_workers=0, drop_last=True, generator=g)

    train_ds = datasets.MNIST(root='./data', train=True, transform=linear_aug, download=True)
    test_ds = datasets.MNIST(root='./data', train=False, transform=linear_aug, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size_eval, shuffle=True, num_workers=0, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size_eval, shuffle=False, num_workers=0)

    return pretrain_loader, train_loader, test_loader


# ----------------------------
# MAIN
# ----------------------------
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--tau", type=float, default=0.98)
    # parser.add_argument("--epochs", type=int, default=25)
    # parser.add_argument("--batch-size", type=int, default=256)
    # parser.add_argument("--linear-epochs", type=int, default=50)
    # args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device:", device)

    # pretrain_loader, train_loader, test_loader = make_dataloaders(
    #     batch_size_pretrain=args.batch_size,
    #     batch_size_eval=args.batch_size
    # )

    # encoder = Encoder(rep_dim=128)
    # byol = BYOL(encoder, projector_dim=64, pred_hidden=128, tau=args.tau)
    # byol.to(device)

    # student_params = list(byol.student_encoder.parameters()) + \
    #                  list(byol.student_projector.parameters()) + \
    #                  list(byol.student_predictor.parameters())

    # optimizer = torch.optim.Adam(student_params, lr=1e-3, weight_decay=1e-6)

    # print("\n==== Обучение BYOL ====")
    # train_byol(byol, pretrain_loader, optimizer, device, epochs=args.epochs)

    # print("\n==== Обучение линейной головы ====")
    # frozen_encoder = byol.student_encoder
    # linear = train_linear(frozen_encoder, train_loader, test_loader,
    #                       device, epochs=args.linear_epochs, lr=1e-3)

    # print("\n==== Сохранение моделей ====")

    # torch.save(byol.student_encoder.state_dict(), "encoder_final.pth")
    # torch.save(byol, "byol_full_model.pth")
    # torch.save(linear.state_dict(), "linear_head.pth")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.98)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--linear-epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_file = os.path.join(results_dir, f"byol_tau{args.tau:.4f}.json")

    tau_str = str(args.tau)
    seed_accuracies = []

    for seed in args.seeds:
        print(f"\n==== Обучение на seed - {seed} ====")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        pretrain_loader, train_loader, test_loader = make_dataloaders(
            batch_size_pretrain=args.batch_size,
            batch_size_eval=args.batch_size
        )

        encoder = Encoder(rep_dim=128)
        byol = BYOL(encoder, projector_dim=64, pred_hidden=128, tau=args.tau)
        byol.to(device)

        student_params = list(byol.student_encoder.parameters()) + \
                         list(byol.student_projector.parameters()) + \
                         list(byol.student_predictor.parameters())

        optimizer = torch.optim.Adam(student_params, lr=1e-3, weight_decay=1e-6)

        print("\n==== Обучение BYOL ====")
        train_byol(byol, pretrain_loader, optimizer, device, epochs=args.epochs)

        print("\n==== Обучение линейной головы ====")
        frozen_encoder = byol.student_encoder
        linear = train_linear(frozen_encoder, train_loader, test_loader,
                              device, epochs=args.linear_epochs, lr=1e-3)

        acc = evaluate_encoder_linear(frozen_encoder, linear, test_loader, device)
        print(f"Seed {seed}, точность: {acc:.2f}%")
        seed_accuracies.append(acc)

    result_data = {
        tau_str: {
            "seed_accuracies": seed_accuracies,
            "mean": statistics.mean(seed_accuracies),
            "std": statistics.stdev(seed_accuracies) if len(seed_accuracies) > 1 else 0.0
        }
    }

    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            old_data = json.load(f)
        old_data.update(result_data)
        result_data = old_data

    with open(json_file, "w") as f:
        json.dump(result_data, f, indent=4)

    print(f"Сохранено в {json_file}")


if __name__ == "__main__":
    main()