import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# semplice MLP per 30 feature
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 16), nn.ReLU(),   # primo layer nascosto, 16 unità
            nn.Linear(16, 16), nn.ReLU(),     # secondo layer nascosto, 16 unità
            nn.Linear(16, 1)                  # output
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze()

def train_and_save(seed, out_path):
    torch.manual_seed(seed)
    # 1) dataset
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    # 2) modello, ottimizzatore, loss
    model = MLP(d_in=X.shape[1])
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    # 3) training
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
    # 4) salva
    torch.save(model.state_dict(), out_path)
    print(f"Saved {out_path}, final loss {loss.item():.4f}")

if __name__ == "__main__":
    import os, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1])
    args = parser.parse_args()
    os.makedirs("models", exist_ok=True)
    for s in args.seeds:
        train_and_save(s, f"models/mlp_seed{s}.pt")