import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients

# --- dati ---
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32)

# --- modello ---
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze()

model = MLP(d_in=X.shape[1])

# --- training veloce ---
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()
for epoch in range(200):
    opt.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward(); opt.step()
print(f"train-loss {loss.item():.4f}")

# --- spiegazione IG su un campione ---
ig = IntegratedGradients(model)

# baseline: vettore di zeri con la *stessa* shape del batch [1, 30]
baseline = torch.zeros((1, X_test.shape[1]))

# attributi (n_steps = 50 è il nostro "p")
attr = ig.attribute(
    X_test[0].unsqueeze(0),   # batch di dimensione 1
    baselines=baseline,       # NB: "baselines", plurale
    n_steps=50
).squeeze().detach().numpy()  # → shape [30]

attr_t = ig.attribute(X_test[0].unsqueeze(0),
                      baselines=baseline,
                      n_steps=50).squeeze()        # tensore shape [30]
top8 = torch.argsort(attr_t.abs(), descending=True)[:8]
for idx in top8:
    print(f"feature {idx.item():2d} → {attr_t[idx].item():+.4f}")
