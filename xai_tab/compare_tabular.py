import numpy as np
import torch
from captum.attr import IntegratedGradients
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from train import MLP  
from metrics import (
    feature_disagreement,
    sign_disagreement,
    euclidean,
    euclidean_abs
)

# 1) dati e preprocessing
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# 2) carica modelli
modelA = MLP(X.shape[1]); modelA.load_state_dict(torch.load("models/mlp_seed0.pt", weights_only=True))
modelB = MLP(X.shape[1]); modelB.load_state_dict(torch.load("models/mlp_seed1.pt", weights_only=True))
modelA.eval(); modelB.eval()

# 3) genera spiegazioni IG
igA = IntegratedGradients(modelA)
igB = IntegratedGradients(modelB)
baseline = torch.zeros((1, X.shape[1])) 
attrA = igA.attribute(X_test_t, baselines=baseline, n_steps=50).detach().numpy()
attrB = igB.attribute(X_test_t, baselines=baseline, n_steps=50).detach().numpy()

# 4) calcola metriche per ciascuna istanza
k = 8  # numero di feature top-k
fd = []; sd = []; eu = []; ea = []
for a, b in zip(attrA, attrB):
    fd.append(feature_disagreement(a, b, k))
    sd.append(sign_disagreement(a, b, k))
    eu.append(euclidean(a, b))
    ea.append(euclidean_abs(a, b))

# 5) stampa media ± std
print(f"FeatureDisagreement : {np.mean(fd):.3f} ± {np.std(fd):.3f}")
print(f"SignDisagreement    : {np.mean(sd):.3f} ± {np.std(sd):.3f}")
print(f"Euclidean           : {np.mean(eu):.3f} ± {np.std(eu):.3f}")
print(f"Euclidean-abs       : {np.mean(ea):.3f} ± {np.std(ea):.3f}")