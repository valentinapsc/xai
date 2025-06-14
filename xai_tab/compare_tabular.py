import numpy as np
import torch
from captum.attr import IntegratedGradients
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import diretto della classe MLP che hai definito in train.py
from train import MLP  

# Import delle quattro funzioni di disaccordo definite in metrics.py
from metrics import (
    feature_disagreement,
    sign_disagreement,
    euclidean,
    euclidean_abs
)

# 1) Caricamento dati e preprocessing

# Scarica i dati del Breast Cancer da sklearn: X = vettore di feature, y = etichette (0/1)
X, y = load_breast_cancer(return_X_y=True)

# Standardizzazione: ogni colonna (feature) viene portata a media 0 e varianza 1
X = StandardScaler().fit_transform(X)

# Divisione in train/test: qui ci serve solo X_test per le spiegazioni
# stratify=y mantiene le proporzioni di classe, random_state=0 rende la divisione riproducibile
_, X_test, _, _ = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=0, 
    stratify=y
)

# Converte X_test in un tensore PyTorch di tipo float32
X_test_t = torch.tensor(X_test, dtype=torch.float32)


# 2) Caricamento dei modelli già addestrati

# Istanzia due reti MLP con lo stesso numero di feature in ingresso
modelA = MLP(X.shape[1])  
modelB = MLP(X.shape[1])

# Carica i pesi salvati precedentemente. weights_only=True evita il warning di sicurezza
modelA.load_state_dict(torch.load("models/mlp_seed0.pt", weights_only=True))
modelB.load_state_dict(torch.load("models/mlp_seed1.pt", weights_only=True))

# Imposta i modelli in modalità "evaluation": disattiva dropout e batchnorm training-mode
modelA.eval()
modelB.eval()

# 3) Generazione delle spiegazioni con Integrated Gradients

# Crea gli oggetti explainer per ciascun modello
igA = IntegratedGradients(modelA)
igB = IntegratedGradients(modelB)

# Definisce il baseline: un vettore di zeri, stessa dimensione delle feature, per il calcolo di IG
baseline = torch.zeros((1, X.shape[1])) 

# Calcola le attribuzioni per tutti i campioni di test:
# - baselines=baseline: punto di partenza per l'integrazione
# - n_steps=50: numero di passi dell'integrazione (più passi = spiegazione più stabile)
# Il risultato è un tensore PyTorch di forma (n_samples, n_features)
attrA = igA.attribute(X_test_t, baselines=baseline, n_steps=50).detach().numpy()
attrB = igB.attribute(X_test_t, baselines=baseline, n_steps=50).detach().numpy()

# indice del campione
i = 0
a, b = attrA[i], attrB[i]

# Calcola disagreement *per quel campione*
fd_i  = feature_disagreement(a, b, k=8) # k=8 è il numero di feature da considerare
sd_i  = sign_disagreement   (a, b, k=8)
euc_i = euclidean           (a, b)
eua_i = euclidean_abs       (a, b)

# 4) Calcolo delle metriche di disaccordo 

import matplotlib.pyplot as plt

feature_names = load_breast_cancer().feature_names

x = np.arange(len(feature_names))
width = 0.35

plt.figure(figsize=(10,5))
plt.bar(x - width/2, a, width, label="Modello A")
plt.bar(x + width/2, b, width, label="Modello B")
plt.xticks(x, feature_names, rotation=45, ha="right")
plt.ylabel("Valore di attributo")
plt.title(
    f"Attributions confronto campione {i}\n"
    f"FD={fd_i:.2f}  SD={sd_i:.2f}  Euc={euc_i:.2f}  Eua={eua_i:.2f}"
)
plt.legend()
plt.tight_layout()
plt.show()