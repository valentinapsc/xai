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

# Caricamento dati e preprocessing

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

# Caricamento dei modelli già addestrati

# Istanzia due reti MLP con lo stesso numero di feature in ingresso
modelA = MLP(X.shape[1])  
modelB = MLP(X.shape[1])

# Carica i pesi salvati precedentemente. weights_only=True evita il warning di sicurezza
modelA.load_state_dict(torch.load("models/mlp_seed0.pt", weights_only=True))
modelB.load_state_dict(torch.load("models/mlp_seed1.pt", weights_only=True))

# Imposta i modelli in modalità "evaluation": disattiva dropout e batchnorm training-mode
modelA.eval()
modelB.eval()

# Generazione delle spiegazioni con Integrated Gradients

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

# Calcola disagreement *per quel campione*
fd_list  = []
sd_list  = []
euc_list = []
eua_list = []

for a, b in zip(attrA, attrB):
    fd_i  = feature_disagreement(a, b, k=8)
    sd_i  = sign_disagreement   (a, b, k=8)
    euc_i = euclidean           (a, b)
    eua_i = euclidean_abs       (a, b)

    # ora appendi i valori calcolati
    fd_list .append(fd_i)
    sd_list .append(sd_i)
    euc_list.append(euc_i)
    eua_list.append(eua_i)

# media e std
fd_mean,  fd_std  = np.mean(fd_list),  np.std(fd_list)
sd_mean,  sd_std  = np.mean(sd_list),  np.std(sd_list)
euc_mean, euc_std = np.mean(euc_list), np.std(euc_list)
eua_mean, eua_std = np.mean(eua_list), np.std(eua_list)

print(f"FeatureDisagreement : {fd_mean :.3f} ± {fd_std :.3f}")
print(f"SignDisagreement    : {sd_mean :.3f} ± {sd_std :.3f}")
print(f"Euclidean           : {euc_mean:.3f} ± {euc_std:.3f}")
print(f"Euclidean-abs       : {eua_mean:.3f} ± {eua_std:.3f}")

# Calcolo delle metriche di disaccordo 

import matplotlib.pyplot as plt

# scegli il campione i che vuoi mostrare
i = 0

# crea la figura con 2 righe, 1 colonna
fig, (ax_glob, ax_case) = plt.subplots(2, 1, figsize=(10, 8))

# bar-plot delle metrcihe globali

labels = ["FD", "SD", "Euc", "Eua"]
means  = [fd_mean, sd_mean, euc_mean, eua_mean]
stds   = [fd_std,  sd_std,  euc_std,  eua_std]
ax_glob.bar(labels, means, yerr=stds, capsize=5, color=["C0","C1","C2","C3"])
ax_glob.set_title("Disaccordo medio ± std su tutto il test set")
ax_glob.set_ylabel("Valore metrica")

# box-plot delle metriche globali
# data   = [fd_list, sd_list, euc_list, eua_list]
# tick_labels = ["FD", "SD", "Euc", "Eua"]

#ax_glob.boxplot(
#    data,
#    labels=labels,
#    showfliers=True,    # mostra gli outlier come puntini
#    patch_artist=True,  # colora le scatole
#    boxprops=dict(facecolor="lightgray", color="black"),
#    whiskerprops=dict(color="black"),
#    medianprops=dict(color="red")
#)
#ax_glob.set_title("Distribuzione delle metriche di disaccordo\nsu tutto il test set")
#ax_glob.set_ylabel("Valore metrica")

# --- pannello 2: case study sul campione i ---
# barre affiancate per feature
feature_names = load_breast_cancer().feature_names
x = np.arange(len(feature_names))
width = 0.35
a, b = attrA[i], attrB[i]
ax_case.bar(x - width/2, a, width, label="Modello A")
ax_case.bar(x + width/2, b, width, label="Modello B")
ax_case.set_xticks(x)
ax_case.set_xticklabels(feature_names, rotation=45, ha="right")
ax_case.set_ylabel("Attribution IG")
ax_case.set_title(f"Attributions confronto campione {i}  |  "
                  f"FD={fd_list[i]:.2f} SD={sd_list[i]:.2f} "
                  f"Euc={euc_list[i]:.2f} Eua={eua_list[i]:.2f}")
ax_case.legend()

plt.tight_layout()
plt.show()