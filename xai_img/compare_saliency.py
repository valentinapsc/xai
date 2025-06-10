import numpy as np                                     # numpy per calcoli e salvataggio mappe
from skimage import io, transform, segmentation, color # librerie per elaborazione immagini
import matplotlib.pyplot as plt                        # per visualizzare le immagini
from pathlib import Path                               # per gestire i percorsi dei file

# ---------- parametri ----------
IMAGE_PATH   = "cat.jpg"          # immagine RGB
MAP1_PATH    = "saliency_A.npy"   # mappe saliency shape (H,W) float32
MAP2_PATH    = "saliency_B.npy"
K_FRAC       = 0.05               # top-k = 5 % dei super-pixel
N_SEGMENTS   = 200                # granularità SLIC (200 super-pixel)

# carica saliency (shape H×W) 
s1 = np.load(MAP1_PATH)
s2 = np.load(MAP2_PATH)

# ridimensiona l’immagine originale alla stessa forma di saliency
# potrebbe essere più grande o più piccola rispetto alle mappe
img_orig = io.imread(IMAGE_PATH)                               # shape (H0,W0,3)
H, W = s1.shape
img = transform.resize(img_orig, (H, W, img_orig.shape[2]),
                       preserve_range=True).astype(img_orig.dtype)

# ---------- super-pixel SLIC ----------
# SLIC è un algoritmo che raggruppa i pixel vicini e simili di colore in blocchi (detti super-pixel)
segments = segmentation.slic(img, n_segments=N_SEGMENTS, compactness=20, start_label=0) # segmenta l'immagine in super-pixel
# segments è un array 2D di etichette di super-pixel (shape H×W)
n_sp = segments.max() + 1 # numero effettivo di blocchi creati

# per ogni super-pixel calcola il valore medio di saliency
def sp_mean(saliency):
    out = np.zeros(n_sp, dtype=np.float32) # vettore di lunghezza n_sp
    for sp in range(n_sp): 
        mask = segments == sp              # maschera per il super-pixel corrente
        out[sp] = saliency[mask].mean()    # calcola la media della saliency per il super-pixel
    return out

vec1 = sp_mean(s1)    # vettori di lunghezza n_sp
vec2 = sp_mean(s2)

# ---------- metriche ----------
def topk_idx(v, k): return np.argsort(np.abs(v))[-k:][::-1] # trova gli indici dei k valori assoluti più grandi in v

# calcola le metriche di disaccordo tra i due vettori di saliency 

# feature_disagreement: disaccordo tra i super-pixel con saliency più alta (Conta la frazione di super-pixel tra i più importanti che non coincidono)
def feature_disagreement(a,b,k):
    A,B = set(topk_idx(a,k)), set(topk_idx(b,k))
    return 1 - len(A & B)/k

# sign_disagreement: disaccordo tra i super-pixel con saliency più alta, considerando il segno (penalizza anche le differenze di segno)
def sign_disagreement(a,b,k):
    A,B = set(topk_idx(a,k)), set(topk_idx(b,k))
    overlap = A & B
    mismatch = sum(np.sign(a[i])!=np.sign(b[i]) for i in overlap)
    return (k - len(overlap) + mismatch)/k

# euclidean: distanza euclidea tra i due vettori di saliency normalizzati (Misura la distanza globale in uno spazio a n_sp dimensioni)
def euclidean(a,b):
    a,b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    return np.linalg.norm(a-b)

# euclidean_abs: distanza euclidea tra i due vettori di saliency normalizzati, considerando il valore assoluto e ignorando il segno
def euclidean_abs(a,b):
    a,b = np.abs(a), np.abs(b)
    a,b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    return np.linalg.norm(a-b)

# numero di super-pixel da considerare (5 % di n_sp)
k = int(K_FRAC * n_sp) 

# restituiscono un numero da 0 (perfetto accordo) a 1 o 2 (massimo disaccordo)
fd  = feature_disagreement(vec1, vec2, k)
sd  = sign_disagreement   (vec1, vec2, k)
euc = euclidean           (vec1, vec2)
eua = euclidean_abs       (vec1, vec2)

print(f"FeatureDisagreement : {fd:.3f}")
print(f"SignDisagreement    : {sd:.3f}")
print(f"Euclidean           : {euc:.3f}")
print(f"Euclidean-abs       : {eua:.3f}")

# ---------- overlay visivo per intuire le differenze ----------
def overlay(img, sal, title):
    sal_norm = (sal - sal.min())/(np.ptp(sal) + 1e-6)      # normalizza la saliency tra 0 e 1
    heat = plt.cm.jet(sal_norm)[..., :3]                   # Applica la mappa colori “jet” per trasformare ogni numero in un colore (rosso = alto, blu = basso)
    blended = 0.6*color.rgb2gray(img)[...,None] + 0.4*heat # blenda l'immagine originale in scala di grigi con la mappa di calore per far vedere sia la forma sia la rilevanza.
    plt.imshow(blended); plt.axis('off'); plt.title(title) 

# visualizza l'immagine originale e le due mappe di saliency
plt.figure(figsize=(12,4)) 
plt.subplot(1,3,1); plt.imshow(img); plt.axis('off'); plt.title("immagine")
plt.subplot(1,3,2); overlay(img, s1, "saliency A")
plt.subplot(1,3,3); overlay(img, s2, "saliency B")
plt.tight_layout(); plt.show()