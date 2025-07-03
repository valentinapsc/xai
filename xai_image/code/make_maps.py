# utilizzo di Integrated Gradients e Saliency per calcolare le mappe di salienza (di calore) che evidenziano le aree importanti dell'immagine

import numpy as np                                    # numpy per salvataggio mappe
import torchvision.models as models                   # modelli pre-addestrati
import torchvision.transforms as T                    # trasformazioni per l'immagine
from captum.attr import IntegratedGradients, Saliency # metodi di interpretazione per modelli PyTorch
from PIL import Image                                 # per caricare l'immagine e a convertirla in array

# --- parametri ---
IMG_PATH = "cat.jpg" 
OUT_A    = "saliency_A.npy"    # IG (è il metodo più robusto e preciso per calcolare le mappe di salienza)
OUT_B    = "saliency_B.npy"    # gradiente puro (saliency è il metodo più semplice e veloce, ma meno preciso)

# --- carica modello e immagine ---
from torchvision.models import ResNet18_Weights           # per caricare il modello ResNet18 con pesi pre-addestrati
model = models.resnet18(weights=ResNet18_Weights.DEFAULT) # carica il modello
tf = T.Compose([                                             # trasformazioni da applicare all'immagine
        T.Resize((224,224)),                                 # ridimensiona l'immagine a 224x224
        T.ToTensor(),                                        # converte l'immagine in un tensore (matrice multidimensionale)
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # scala ciascun canale RGB sottraendo mean e dividendo per std, seguendo la normalizzazione usata in addestramento
])
x = tf(Image.open(IMG_PATH).convert("RGB")).unsqueeze(0) # apre il file JPEG e garantisce che sia in formato a 3 canali (rosso, verde, blu) [PyTorch si aspetta sempre una forma (batch_size, canali, H, W)]
pred = model(x).argmax(dim=1)                            # predizione del modello (classe con probabilità più alta)

# --- Integrated Gradients ---
ig = IntegratedGradients(model) # calcola gli attributi di importanza delle caratteristiche con Integrated Gradients
attr_ig = ig.attribute(
        x, 
        baselines=x*0,                                  # definisce la “riferimento zero” come un’immagine nera (stessi pixel moltiplicati per zero).
        target=pred).sum(1).squeeze().detach().numpy()  # somma lungo l'asse dei canali e rimuove la dimensione del batch (1), poi converte in numpy
np.save(OUT_A, attr_ig)                                 # salva gli attributi in un file numpy

# --- Saliency (gradiente) ---
sal = Saliency(model)     # calcola gli attributi di importanza delle caratteristiche con Saliency
attr_grad = sal.attribute(x, target=pred).sum(1).squeeze().detach().numpy() 
np.save(OUT_B, attr_grad) # salva gli attributi in un file numpy

print("saliency_A.npy e saliency_B.npy salvati.")