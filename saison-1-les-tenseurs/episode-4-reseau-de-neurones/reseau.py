# Épisode 4 — Comment un réseau de neurones utilise les tenseurs
from pathlib import Path
import torch
from torch import nn

# Le chiffre 7 des épisodes précédents
image = torch.tensor([
    [0,   1,   5,  13,  16,  11,   4, 0],
    [0, 199, 250, 253, 253, 253, 188, 0],
    [0,  19,  22,  11,   9, 192, 115, 0],
    [0,   0,   0,   0,  43, 242,  17, 0],
    [0,   0,   0,   0, 153, 150,   0, 0],
    [0,   0,   0,  20, 243,  40,   0, 0],
    [0,   0,   0, 114, 181,   0,   0, 0],
    [0,   0,   0,  27,  27,   0,   0, 0],
], dtype=torch.uint8)

# 1. Un réseau de deux couches : 64 -> 16 -> 10
torch.manual_seed(0)
reseau = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
)

# 2. Ses poids sont des tenseurs
print(reseau[1].weight.shape)
print(reseau[3].weight.shape)
print(sum(p.numel() for p in reseau.parameters()))

# 3. L'entrée : un lot d'une image, valeurs entre 0 et 1
x = (image / 255).unsqueeze(0)
print(x.shape)

# 4. Le trajet du tenseur à travers le réseau
print(reseau[0](x).shape)
print(reseau[1](reseau[0](x)).shape)
scores = reseau(x)
print(scores.shape)

# 5. Avant entraînement : des poids au hasard
proba = scores.softmax(dim=1)
print(proba.argmax(dim=1))

# 6. Après entraînement (poids produits par entrainer.py)
reseau.load_state_dict(torch.load(Path(__file__).with_name("reseau_entraine.pt")))
proba = reseau(x).softmax(dim=1)
print(proba.argmax(dim=1))
print(round(proba[0, 7].item(), 3))
