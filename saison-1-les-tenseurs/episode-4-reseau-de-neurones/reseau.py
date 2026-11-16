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
print("1. Le réseau, couche par couche :")
print(reseau)

# 2. Ses poids sont des tenseurs
print("2. Poids de la couche 1 (16 neurones x 64 pixels) :", reseau[1].weight.shape)
print("   Poids de la couche 2 (10 chiffres x 16 neurones) :", reseau[3].weight.shape)
print("   Nombre total de paramètres (poids + biais) :", sum(p.numel() for p in reseau.parameters()))

# 3. L'entrée : un lot d'une image, valeurs entre 0 et 1
x = (image / 255).unsqueeze(0)
print("3. Forme de l'entrée (1 image, 8 lignes, 8 colonnes) :", x.shape)

# 4. Le trajet du tenseur à travers le réseau
print("4. Après Flatten (64 pixels à la suite) :", reseau[0](x).shape)
print("   Après la couche 1 (16 neurones) :", reseau[1](reseau[0](x)).shape)
scores = reseau(x)
print("   Sortie du réseau (un score par chiffre, de 0 à 9) :", scores.shape)

# 5. Avant entraînement : des poids au hasard
proba = scores.softmax(dim=1)
print("5. Réponse avec des poids au hasard (chiffre le plus probable) :", proba.argmax(dim=1))

# 6. Après entraînement (poids produits par entrainer.py)
reseau.load_state_dict(torch.load(Path(__file__).with_name("reseau_entraine.pt")))
proba = reseau(x).softmax(dim=1)
print("6. Réponse après entraînement (chiffre le plus probable) :", proba.argmax(dim=1))
print("   Probabilité que ce soit un 7 :", round(proba[0, 7].item(), 3))
