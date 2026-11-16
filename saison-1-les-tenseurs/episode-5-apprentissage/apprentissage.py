# Épisode 5 — Comment l'IA apprend à partir des données
# Même réseau, mêmes données et mêmes réglages que l'épisode 4 (entrainer.py),
# mais en enregistrant ce qui se passe à chaque passe.
import json
from pathlib import Path
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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
x7 = (image / 255).unsqueeze(0)

donnees = load_digits()
X = torch.tensor(donnees.images / 16, dtype=torch.float32)
y = torch.tensor(donnees.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

torch.manual_seed(0)
reseau = nn.Sequential(nn.Flatten(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10))

# 1. La perte : un nombre qui mesure l'erreur
perte = nn.CrossEntropyLoss()
l = perte(reseau(X_train), y_train)
print(round(l.item(), 3))

# 2. Les gradients : un tenseur de même forme que chaque poids
l.backward()
print(reseau[1].weight.grad.shape)
print(reseau[3].weight.grad.shape)
grad0 = reseau[1].weight.grad.clone()
reseau.zero_grad()

# 3. La boucle d'entraînement
optimiseur = torch.optim.Adam(reseau.parameters(), lr=0.01)
journal = []
for epoque in range(400):
    optimiseur.zero_grad()
    l = perte(reseau(X_train), y_train)
    l.backward()
    optimiseur.step()
    with torch.no_grad():
        journal.append({
            "epoque": epoque + 1,
            "perte": l.item(),
            "test": (reseau(X_test).argmax(dim=1) == y_test).float().mean().item(),
            "p7": reseau(x7).softmax(dim=1)[0, 7].item(),
        })

with torch.no_grad():
    print(round(perte(reseau(X_train), y_train).item(), 3))
    print(reseau(x7).softmax(dim=1).argmax(dim=1))
    print(round(reseau(x7).softmax(dim=1)[0, 7].item(), 3))
    print(round((reseau(X_test).argmax(dim=1) == y_test).float().mean().item(), 3))

# Données pour les animations
json.dump({"journal": journal, "grad0": grad0.tolist()}, open(Path(__file__).with_name("journal_entrainement.json"), "w"))
