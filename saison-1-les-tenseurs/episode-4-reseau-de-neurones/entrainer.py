# Épisode 4 — entraînement (hors caméra) du réseau 64 -> 16 -> 10
# Base : chiffres manuscrits 8 x 8 de scikit-learn (1 797 images, valeurs de 0 à 16).
# Le détail de l'apprentissage sera expliqué à l'épisode 5 ; ce script sert seulement
# à produire des poids réels et vérifiables pour la démonstration.
from pathlib import Path
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

donnees = load_digits()
X = torch.tensor(donnees.images / 16, dtype=torch.float32)       # valeurs entre 0 et 1
y = torch.tensor(donnees.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

torch.manual_seed(0)
reseau = nn.Sequential(nn.Flatten(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10))
optimiseur = torch.optim.Adam(reseau.parameters(), lr=0.01)
perte = nn.CrossEntropyLoss()
for epoque in range(400):
    optimiseur.zero_grad()
    l = perte(reseau(X_train), y_train)
    l.backward()
    optimiseur.step()

with torch.no_grad():
    exactitude = (reseau(X_test).argmax(dim=1) == y_test).float().mean().item()
print(f"exactitude sur {len(y_test)} images jamais vues : {exactitude:.1%}")
torch.save(reseau.state_dict(), Path(__file__).with_name("reseau_entraine.pt"))
