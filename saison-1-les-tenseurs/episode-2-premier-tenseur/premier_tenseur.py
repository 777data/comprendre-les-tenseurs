# Épisode 2 — Créer son premier tenseur avec Python
# Installation (une seule fois) :  pip install torch

import torch

# Les 64 valeurs du chiffre 7 de l'épisode 1, ligne par ligne
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

# 1. Sa forme : 8 lignes, 8 colonnes
print("1. Forme du tenseur (lignes, colonnes) :", image.shape)

# 2. Son nombre de valeurs
print("2. Nombre de valeurs (8 x 8) :", image.numel())

# 3. Le type de ses valeurs : entiers sur 8 bits
print("3. Type des valeurs (entiers de 0 à 255, sur 8 bits) :", image.dtype)

# 4. Lire un pixel : ligne 2, colonne 4 (on compte à partir de 0)
print("4. Pixel ligne 2, colonne 4 (image[1, 3]) :", image[1, 3])

# 5. Lire une ligne entière : la deuxième
print("5. Deuxième ligne entière (image[1]) :", image[1])

# 6. Calculer sur les 64 valeurs à la fois : l'image en négatif
negatif = 255 - image
print("6. En négatif (255 - valeur), le même pixel blanc devient presque noir :", negatif[1, 3])
print("   et le coin noir en haut à gauche devient blanc :", negatif[0, 0])

# 7. Ramener les valeurs entre 0 et 1
image_norm = image / 255
print("7. Le même pixel ramené entre 0 et 1 (253 / 255) :", image_norm[1, 3])
print("   Type des valeurs après la division (nombres à virgule) :", image_norm.dtype)
