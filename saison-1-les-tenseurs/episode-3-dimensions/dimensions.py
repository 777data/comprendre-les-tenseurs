# Épisode 3 — Comprendre les dimensions d'un tenseur
import torch

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

# 1. Rang 0, 1, 2 : un pixel, une ligne, l'image
pixel = image[1, 3]
ligne = image[1]
print(pixel.ndim, pixel.shape)
print(ligne.ndim, ligne.shape)
print(image.ndim, image.shape)

# 2. Les axes
#    axe 0 : on additionne en descendant les lignes  -> une somme par colonne
#    axe 1 : on additionne en parcourant chaque ligne -> une somme par ligne
print(image.sum(dim=0))
print(image.sum(dim=1))

# 3. Aplatir, puis remodeler
liste = image.flatten()
print(liste.shape)
grille = liste.reshape(8, 8)
print(grille.shape)
print(torch.equal(grille, image))

# 4. Rang 3 : un 7 rouge, en trois canaux (rouge, vert, bleu)
noir = torch.zeros_like(image)
rouge = torch.stack([image, noir, noir])
print(rouge.ndim, rouge.shape)

# 5. Rang 4 : un lot d'images
lot = rouge.unsqueeze(0)
print(lot.ndim, lot.shape)
