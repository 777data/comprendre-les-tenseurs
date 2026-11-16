# Épisode 1 — De l'image à la donnée numérique
# Pour un ordinateur, une image n'est qu'une grille de nombres.
# Ici, aucune bibliothèque d'IA : seulement Python et le fichier matrice_8x8.csv.
import csv
from pathlib import Path

# 1. Lire les 64 valeurs du chiffre 7 (0 = noir, 255 = blanc)
with open(Path(__file__).with_name("matrice_8x8.csv")) as f:
    image = [[int(v) for v in ligne] for ligne in csv.reader(f)]

print("1. Taille de l'image :", len(image), "lignes x", len(image[0]), "colonnes")
print("   Nombre total de pixels (donc de nombres) :", sum(len(ligne) for ligne in image))

# 2. La matrice, telle que l'ordinateur la voit
print()
print("2. L'image telle que l'ordinateur la voit : 64 nombres entre 0 (noir) et 255 (blanc)")
for ligne in image:
    print(" ".join(f"{v:3d}" for v in ligne))

# 3. Les mêmes nombres, redessinés : plus la valeur est grande, plus le caractère est dense
print()
print("3. Les mêmes nombres redessinés avec des caractères : on reconnaît le 7")
nuances = " .:-=+*#%@"
for ligne in image:
    print("".join(nuances[v * (len(nuances) - 1) // 255] * 2 for v in ligne))

# 4. Un pixel : ligne 2, colonne 4 (on compte à partir de 0)
print()
print("4. Valeur du pixel ligne 2, colonne 4 (image[1][3], on compte à partir de 0) :", image[1][3])

# 5. La valeur la plus claire et la plus sombre
valeurs = [v for ligne in image for v in ligne]
print("5. Pixel le plus clair :", max(valeurs), "| pixel le plus sombre :", min(valeurs))
