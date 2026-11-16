# Épisode 4 — Comment un réseau de neurones utilise les tenseurs

Un réseau de deux couches 64 → 16 → 10, soit 1 210 poids rangés dans des tenseurs. On suit la forme du tenseur couche après couche. Avec des poids au hasard, le réseau se trompe ; avec les poids entraînés (`reseau_entraine.pt`), il reconnaît le 7.

Les poids entraînés sont fournis. Pour les refaire vous-même (quelques secondes) :

```bash
python entrainer.py
```

```bash
python reseau.py
```

La sortie attendue est dans [resultats_attendus.txt](resultats_attendus.txt).
