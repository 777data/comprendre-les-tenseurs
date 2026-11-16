# Comprendre les tenseurs en intelligence artificielle

Le code de la série vidéo, épisode par épisode. Chaque script est celui montré à l'écran : vous pouvez le lancer et retrouver exactement les résultats de la vidéo.

Le fil rouge de toute la série est un même chiffre 7 manuscrit, en 8 × 8 pixels. Le but final est de comprendre comment fonctionne la génération d'images par IA. On commence par les fondamentaux.

## Saisons

| Saison | Thème | État |
|---|---|---|
| [Saison 1](saison-1-les-tenseurs/) | Les tenseurs : de l'image au réseau qui apprend | ✅ 5 épisodes |
| [Saison 2](saison-2-convolution/) | Les réseaux qui voient (convolution) | À venir |
| [Saison 3](saison-3-espace-latent/) | Compresser et représenter (espace latent) | À venir |
| [Saison 4](saison-4-texte/) | Comprendre le texte (pour guider l'image) | À venir |
| [Saison 5](saison-5-diffusion/) | La génération d'images par diffusion | À venir |

## Installation

Il faut Python 3.10 ou plus récent.

> **Mac Intel** : PyTorch s'arrête à la version 2.2, qui n'existe que pour Python 3.12 au maximum. Si `pip` répond « No matching distribution found for torch », créez l'environnement avec Python 3.12 : `python3.12 -m venv .venv` (Python 3.12 s'installe par exemple avec `brew install python@3.12`).

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer un épisode

Chaque script se lance depuis n'importe quel dossier :

```bash
python saison-1-les-tenseurs/episode-2-premier-tenseur/premier_tenseur.py
```

## Tout vérifier d'un coup

```bash
pytest
```

Le test lance chaque script et compare sa sortie au fichier `resultats_attendus.txt` de l'épisode, c'est-à-dire aux résultats affichés dans la vidéo. L'épisode 5 entraîne un réseau pour de vrai, ce qui prend quelques secondes.
