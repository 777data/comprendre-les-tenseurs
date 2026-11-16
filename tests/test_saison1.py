"""Exécute chaque script de la saison 1 et compare sa sortie aux résultats montrés dans les vidéos.

Lancer depuis la racine du dépôt :  pytest
"""
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
SAISON1 = RACINE / "saison-1-les-tenseurs"

SCRIPTS = [
    "episode-1-image-en-nombres/image_en_nombres.py",
    "episode-2-premier-tenseur/premier_tenseur.py",
    "episode-3-dimensions/dimensions.py",
    "episode-4-reseau-de-neurones/reseau.py",
    "episode-5-apprentissage/apprentissage.py",
]


def lignes(texte):
    return [l.rstrip() for l in texte.strip().splitlines()]


@pytest.mark.parametrize("script", SCRIPTS)
def test_sortie_identique_a_la_video(script):
    chemin = SAISON1 / script
    resultat = subprocess.run(
        [sys.executable, chemin.name], cwd=chemin.parent,
        capture_output=True, text=True, timeout=600,
    )
    assert resultat.returncode == 0, resultat.stderr
    attendu = (chemin.parent / "resultats_attendus.txt").read_text()
    assert lignes(resultat.stdout) == lignes(attendu)
