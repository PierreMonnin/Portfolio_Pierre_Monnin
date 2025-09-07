# Portfolio — Pierre Monnin

Projets classés par typologie.

## Actuariat
- [Dossier](./actuariat)

## Data science
- [Dossier](./data-science)

## Théorique
- [Dossier](./theorique)
---

## Conventions
- Noms de projets en **minuscules**, séparés par des `-` : `pricing-glm`, `loss-triangles`, `cv-nlp`…
- Arborescence type (ML) : `src/`, `notebooks/`, `data/`, `tests/`.
- Le dossier `data/` est ignoré par Git (sauf un `README.md` descriptif).

## Démarrer un projet Python (type ML)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\main.py

