# INF232 — Thème D : Établissement scolaire secondaire

Analyse statistique complète des performances d'élèves de terminale : de la
génération déterministe des données à la classification supervisée, en passant
par l'analyse univariée, bivariée et non supervisée.

## Structure du projet

```
INF232_TP_GROUPEXX/
├── app.py                     # Point d'entrée — génération + 4 analyses
├── config/config.yaml         # Paramètres (génération, orientation, analyse)
├── data/
│   ├── raw/                   # Copie de sauvegarde des données brutes
│   └── generated/             # Jeu de données final (students_data.csv)
├── src/
│   ├── generator/
│   │   ├── hasher.py          # Normalisation du nom → graine entière
│   │   └── data_generator.py  # Génération bivariée des profils élèves
│   ├── analysis/
│   │   ├── univariat.py       # Question 1 — Stats descriptives + atypiques
│   │   ├── bivariat.py        # Question 2 — Corrélation + régression
│   │   ├── clustering.py      # Question 3 — K-Means (profils types)
│   │   └── classification.py  # Question 4 — Prédiction de la filière
│   └── utils/helpers.py       # Chargement config et données
├── output/                    # Graphiques et résultats JSON (auto-généré)
├── notebooks/                 # Cahiers Jupyter d'expérimentation
├── docs/                      # Rapport et documentation
├── justfile                   # Raccourcis (just s, just reset, …)
└── requirements.txt
```

## Utilisation

```bash
# Lancement complet (génération + analyses)
python3 app.py

# Génération seule (sans analyses)
python3 app.py --generate-only

# Ou via just (si installé)
just s
just generate
just reset        # nettoie + regénère
```


## Dépendances

```bash
pip install -r requirements.txt
```
