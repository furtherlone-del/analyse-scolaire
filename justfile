app: generate analyse

# Lance toute l'application (génération + analyses)
s: app

# Génère le jeu de données uniquement
generate:
    python3 app.py --generate-only

# Exécute les 4 analyses (nécessite students_data.csv)
analyse:
    python3 app.py

# Réinitialise les sorties (output/ et data/generated/)
clean:
    rm -rf output/ data/generated/

# Nettoie + regénère tout
reset: clean generate analyse
