# Rapport — Thème D : Établissement scolaire secondaire

**Groupe :** INF232_TP_GROUPE31 — **Chef de groupe :** VAMI NEGUEM YVO FREED
**Langage :** Python 3 — NumPy, Pandas, SciPy, Matplotlib, Scikit-learn

*Le choix de Python et de son écosystème scientifique (NumPy, Pandas, SciPy, Scikit-learn, Matplotlib) est motivé par la présence d'outils intégrés et performants pour l'analyse de données et la modélisation statistique, garantissant également la lisibilité du code.*

---

## 1. Génération des données

### 1.1 De la chaîne de caractères à la graine numérique

Le mécanisme de génération est l'élément obligatoire qui garantit l'unicité et la reproductibilité des données de chaque groupe. Le point de départ est le nom complet du chef de groupe.

Notre algorithme suit le principe de l'annexe du sujet : **prénoms avant nom**, en majuscules, sans accents ni espaces.

```
Entrée : "VAMI NEGUEM YVO FREED"
  → décomposition NFD (isole les diacritiques)
  → suppression des accents (catégorie Unicode "Mn")
  → mise en majuscules → "VAMI NEGUEM YVO FREED"
  → découpage en mots : ["VAMI", "NEGUEM", "YVO", "FREED"]
  → moitié gauche = NOM ("VAMI" + "NEGUEM")
  → moitié droite = prénoms ("YVO" + "FREED")
  → réorganisation : prénoms + nom
  → suppression de tout caractère non alphabétique
Sortie : "YVOFREEDVAMINEGUEM"
```

Cette chaîne normalisée est ensuite transformée en un entier reproductible via un **hachage polynomial** (inspiré de `String.hashCode()` de Java) :

```
h[0] = 0
h[i] = (h[i−1] × 31 + code_unicode(caractère[i])) mod (2³¹ − 1)
```

Ce choix n'est pas anodin : la base 31 est un nombre premier historiquement validé pour le hachage de chaînes (bonne diffusion, peu de collisions). Le modulo 2³¹−1 (nombre de Mersenne premier) garantit que le résultat tient dans un entier signé 32 bits, ce qui permet une compatibilité maximale avec les générateurs aléatoires standards.

**Graine obtenue : 1976473691.**

```python
# Code de génération de la graine à partir du nom
import re
import unicodedata

def normalize_name(full_name: str) -> str:
    decomposed = unicodedata.normalize("NFD", full_name.strip())
    without_accents = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    upper = without_accents.upper()
    words = re.findall(r"[A-Z]+", upper)
    if len(words) < 2:
        return "".join(words)
    n = len(words) // 2
    return "".join(words[n:]) + "".join(words[:n])

def name_to_seed(full_name: str) -> int:
    normalized = normalize_name(full_name)
    seed = 0
    for char in normalized:
        seed = (seed * 31 + ord(char)) % (2**31 - 1)
    return seed
```

### 1.2 Les données produites

Une fois la graine en main, nous initialisons un générateur pseudo-aléatoire NumPy (`default_rng(seed)`) et générons 150 profils d'élèves selon une loi normale bivariée. Les paramètres ont été choisis pour refléter une situation scolaire réaliste :

| Variable | Description | Plage |
|---|---|---|
| `average_grade` | Moyenne générale sur 20 (évaluation interne) | [4,0 ; 19,5] |
| `attendance` | Taux de présence annuel (pourcentage) | [55 ; 100] |
| `major` | Orientation conseillée par le conseil de classe | scientific / literary |

Le tirage bivarié introduit une corrélation volontaire de ρ = 0,58 entre la note et l'assiduité, ce qui reproduit une tendance naturelle (les élèves présents réussissent en moyenne mieux) sans que le lien soit mécanique. La variable `major` est ensuite attribuée via un score logistique bruité, garantissant un chevauchement réaliste entre les deux filières.

```
student_id  average_grade  attendance       major
     E001           12.97       100.0     literary
     E002            9.52        75.4     literary
     E003           10.40        73.5     literary
     E004           12.65        89.9     literary
     E005            9.30        69.3     literary
```

```python
# Code de génération du jeu de données synthétique
import numpy as np
import pandas as pd

def generate_dataset(seed: int, n_eleves=150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mean_vector = np.array([11.8, 87.5])
    cov = 0.58 * 3.1 * 11.0
    cov_matrix = np.array([[3.1**2, cov], [cov, 11.0**2]])
    samples = rng.multivariate_normal(mean_vector, cov_matrix, size=n_eleves)
    notes = np.clip(samples[:, 0], 4.0, 19.5)
    assiduites = np.clip(samples[:, 1], 55.0, 100.0)
    noise = rng.normal(0.0, 0.85, size=len(notes))
    score = 0.42 * notes + 0.028 * assiduites + noise
    majors = np.where(score >= 7.4, "scientific", "literary")
    return pd.DataFrame({
        "student_id": [f"E{i:03d}" for i in range(1, n_eleves + 1)],
        "average_grade": np.round(notes, 2),
        "attendance": np.round(assiduites, 1),
        "major": majors,
    })
```

150 élèves, c'est un effectif suffisant pour appliquer des méthodes statistiques (la règle empirique veut au moins 30 observations par groupe pour le théorème central limite), tout en restant plausible pour un lycée de taille moyenne.

---

## 2. Question 1 — Comment se répartissent les notes ?

Le proviseur veut un résumé clair de la distribution des notes, savoir s'il y a des cas extrêmes, et pouvoir le présenter simplement à son conseil pédagogique. Nous avons mobilisé deux familles d'outils : les indicateurs classiques de tendance centrale et de dispersion, et deux méthodes de détection des valeurs atypiques (IQR et Z-score).

### Résultats chiffrés

| Indicateur | Valeur |
|---|---|
| Effectif | 150 élèves |
| Moyenne | 11,76 /20 |
| Médiane | 11,61 /20 |
| Écart-type | 2,88 |
| Étendue | [4,0 ; 19,5] |
| Q1 – Q3 | [9,85 ; 13,67] |
| Écart interquartile (IQR) | 3,82 |

La distribution est relativement symétrique (moyenne et médiane proches), ce qui est cohérent avec une génération gaussienne.

### Élèves atypiques

Deux méthodes convergent vers les mêmes profils hors norme :

- **IQR (méthode de Tukey, k = 1,5)** : 3 élèves détectés (E023, E049, E067), avec des notes sous 4,11 ou au-dessus de 19,40.
- **Z-score (seuil |Z| > 2)** : 8 élèves, que nous pouvons nommer :
  - En difficulté : E023 (4,00, Z = −2,70), E067 (4,00, Z = −2,70), E095 (5,34, Z = −2,23), E097 (5,98, Z = −2,01)
  - En tête : E049 (19,50, Z = +2,69), E125 (17,64, Z = +2,04), E132 (17,82, Z = +2,11), E136 (17,71, Z = +2,07)

Ces résultats ne sortent pas de nulle part : les quatre meilleurs élèves sont tous en filière scientifique, les quatre plus faibles en littéraire. C'est le premier indice que note et orientation sont liées.

### Ce qu'il faut retenir

La grande majorité des élèves (94 %) se situe dans une zone « normale » entre 4 et 19,5/20. Les cas extrêmes existent mais restent rares et cohérents avec leur filière. Si je devais résumer cela au conseil pédagogique en une phrase : *« Neuf élèves sur dix ont une note comprise entre 7 et 15/20 ; seuls huit élèves sortent franchement du lot, et leur orientation actuelle correspond à leur niveau. »*

### Limites à garder en tête

Le Z-score suppose que les notes suivent une loi normale — c'est approximativement vrai ici mais pas parfait. L'IQR avec k = 1,5 n'a détecté que 3 élèves car nos bornes d'édition [4,0 ; 19,5] sont naturellement resserrées. Ces méthodes sont des signaux d'alerte, pas des diagnostics absolus.

---

## 3. Question 2 — L'assiduité permet-elle d'anticiper la note ?

Le proviseur aimerait savoir si, à l'avenir, on pourrait estimer la note d'un élève à partir de sa seule assiduité (par exemple pour les nouveaux arrivants). Avant de répondre, vérifions déjà si le lien existe.

### Normalité d'abord

Nous avons soumis les deux variables au test de Shapiro-Wilk :

- `attendance` : W = 0,941, p < 0,001 → **non normale**
- `average_grade` : W = 0,994, p = 0,795 → **normale**

Cette non-normalité de l'assiduité nous oriente naturellement vers le coefficient de Spearman (qui n'exige pas de normalité) comme mesure principale, en complément du coefficient de Pearson.

### Corrélation : un lien modéré à fort

Le coefficient de Spearman vaut **ρ = 0,643** (p < 0,001). C'est un lien positif et statistiquement très significatif : en moyenne, plus un élève est assidu, meilleure est sa note. Le coefficient de Pearson (r = 0,631) confirme la tendance.

En langage courant : si l'assiduité était un parfait prédicteur de la note, on aurait ρ = 1. Si elle n'avait aucun lien, on aurait ρ = 0. À 0,64 (0,6425), on est dans une zone « modérée à forte » — le lien existe mais il est loin d'être absolu.

### Régression : peut-on estimer la note ?

Nous avons ajusté une droite de régression linéaire :

**Note estimée = 0,1832 × assiduité − 4,2670**

- **R² = 39,7 %** — l'assiduité explique moins de la moitié de la variance des notes.
- **RMSE = 2,24 points** — l'erreur moyenne de prédiction est de plus de deux points sur 20.

Si le proviseur veut estimer la note d'un élève à 80 % d'assiduité : **10,39/20** en moyenne. Mais l'intervalle de prédiction à 95 % nous dit que la vraie note a 95 % de chances de se situer entre **5,94 et 14,84**. Une fourchette de près de 9 points, c'est trop large pour une décision individuelle.

### Quand l'estimation devient-elle trop incertaine ?

La demi-largeur minimale de l'intervalle de prédiction est de **±4,44 points** (au centre du nuage). En dehors de la zone d'assiduité comprise entre 68,1 % et 100 %, cette incertitude s'accroît encore. Concrètement, pour un élève avec une assiduité inférieure à 72,6 %, la marge d'erreur dépasse ce qui est raisonnable pour une utilisation pédagogique sérieuse (les points de comparaison devenant trop rares).

### Des contre-exemples qui en disent long

Nous avons cherché les élèves qui contredisent franchement la tendance :

- **Élèves assidus mais sous-performants** (assiduité ≥ médiane, note bien inférieure à la prédiction) : leur présence assidue n'a pas suffi à garantir le résultat.
- **Élèves peu assidus mais performants** (assiduité < médiane, note bien supérieure à la prédiction) : ils réussissent malgré une présence irrégulière.

Ces profils sont importants : ils nous rappellent que l'assiduité n'est qu'un facteur parmi d'autres. Un élève peut être présent tous les jours sans pour autant maîtriser les notions, et inversement.

### Réponse au proviseur

Oui, le lien entre assiduité et note existe et il est significatif. Non, on ne peut pas utiliser l'assiduité seule pour estimer fiablement la note d'un élève en particulier — la marge d'erreur est trop grande. L'assiduité peut donner une tendance générale, mais pas une prédiction individuelle utilisable en conseil de classe.

---

## 4. Question 3 — Y a-t-il des profils types d'élèves ?

Indépendamment de l'orientation déjà attribuée, le proviseur veut savoir si les données révèlent naturellement des groupes d'élèves aux profils proches. Nous avons utilisé l'algorithme des **K-Means**, après avoir normalisé les variables (StandardScaler) pour que la note et l'assiduité pèsent également dans le calcul des distances.

### Combien de groupes ?

Deux méthodes indépendantes convergent :

- La **méthode du coude** (inertie intra-cluster) montre une cassure nette à k = 2.
- Le **score de silhouette** (compacité + séparabilité) est maximal à k = 2 (0,476).

Nous retenons donc **deux profils types**.

### Le portrait de chaque groupe

**Groupe 1 — Les assidus performants (55,3 % de l'effectif)**

| Caractéristique | Valeur |
|---|---|
| Note moyenne | 13,66 /20 |
| Assiduité | 94,38 % |
| Orientation dominante | Scientifique |

Ces élèves cumulent travail et résultats. Leur assiduité est quasi parfaite et leurs notes sont solides. C'est le profil type de l'élève scientifique qui suit sérieusement sa scolarité.

**Groupe 2 — Les élèves en difficulté (44,7 % de l'effectif)**

| Caractéristique | Valeur |
|---|---|
| Note moyenne | 9,40 /20 |
| Assiduité | 78,93 % |
| Orientation dominante | Littéraire |

Ce groupe présente des notes plus fragiles et une assiduité irrégulière. Tous ne sont pas en échec, mais la moyenne de 9,4/20 indique une fragilité globale.

### Ce que ça recoupe avec la Question 1

Les 8 élèves détectés comme atypiques (Z-score) se répartissent de façon logique dans les deux groupes : les 4 meilleurs dans le groupe performant, les 4 plus faibles dans le groupe en difficulté. C'est un signe de cohérence interne de l'analyse.

### Ce que ça signifie pour le proviseur

Il existe bien deux grands profils d'élèves dans l'établissement, et ils correspondent en grande partie à l'orientation scientifique/littéraire — mais pas totalement. Environ un quart des élèves du groupe « performant » est en filière littéraire, et inversement. Autrement dit, l'orientation ne se résume pas au niveau : elle dépend aussi de choix personnels, de goûts disciplinaires que les données chiffrées ne capturent pas.

---

## 5. Question 4 — Peut-on prédire l'orientation automatiquement ?

Le proviseur voudrait qu'à l'avenir, un système suggère automatiquement une orientation (scientifique ou littéraire) pour un nouvel élève, avant même la délibération du conseil de classe, à partir de sa note et de son assiduité. Nous avons testé deux approches complémentaires.

### Les modèles comparés

- **Régression Logistique** : un modèle linéaire, simple, interprétable, qui donne une probabilité.
- **Arbre de Décision** : un modèle non linéaire, qui peut capturer des interactions complexes mais risque le surapprentissage.

Les données ont été séparées en 70 % d'entraînement et 30 % de test (45 élèves), avec stratification pour préserver la proportion des deux filières.

### Performances

| Métrique | Régression Logistique | Arbre de Décision |
|---|---|---|
| **Accuracy** | **73,33 %** | 71,11 % |
| Précision (scientifique) | 67,86 % | 65,52 % |
| Rappel (scientifique) | 86,36 % | 86,36 % |

**Modèle retenu : Régression Logistique** (73,3 % de bonnes réponses). L'arbre de décision n'est pas ridicule mais il est moins stable.

### Matrice de confusion (Régression Logistique)

| | Prédit Littéraire | Prédit Scientifique |
|---|---|---|
| **Réel Littéraire** | 14 (vrais négatifs) | 9 (faux positifs) |
| **Réel Scientifique** | 3 (faux négatifs) | 19 (vrais positifs) |

Sur 45 élèves testés, 12 sont mal classés (26,7 %). C'est mieux qu'un tirage au sort, mais insuffisant pour une utilisation autonome.

### Analyse détaillée des erreurs

- **9 faux positifs** — des élèves littéraires que le modèle envoie en scientifique. C'est le risque le plus problématique : orienter un élève vers une filière trop exigeante pour lui.
- **3 faux négatifs** — des élèves scientifiques que le modèle oriente en littéraire. Moins nombreux, mais le préjudice est réel : freiner un élève prometteur.
- **Biais** : le modèle est légèrement « optimiste », il surestime un peu le potentiel des élèves littéraires.

Toutes les erreurs concernent des élèves dont les caractéristiques se situent dans la **zone de recouvrement** des deux filières (notes autour de 10-12/20, assiduité entre 75 et 88 %). Autrement dit, là où le modèle se trompe, c'est exactement là où le conseil de classe hésite légitimement. Pas de hasard.

### Lien avec le reste de l'analyse

La frontière de décision du modèle prolonge naturellement la droite de régression de la Question 2 et recoupe les deux clusters de la Question 3. C'est une bonne nouvelle : cela signifie que les quatre analyses, bien qu'indépendantes dans leur méthode, racontent la même histoire. La cohérence interne renforce la confiance dans les résultats.

### Risques pour l'établissement

Si le proviseur utilisait ce système sans précaution, voici ce qui pourrait arriver :

- Un élève littéraire orienté vers la filière scientifique pourrait se retrouver en difficulté, perdre confiance, voire décrocher.
- Un élève scientifique orienté vers la filière littéraire pourrait s'ennuyer et ne pas développer son potentiel.
- Plus subtil : les deux variables (note et assiduité) ne capturent pas la motivation, les aptitudes disciplinaires spécifiques, ni le projet personnel de l'élève — autant d'éléments que seul le dialogue en conseil de classe peut apprécier.

### Préconisations concrètes

1. **Ne jamais utiliser la prédiction comme unique critère** d'orientation. Le conseil de classe conserve seul la responsabilité de la décision finale.
2. **Examiner manuellement les cas de désaccord** entre le modèle et l'avis des enseignants : ce sont eux qui révèlent les situations les plus complexes.
3. **Réévaluer le modèle chaque année** sur les nouvelles cohortes : les relations statistiques peuvent évoluer avec le temps, les programmes, les équipes pédagogiques.
4. **Ajouter d'autres variables à l'avenir** : notes par discipline, appréciations des professeurs, tests de positionnement — plus le modèle aura d'informations, meilleure sera sa fiabilité.

### Conclusion sur la Question 4

Le système atteint 73 % de bonnes réponses, ce qui en fait un outil d'aide à la décision utilisable — à condition de ne pas lui faire confiance aveuglément. La marge d'erreur de 27 % est trop élevée pour une utilisation autonome. Je le vois plutôt comme un second regard objectif qui peut aider le conseil pédagogique à identifier les cas limites et à concentrer sa discussion là où c'est le plus nécessaire.

---

## 6. Synthèse — Ce que tout cela raconte

Ces quatre questions ne sont pas indépendantes. Elles forment un parcours cohérent qui, étape par étape, construit une compréhension de plus en plus fine des données :

1. **Q1** a posé le décor : qui sont les élèves, comment se répartissent leurs notes, y a-t-il des cas particuliers ?
2. **Q2** a établi le lien entre les deux variables mesurées : l'assiduité influence la note, mais pas suffisamment pour servir de prédicteur fiable.
3. **Q3** a révélé la structure cachée : il y a deux grands profils d'élèves dans l'établissement, et ils ne se superposent pas parfaitement à l'orientation administrative.
4. **Q4** a tenté d'exploiter ces informations pour prédire l'orientation, avec une exactitude de 73 % — utile comme outil d'aide, dangereux comme décideur.

À chaque étape, les résultats se répondent : les élèves atypiques de Q1 se retrouvent naturellement dans les clusters de Q3, la relation linéaire de Q2 se prolonge dans la frontière de décision de Q4. Cette cohérence globale est le signe d'une analyse statistique robuste.

### Ce que le proviseur doit retenir (en trois phrases)

Les données de votre établissement révèlent une structure claire : les élèves les plus assidus obtiennent les meilleures notes et sont majoritairement en filière scientifique. Un système automatique peut reproduire cette logique avec environ 73 % de réussite, mais il se trompe encore une fois sur quatre — et ses erreurs tombent exactement sur les cas les plus difficiles, ceux où l'humain est le plus nécessaire. Utilisez cet outil comme un éclairage complémentaire, pas comme un remplaçant du conseil de classe.

---

*Graine : 1976473691 — Groupe INF232_TP_GROUPE31 — Juillet 2026*
