# Justification méthodologique — Génération des données (Thème D)

## Paragraphe pour le rapport final

Pour modéliser la situation d'un établissement secondaire camerounais en classe de terminale, nous avons retenu deux indicateurs scolaires complémentaires et une variable cible d'orientation. La **note moyenne sur 20** synthétise le résultat d'une évaluation interne couvrant l'ensemble des disciplines ; elle constitue l'indicateur principal de performance académique mobilisé par le proviseur dans sa première question. L'**assiduité**, exprimée en pourcentage de présence effective sur l'année scolaire, capture un comportement d'investissement distinct de la pure aptitude : un élève peut être assidu sans exceller, ou inversement brillant mais irrégulier. Ces deux mesures sont générées conjointement à partir d'une **loi normale bivariée** de moyennes respectives μ_note = 11,8 et μ_assiduité = 87,5 %, d'écarts-types σ_note = 3,1 et σ_assiduité = 11,0 points, et d'une **corrélation de Pearson ρ = 0,58**. Ce niveau de corrélation positive, inférieur à 0,80, traduit un lien réel mais imparfait entre présence et résultats — condition nécessaire pour que l'estimation d'une variable à partir de l'autre (question 2) soit utile tout en restant entachée d'incertitude. Les valeurs sont ensuite tronquées dans des bornes plausibles ([4 ; 19,5] pour la note, [55 ; 100] pour l'assiduité) afin d'exclure les cas extrêmes non représentatifs d'une population de terminale. La variable cible **filière** (scientifique ou littéraire) est dérivée d'un score d'orientation combinant linéairement note et assiduité, augmenté d'un bruit gaussien (σ = 0,85), puis comparé à un seuil fixe ; ce mécanisme reproduit la logique du conseil de classe tout en introduisant un recouvrement réaliste entre les deux filières, indispensable pour étudier la classification supervisée (question 4) sans séparabilité parfaite. L'échantillon de **150 élèves** correspond à environ cinq classes de terminale, taille suffisante pour appliquer des méthodes descriptives, des tests de corrélation, une classification non supervisée (k-means) et un modèle prédictif tout en restant cohérent avec l'effectif d'un lycée moyen. Enfin, la **graine entière** est obtenue par hachage polynomial déterministe (base 31) du nom normalisé du chef de groupe « VAMI NEGUEM YVO FREED » → `VAMINEGUEMYVOFREED`, alimentant le générateur `numpy.random.Generator` : ainsi, l'ensemble du processus est reproductible à l'identique à chaque exécution.

---

## Algorithme de génération de la graine

### Étape 1 — Normalisation

```
"VAMI NEGUEM YVO FREED"
    → suppression accents (NFD + filtre Mn)
    → majuscules
    → moitié gauche = NOM ("VAMI NEGUEM"), moitié droite = prénoms ("YVO FREED")
    → réorganisation : prénoms + nom
    → suppression non-alphabétique
    → "YVOFREEDVAMINEGUEM"
```

### Étape 2 — Hachage polynomial

Pour chaque caractère `c` de la chaîne normalisée :

```
h ← (h × 31 + ord(c)) mod (2³¹ − 1)
```

La graine finale est l'entier `h` obtenu après le dernier caractère.

**Justification du choix** : contrairement à `hash()` Python (salé et variable selon les sessions), ce hachage est purement arithmétique, documenté et identique sur toute machine.

---

## Paramètres statistiques retenus

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| n (effectif) | 150 | ~5 classes × 30 élèves ; n > 30 par filière pour classification |
| μ_note | 11,8 / 20 | Proche de la moyenne nationale observée en terminale |
| σ_note | 3,1 | Dispersion réaliste (majorité entre 8 et 16) |
| μ_assiduité | 87,5 % | Présence élevée mais pas parfaite en lycée |
| σ_assiduité | 11,0 | Écart-type permettant des profils très assidus ou fragiles |
| ρ(note, assiduité) | 0,58 | Corrélation modérée-forte, compatible avec régression |
| Proportion scientifique | ~55 % | Légère majorité scientifique, courante dans les lycées |
| Bruit orientation | σ = 0,85 | Recouvrement filières pour classification non triviale |

---

## Matrice de covariance

```
Σ = | σ²_note     ρ·σ_note·σ_assid  |
    | ρ·σ_note·σ_assid    σ²_assid  |

  = | 9,61    19,66 |
    | 19,66  121,00 |
```

---

## Logique d'attribution de la filière

```
score_i = 0,42 × note_i + 0,028 × assiduité_i + ε_i,   ε_i ~ N(0 ; 0,85²)

filière_i = "scientifique"  si score_i ≥ 7,4
          = "litteraire"    sinon
```

Les coefficients et le seuil ont été calibrés pour obtenir environ 55 % d'élèves en filière scientifique tout en conservant un lien interprétable avec les deux indicateurs.
