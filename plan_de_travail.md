# Bobathon ESILV — Plan complet de réalisation avec Bob IDE

## 1. Présentation du projet

Ce projet correspond à une compétition de machine learning consacrée à la prédiction du score moteur réel OFF chez des patients atteints de la maladie de Parkinson.

Les données représentent plusieurs visites médicales réalisées pour différents patients. Pour chaque visite, plusieurs informations peuvent être disponibles :

- l’âge du patient ;
- l’âge au diagnostic ;
- le sexe ;
- la cohorte ;
- des informations génétiques ;
- la dose de traitement ;
- le temps écoulé depuis la prise du traitement ;
- un score moteur ON ;
- un score moteur OFF ;
- l’identifiant de l’évaluateur médical.

Certaines valeurs sont absentes. Ces valeurs manquantes peuvent être informatives et ne doivent pas être supprimées automatiquement.

L’objectif de la compétition est de construire un modèle capable de prédire la colonne `target`, qui correspond à une estimation corrigée du score moteur OFF réel.

La réalisation complète du projet doit être effectuée uniquement avec :

- Bob IDE ;
- l’IA intégrée à Bob ;
- les fichiers présents dans le dépôt ;
- Skore pour l’évaluation et le suivi des expériences ;
- Kaggle pour les soumissions.

Ce document sert de feuille de route complète. Il part du principe que la réalisation du projet commence maintenant, sans utiliser de code, de résultat ou d’expérience précédemment créé.

> L’environnement de développement, les dépendances Python, Bob IDE, Skore Hub et les configurations nécessaires sont déjà opérationnels. Ce README ne décrit donc pas l’installation ni la configuration de l’environnement.

---

## 2. Objectif final

L’objectif final est de produire un fichier de prédictions pour le jeu de test.

Le fichier doit contenir deux colonnes :

```text
Index,target
```

Exemple :

```csv
Index,target
66,37.4
67,38.1
68,41.8
```

Le fichier final doit :

1. contenir une prédiction pour chaque ligne de `X_test.csv` ;
2. conserver les mêmes valeurs de `Index` que `X_test.csv` ;
3. conserver le même ordre que `X_test.csv` ;
4. ne contenir aucune valeur manquante ;
5. être enregistré dans un dossier `submissions/` ;
6. être évalué avec Skore ;
7. être associé à un rapport Skore Hub ;
8. être soumis sur Kaggle ;
9. contenir une description Kaggle avec l’URL du rapport Skore.

Le projet doit également permettre d’expliquer :

- comment les données ont été préparées ;
- quelles variables ont été utilisées ;
- quels modèles ont été testés ;
- comment les modèles ont été évalués ;
- pourquoi le modèle final a été choisi ;
- quelles sont ses limites.

---

## 3. Contexte scientifique

La maladie de Parkinson peut provoquer plusieurs symptômes moteurs :

- tremblements ;
- rigidité ;
- lenteur des mouvements ;
- difficultés à initier certains mouvements.

Les patients peuvent prendre des traitements destinés à améliorer leurs symptômes. L’état moteur peut alors varier selon l’effet du médicament.

On distingue notamment :

- l’état **ON**, lorsque le traitement agit ;
- l’état **OFF**, lorsque l’effet du traitement est faible ou absent.

Le score OFF est utile pour estimer la sévérité réelle de la maladie. Cependant, la mesure observée peut être influencée par :

- le moment de la prise du médicament ;
- le délai entre la prise et l’évaluation ;
- la dose reçue ;
- l’absence d’un examen OFF ;
- la variabilité des évaluateurs ;
- la progression de la maladie ;
- les erreurs de mesure ;
- les données manquantes.

Le modèle doit donc apprendre à estimer un score OFF corrigé à partir des informations disponibles lors d’une visite.

---

## 4. Données disponibles

Les données sont placées dans le dossier `data/`.

```text
data/
├── X_train.csv
├── y_train.csv
├── X_test.csv
└── sample_submission.csv
```

### 4.1 `X_train.csv`

Ce fichier contient les variables explicatives des visites d’entraînement.

Les colonnes attendues sont :

| Colonne | Description |
|---|---|
| `Index` | identifiant unique de la ligne |
| `patient_id` | identifiant du patient |
| `cohort` | cohorte d’origine |
| `sexM` | indicateur numérique du sexe |
| `gene` | information génétique |
| `age_at_diagnosis` | âge au moment du diagnostic |
| `age` | âge lors de la visite |
| `ledd` | dose équivalente quotidienne de lévodopa |
| `time_since_intake_on` | délai depuis la prise pour la mesure ON |
| `time_since_intake_off` | délai depuis la prise pour la mesure OFF |
| `rater_id` | identifiant de l’évaluateur |
| `on` | score moteur ON observé |
| `off` | score moteur OFF observé |

### 4.2 `y_train.csv`

Ce fichier contient :

```text
Index,target
```

La colonne `target` est la variable à prédire.

### 4.3 `X_test.csv`

Ce fichier contient les variables explicatives des visites pour lesquelles les prédictions doivent être produites.

La colonne `target` n’est pas présente dans ce fichier.

### 4.4 `sample_submission.csv`

Ce fichier indique le format attendu par Kaggle :

```text
Index,target
```

Il doit être utilisé comme référence pour contrôler les fichiers de soumission.

---

## 5. Principe général de l’architecture

Le projet doit être organisé autour de notebooks thématiques.

Les notebooks permettent de :

- charger et explorer les données ;
- afficher des tableaux et graphiques ;
- tester des modèles ;
- comparer les performances ;
- analyser les erreurs ;
- documenter les conclusions.

Les notebooks doivent être exécutés dans Bob IDE.

La structure générale du projet est :

```text
Données CSV
    |
    v
Chargement des données
    |
    v
Fusion entre X_train et y_train
    |
    v
Exploration des données
    |
    v
Baseline
    |
    v
Premiers modèles
    |
    v
Validation par patient
    |
    v
Modèles avancés
    |
    v
Feature engineering
    |
    v
Analyse des erreurs
    |
    v
Sélection du meilleur modèle
    |
    v
Génération de la soumission
    |
    v
Skore Hub
    |
    v
Kaggle
```

---

## 6. Structure des notebooks du projet

La structure suivante doit être mise en place progressivement depuis Bob IDE :

```text
bobathon-esilv/
├── data/
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_test.csv
│   └── sample_submission.csv
├── docs/
│   ├── CONTEXT.md
│   ├── DAY1.md
│   ├── DAY2.md
│   └── SETUP.md
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_ridge.ipynb
│   ├── 04_grouped_validation.ipynb
│   ├── 05_hist_gradient_boosting.ipynb
│   ├── 06_tabular_pipeline.ipynb
│   ├── 07_feature_engineering.ipynb
│   ├── 08_error_analysis.ipynb
│   ├── 09_final_model.ipynb
│   ├── 10_improved_model.ipynb
│   ├── 11_lgbm_advanced.ipynb
│   ├── 12_optimized.ipynb
│   ├── 13_xgb_ensemble.ipynb
│   ├── 14_ensemble_l2.ipynb
│   └── 15_multi_models.ipynb
├── submissions/
├── README.md
└── autres fichiers déjà présents dans le dépôt
```

Les notebooks sont exécutés dans cet ordre. Les notebooks 01 à 09
correspondent au parcours initial ; les notebooks 10 à 15 prolongent
l'expérimentation après des scores insuffisants du premier modèle final.

Aucun notebook ne doit supposer qu’un autre notebook a déjà été exécuté en mémoire.

Chaque notebook doit pouvoir être relancé depuis le début.

---

# 7. Règles générales concernant les notebooks

Chaque notebook doit être organisé comme suit :

```text
1. Titre
2. Objectif
3. Imports
4. Chargement des données
5. Vérifications
6. Préparation des données
7. Analyse ou modélisation
8. Résultats
9. Interprétation
10. Conclusion
```

Chaque notebook doit commencer par une cellule Markdown expliquant :

- le but du notebook ;
- les fichiers utilisés ;
- les variables étudiées ;
- le résultat attendu ;
- le notebook suivant.

Avant de considérer un notebook comme terminé :

1. redémarrer le noyau ;
2. exécuter toutes les cellules dans l’ordre ;
3. vérifier qu’aucune erreur n’apparaît ;
4. vérifier que les résultats sont cohérents ;
5. supprimer les cellules inutiles ;
6. conserver une conclusion écrite.

---

# 8. Phase 1 — Créer le notebook d’exploration

## Notebook

```text
notebooks/01_exploration.ipynb
```

## Objectif

Ce notebook doit charger les fichiers, fusionner les données d’entraînement et comprendre leur structure.

## Consigne à donner à Bob

```text
Crée le notebook notebooks/01_exploration.ipynb.

Le notebook doit :
- charger data/X_train.csv ;
- charger data/y_train.csv ;
- charger data/X_test.csv ;
- fusionner X_train et y_train avec la colonne Index ;
- vérifier les dimensions ;
- vérifier l’unicité de Index ;
- afficher les colonnes ;
- afficher les types ;
- compter les patients ;
- analyser le nombre de visites par patient ;
- analyser les valeurs manquantes ;
- analyser la distribution de target ;
- créer des visualisations ;
- expliquer les risques de fuite de données.

Le notebook doit être autonome, organisé en sections Markdown et cellules de code.
Il ne doit utiliser aucun fichier Python créé précédemment.
```

## Chargement recommandé

```python
from pathlib import Path

import pandas as pd


DATA_DIR = Path("../data")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

visits = X_train.merge(
    y_train,
    on="Index",
    how="inner",
    validate="one_to_one",
)
```

Si le notebook est exécuté depuis la racine du dépôt, utiliser plutôt :

```python
DATA_DIR = Path("data")
```

Le chemin choisi doit être cohérent avec le répertoire de travail de Bob.

## Vérifications obligatoires

```python
assert X_train["Index"].is_unique
assert y_train["Index"].is_unique
assert X_test["Index"].is_unique

assert set(y_train.columns) == {
    "Index",
    "target",
}

assert len(visits) == len(X_train)
assert len(visits) == len(y_train)

assert visits["target"].notna().all()
```

## Analyses à réaliser

Le notebook doit analyser :

- les dimensions des fichiers ;
- les colonnes disponibles ;
- les types de données ;
- les valeurs manquantes ;
- le nombre de patients ;
- le nombre de visites par patient ;
- la distribution de `target` ;
- les valeurs minimales et maximales ;
- les catégories de `cohort` ;
- les catégories de `gene` ;
- les catégories de `rater_id` ;
- les relations entre les variables et `target`.

## Critères de validation

Cette phase est terminée si :

- la fusion fonctionne ;
- aucune ligne n’est perdue ;
- la cible est correctement associée à `Index` ;
- les valeurs manquantes sont documentées ;
- les risques de fuite sont expliqués ;
- les graphiques sont lisibles ;
- une conclusion est rédigée dans le notebook.

---

# 9. Phase 2 — Construire la baseline

## Notebook

```text
notebooks/02_baseline.ipynb
```

## Objectif

Construire un modèle de référence minimal.

La baseline doit prédire la moyenne de `target` pour toutes les lignes.

## Consigne à donner à Bob

```text
Crée notebooks/02_baseline.ipynb.

Le notebook doit :
- recharger les données de manière autonome ;
- fusionner X_train et y_train ;
- sélectionner des variables de départ ;
- construire un DummyRegressor(strategy="mean") ;
- l’évaluer avec skore.evaluate ;
- calculer la RMSE ;
- expliquer le rôle de la baseline ;
- envoyer le rapport à Skore Hub avec une clé unique ;
- générer un fichier dans submissions/.

Ne réutilise aucune variable d’un autre notebook.
```

## Modèle

```python
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(
    strategy="mean"
)
```

## Variables initiales

```python
feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X = visits[feature_cols]
y = visits["target"]
```

## Évaluation

```python
from skore import evaluate

report = evaluate(
    dummy,
    X,
    y,
)

rmse = report.metrics.rmse()

print(f"RMSE : {rmse:.6f}")
```

## Rapport Skore

Le rapport doit recevoir une clé unique, par exemple :

```text
01_dummy
```

L’URL affichée par Skore doit être conservée dans le notebook.

## Création d’une soumission

```python
from pathlib import Path

dummy.fit(X, y)

X_test_model = X_test[feature_cols].copy()

predictions = dummy.predict(
    X_test_model
)

submission = X_test[["Index"]].copy()
submission["target"] = predictions

Path("../submissions").mkdir(
    exist_ok=True
)

submission.to_csv(
    "../submissions/01_dummy.csv",
    index=False,
)
```

Le chemin doit être adapté au répertoire depuis lequel le notebook est exécuté.

## Vérifications

```python
assert list(submission.columns) == [
    "Index",
    "target",
]

assert len(submission) == len(X_test)

assert submission["Index"].equals(
    X_test["Index"]
)

assert submission["target"].notna().all()
```

## Critères de validation

La baseline est terminée si :

- elle est évaluée avec Skore ;
- le rapport est envoyé ;
- le fichier CSV est créé ;
- le format du fichier est valide ;
- le fichier est accepté par Kaggle ;
- la RMSE et le score Kaggle sont documentés.

---

# 10. Phase 3 — Construire un modèle Ridge

## Notebook

```text
notebooks/03_ridge.ipynb
```

## Objectif

Tester un premier modèle réel de régression.

## Consigne à donner à Bob

```text
Crée notebooks/03_ridge.ipynb.

Le notebook doit :
- recharger les données de manière autonome ;
- fusionner X_train et y_train ;
- sélectionner les variables numériques ;
- construire un pipeline SimpleImputer(strategy="median") + Ridge ;
- tester plusieurs valeurs de alpha ;
- comparer Ridge à la baseline ;
- utiliser skore.evaluate ;
- afficher un tableau des résultats ;
- indiquer la meilleure configuration ;
- envoyer le meilleur rapport à Skore Hub.

Ne dépends d’aucun autre notebook.
```

## Pipeline

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ridge = make_pipeline(
    SimpleImputer(
        strategy="median"
    ),
    Ridge(
        alpha=1.0
    ),
)
```

## Valeurs à tester

```python
alphas = [
    0.1,
    1.0,
    10.0,
    100.0,
]
```

## Évaluation

Chaque valeur doit être évaluée séparément.

```python
results = []

for alpha in alphas:
    model = make_pipeline(
        SimpleImputer(
            strategy="median"
        ),
        Ridge(
            alpha=alpha
        ),
    )

    report = evaluate(
        model,
        X,
        y,
    )

    results.append(
        {
            "alpha": alpha,
            "rmse": report.metrics.rmse(),
        }
    )

results_df = pd.DataFrame(results)

display(
    results_df.sort_values("rmse")
)
```

## Critères de validation

La phase est terminée si :

- Ridge fonctionne ;
- les valeurs manquantes sont traitées ;
- plusieurs valeurs de `alpha` sont comparées ;
- la baseline est utilisée comme référence ;
- la meilleure configuration est identifiée ;
- le résultat est expliqué.

---

# 11. Phase 4 — Validation par patient

## Notebook

```text
notebooks/04_grouped_validation.ipynb
```

## Objectif

Évaluer les modèles sans placer un même patient dans les ensembles d’entraînement et de validation.

## Pourquoi cette validation est nécessaire ?

Chaque patient possède plusieurs visites.

Une validation aléatoire par ligne peut placer certaines visites du même patient dans le train et d’autres dans la validation.

Le modèle peut alors bénéficier indirectement d’informations sur un patient qu’il est censé ne pas connaître.

La validation doit donc être groupée par `patient_id`.

## Consigne à donner à Bob

```text
Crée notebooks/04_grouped_validation.ipynb.

Le notebook doit :
- recharger les données de manière autonome ;
- construire X, y et groups ;
- utiliser GroupKFold(n_splits=5) ;
- vérifier qu’un patient n’est jamais présent dans les deux parties d’un fold ;
- comparer DummyRegressor et Ridge ;
- afficher la RMSE moyenne ;
- afficher la RMSE de chaque fold ;
- expliquer la différence avec une validation aléatoire.
```

## Création des groupes

```python
from sklearn.model_selection import GroupKFold

groups = visits["patient_id"]

cv_splits = list(
    GroupKFold(
        n_splits=5
    ).split(
        X,
        y,
        groups=groups,
    )
)
```

## Vérification des groupes

```python
for train_indices, valid_indices in cv_splits:
    train_patients = set(
        groups.iloc[train_indices]
    )

    valid_patients = set(
        groups.iloc[valid_indices]
    )

    assert train_patients.isdisjoint(
        valid_patients
    )
```

## Évaluation avec Skore

```python
report = evaluate(
    {
        "dummy": dummy,
        "ridge": ridge,
    },
    X,
    y,
    splitter=cv_splits,
)
```

## Critères de validation

Cette phase est terminée si :

- les patients sont utilisés comme groupes ;
- aucun patient n’est partagé dans un fold ;
- les mêmes folds servent à comparer tous les modèles ;
- la RMSE par fold est affichée ;
- la RMSE moyenne est affichée ;
- les conclusions sont documentées.

---

# 12. Phase 5 — Tester HistGradientBoosting

## Notebook

```text
notebooks/05_hist_gradient_boosting.ipynb
```

## Objectif

Tester un modèle non linéaire capable de mieux exploiter les interactions entre variables et certaines valeurs manquantes.

## Modèle

```python
from sklearn.ensemble import (
    HistGradientBoostingRegressor
)

hgbr = HistGradientBoostingRegressor(
    random_state=0
)
```

## Évaluation

Utiliser les mêmes folds que Ridge :

```python
report = evaluate(
    {
        "ridge": ridge,
        "hgbr": hgbr,
    },
    X,
    y,
    splitter=cv_splits,
)
```

## Analyses

Comparer :

- RMSE moyenne ;
- RMSE de chaque fold ;
- dispersion des scores ;
- temps d’entraînement ;
- temps de prédiction ;
- erreurs sur les lignes où `off` manque ;
- erreurs sur les lignes où `on` manque.

## Critères de validation

HGBR est intéressant si :

- il bat Ridge avec la même validation ;
- l’amélioration est stable ;
- aucun fold ne présente une dégradation excessive ;
- le modèle produit des prédictions valides ;
- le résultat est expliqué dans le notebook.

---

# 13. Phase 6 — Intégrer les variables catégorielles

## Notebook

```text
notebooks/06_tabular_pipeline.ipynb
```

## Objectif

Tester les colonnes catégorielles :

- `gene` ;
- `cohort` ;
- éventuellement `rater_id`.

## Consigne à donner à Bob

```text
Crée notebooks/06_tabular_pipeline.ipynb.

Le notebook doit :
- recharger les données ;
- supprimer Index, patient_id et target des variables explicatives ;
- conserver les variables numériques et catégorielles pertinentes ;
- utiliser skrub.tabular_pipeline("regressor") ;
- évaluer le modèle avec GroupKFold ;
- comparer les résultats au modèle numérique ;
- expliquer le traitement des types de colonnes.
```

## Préparation

```python
X_full = visits.drop(
    columns=[
        "Index",
        "patient_id",
        "target",
    ],
)
```

## Pipeline

```python
from skrub import tabular_pipeline

tabular_model = tabular_pipeline(
    "regressor"
)
```

## Évaluation

```python
report = evaluate(
    tabular_model,
    X_full,
    y,
    splitter=cv_splits,
)
```

## Points de vigilance

`Index` et `patient_id` doivent être supprimés par défaut.

`rater_id` doit être testé avec prudence. Il peut représenter un effet réel de l’évaluateur, mais aussi un biais artificiel.

## Critères de validation

La phase est terminée si :

- les variables catégorielles sont prises en compte ;
- les identifiants sont traités correctement ;
- la validation groupée est utilisée ;
- le modèle est comparé aux modèles précédents ;
- le notebook contient une conclusion.

---

# 14. Phase 7 — Feature engineering

## Notebook

```text
notebooks/07_feature_engineering.ipynb
```

## Objectif

Créer des variables susceptibles d’améliorer les performances.

Cette phase doit commencer uniquement après avoir obtenu des évaluations fiables avec les modèles de base.

## Feature 1 — Temps depuis le diagnostic

```python
visits["years_since_diagnosis"] = (
    visits["age"]
    - visits["age_at_diagnosis"]
)
```

## Feature 2 — Indicateurs de valeurs manquantes

```python
missing_columns = [
    "on",
    "off",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
]

for column in missing_columns:
    visits[f"{column}_missing"] = (
        visits[column]
        .isna()
        .astype(int)
    )
```

## Feature 3 — Différence entre OFF et ON

```python
visits["off_on_gap"] = (
    visits["off"]
    - visits["on"]
)
```

## Règles obligatoires

Toute nouvelle variable doit :

1. être calculée uniquement à partir des données disponibles ;
2. être créée de la même manière sur train et test ;
3. ne jamais utiliser `target` ;
4. être évaluée avec GroupKFold ;
5. être comparée au modèle sans cette variable ;
6. être conservée uniquement si le gain est stable.

---

# 15. Phase 8 — Analyse des erreurs

## Notebook

```text
notebooks/08_error_analysis.ipynb
```

## Objectif

Comprendre dans quelles situations le modèle commet ses plus grandes erreurs.

Créer un tableau contenant :

- `Index` ;
- `patient_id` ;
- `target` ;
- prédiction ;
- erreur signée ;
- erreur absolue ;
- `age` ;
- `cohort` ;
- `on` ;
- `off` ;
- `ledd` ;
- indicateurs de valeurs manquantes.

## Exemple

```python
errors = visits[
    [
        "Index",
        "patient_id",
        "target",
    ]
].copy()

errors["prediction"] = predictions

errors["error"] = (
    errors["prediction"]
    - errors["target"]
)

errors["absolute_error"] = (
    errors["error"]
    .abs()
)

errors = errors.sort_values(
    "absolute_error",
    ascending=False,
)
```

## Questions à étudier

- Le modèle se trompe-t-il davantage lorsque `off` est manquant ?
- Les scores élevés sont-ils sous-estimés ?
- Certaines cohortes sont-elles plus difficiles ?
- Les erreurs augmentent-elles avec l’âge ?
- Les valeurs de `ledd` améliorent-elles les prédictions ?
- Les délais de prise du traitement sont-ils utiles ?
- Certains profils de patients sont-ils systématiquement mal prédits ?

---

# 16. Phase 9 — Première sélection du modèle final

## Notebook

```text
notebooks/09_final_model.ipynb
```

## Objectif

Comparer les expériences initiales et sélectionner un premier modèle final.
Cette étape constitue un jalon intermédiaire : les scores obtenus ont conduit
à poursuivre les expérimentations dans les notebooks 10 à 15.

Le choix doit être basé sur :

- la RMSE GroupKFold ;
- la stabilité entre les folds ;
- la cohérence avec Kaggle ;
- la qualité des prédictions ;
- la simplicité du pipeline ;
- la reproductibilité ;
- l’absence de fuite de données.

## Tableau de comparaison

Créer un tableau contenant :

| Modèle | Variables | Validation | RMSE moyenne | Écart-type | Score Kaggle | Décision |
|---|---|---|---:|---:|---:|---|
| Dummy | baseline | GroupKFold | à compléter | à compléter | à compléter | référence |
| Ridge | numériques | GroupKFold | à compléter | à compléter | à compléter | à analyser |
| HGBR | numériques | GroupKFold | à compléter | à compléter | à compléter | à analyser |
| Tabular pipeline | mixtes | GroupKFold | à compléter | à compléter | à compléter | à analyser |
| Premier modèle final | sélectionné | GroupKFold | à compléter | à compléter | à compléter | jalon intermédiaire |

## Critères de sélection

Le modèle final doit :

- être meilleur que la baseline ;
- être évalué avec une validation par patient ;
- ne pas utiliser `target` comme variable ;
- ne pas dépendre d’un état mémoire de notebook ;
- pouvoir être réentraîné sur toutes les données ;
- produire une soumission conforme.

---

# 17. Phase 10 — Amélioration du modèle

## Notebook

```text
notebooks/10_improved_model.ipynb
```

## Objectif

Améliorer le meilleur modèle du notebook 09 en testant systématiquement les
hyperparamètres du pipeline HGBR/skrub : nombre d'itérations, taux
d'apprentissage, profondeur, taille minimale des feuilles et régularisation.
Le notebook teste également l'impact de la suppression de `rater_id`.

## Critères de validation

Cette phase est terminée si :

- plusieurs configurations sont comparées avec GroupKFold ;
- les scores sont comparés à la référence du notebook 09 ;
- l'effet de `rater_id` est documenté ;
- le meilleur modèle est réentraîné sur toutes les données ;
- `submissions/improved.csv` est généré et vérifié.

# 18. Phase 11 — LightGBM et trajectoires patient

## Notebook

```text
notebooks/11_lgbm_advanced.ipynb
```

## Objectif

Tester LightGBM avec des variables avancées décrivant les trajectoires des
patients : pentes, statistiques, valeurs retardées et avancées, tendances
quadratiques et informations leave-one-out. Le notebook utilise des variables
catégorielles et l'early stopping, puis agrège les prédictions des cinq folds.

## Critères de validation

Cette phase est terminée si :

- les variables de trajectoire sont construites sans utiliser `target` ;
- la validation reste groupée par patient ;
- les cinq folds et leur dispersion sont affichés ;
- LightGBM est comparé au meilleur modèle précédent ;
- `submissions/lgbm_advanced.csv` est généré et vérifié.

# 19. Phase 12 — Optimisation LightGBM et stacking

## Notebook

```text
notebooks/12_optimized.ipynb
```

## Objectif

Optimiser l'approche LightGBM du notebook 11 avec Optuna, des variables
supplémentaires, un encodage de `rater_id`, un stacking Ridge sur les
prédictions OOF et un bornage cohérent avec l'échelle du score moteur.

## Critères de validation

Cette phase est terminée si :

- les essais Optuna sont comparés avec une validation groupée ;
- les nouvelles variables sont calculées de façon reproductible ;
- le stacking est évalué sur des prédictions OOF ;
- le modèle optimisé est comparé aux notebooks 10 et 11 ;
- `submissions/optimized.csv` est généré et vérifié.

# 20. Phase 13 — Ensemble XGBoost et LightGBM

## Notebook

```text
notebooks/13_xgb_ensemble.ipynb
```

## Objectif

Tester XGBoost avec Optuna et le combiner à LightGBM. Le notebook corrige
également le budget d'itérations de LightGBM pour les folds qui n'avaient pas
convergé et recherche une pondération d'ensemble sur les prédictions OOF.

## Critères de validation

Cette phase est terminée si :

- XGBoost est évalué avec les mêmes groupes patients ;
- l'ensemble est comparé séparément à LightGBM et XGBoost ;
- les pondérations sont déterminées à partir des prédictions OOF ;
- les prédictions finales ne contiennent aucune valeur manquante ;
- `submissions/xgb_ensemble.csv` est généré et vérifié.

# 21. Phase 14 — Ensemble de niveau 2 sans fuite

## Notebook

```text
notebooks/14_ensemble_l2.ipynb
```

## Objectif

Construire une validation honnête en corrigeant la fuite du target encoding de
`rater_id` identifiée dans le notebook 12. Les features sont calculées à partir
de `X` uniquement, l'encodage est limité aux données d'entraînement de chaque
fold, puis un second niveau exploite les prédictions OOF et le lissage des
trajectoires par patient.

## Critères de validation

Cette phase est terminée si :

- le target encoding est recalculé indépendamment dans chaque fold ;
- les transformations respectent la séparation train/validation ;
- l'ensemble de niveau 1 et le niveau 2 sont comparés ;
- la projection monotone et le clipping sont évalués explicitement ;
- `submissions/14_ensemble_l2.csv` est généré et vérifié.

# 22. Phase 15 — Ensemble multi-algorithmes final

## Notebook

```text
notebooks/15_multi_models.ipynb
```

## Objectif

Prolonger le notebook 14 avec plusieurs familles de modèles au niveau 1 :
LightGBM, CatBoost, XGBoost, HistGradientBoosting, MLP et Ridge. Le notebook
retient automatiquement les modèles utiles, optimise le blend, applique le
lissage patient et produit la soumission retenue du projet.

## Critères de validation

Cette phase est terminée si :

- tous les modèles sont comparés avec la même validation GroupKFold ;
- aucune feature ou statistique de validation n'utilise la cible du fold
  courant ;
- les contributions des modèles au blend sont documentées ;
- les performances OOF et la stabilité par fold sont affichées ;
- la soumission finale et son rapport Skore sont vérifiés ;
- `submissions/15_multi_models.csv` est généré et retenu.

# 23. Génération de la soumission finale

La soumission retenue doit être produite dans Bob IDE depuis
`notebooks/15_multi_models.ipynb`. Le notebook 09 reste un jalon de sélection
initial et n'est plus le dernier notebook du projet.

Elle est générée depuis le notebook 15, qui doit conserver une dernière
cellule claire et reproductible.

## Étapes

1. reconstruire le pipeline final ;
2. recharger les données ;
3. fusionner `X_train` et `y_train` ;
4. préparer les features ;
5. entraîner sur toutes les données ;
6. appliquer les mêmes transformations à `X_test` ;
7. prédire `target` ;
8. construire le fichier de soumission ;
9. vérifier son format ;
10. envoyer le rapport à Skore ;
11. soumettre `submissions/15_multi_models.csv` sur Kaggle.

## Format attendu

```python
submission = X_test[["Index"]].copy()
submission["target"] = predictions
```

## Vérifications obligatoires

```python
assert list(submission.columns) == [
    "Index",
    "target",
]

assert len(submission) == len(X_test)

assert submission["Index"].equals(
    X_test["Index"]
)

assert submission["target"].notna().all()
```

Le fichier doit être enregistré dans :

```text
submissions/
```

---

# 24. Utilisation de Skore

Chaque expérience doit être évaluée avec `skore.evaluate`.

Chaque rapport doit recevoir une clé différente :

```text
01_dummy
02_ridge
03_grouped_ridge
04_hgbr
05_tabular
06_features
07_final
08_improved
09_lgbm_advanced
10_optimized
11_lgbm_final
12_xgb_final
14_ensemble_l2
15_multi_models
```

Pour chaque expérience, documenter :

- le modèle ;
- les variables ;
- les transformations ;
- la validation ;
- la RMSE ;
- les hyperparamètres ;
- les conclusions ;
- l’URL du rapport Skore.

L’URL Skore doit être ajoutée dans la description Kaggle de chaque soumission.

---

# 25. Bonnes pratiques dans Bob IDE

Toutes les modifications doivent être effectuées depuis Bob IDE.

Pour chaque étape, demander à Bob de :

1. lire les fichiers concernés ;
2. créer le notebook demandé ;
3. ne pas supprimer les notebooks précédents ;
4. ne pas inventer de colonnes ;
5. vérifier les chemins relatifs ;
6. séparer le texte explicatif du code ;
7. exécuter le notebook ;
8. analyser les résultats ;
9. corriger les erreurs ;
10. rédiger une conclusion.

Exemple de consigne générale :

```text
Travaille uniquement dans Bob IDE.

Crée le notebook demandé de manière autonome.
Ne réutilise aucun résultat, aucune variable en mémoire et aucun code
provenant d’un ancien notebook.

Le notebook doit recharger les fichiers depuis data/, être exécutable
de haut en bas après redémarrage du noyau, afficher les résultats,
expliquer les décisions et vérifier les sorties.
```

---

# 26. Difficultés à anticiper

## Valeurs manquantes

Certaines colonnes contiennent beaucoup de valeurs absentes.

Il faut choisir une stratégie adaptée :

- imputation pour Ridge ;
- gestion native pour HGBR ;
- vectorisation adaptée pour les modèles tabulaires ;
- indicateurs binaires de missingness lorsque cela est pertinent.

## Fuite entre patients

La présence de plusieurs visites par patient impose une validation groupée.

`GroupKFold` doit utiliser :

```python
groups = visits["patient_id"]
```

## Différence entre validation locale et Kaggle

Les scores locaux et Kaggle peuvent différer.

Il faut surtout rechercher une tendance cohérente entre :

- la validation GroupKFold ;
- les différentes soumissions ;
- le leaderboard Kaggle.

## État des notebooks

Une variable conservée en mémoire peut rendre un notebook trompeur.

Toujours redémarrer le noyau puis exécuter toutes les cellules.

## Ordre des lignes

Un tri ou une fusion peut modifier l’ordre des lignes.

La soumission doit respecter exactement l’ordre de `X_test`.

## Colonnes catégorielles

Les catégories du test peuvent différer de celles du train.

Le pipeline doit gérer ces situations.

## Sur-optimisation

Tester trop de modèles sur une seule validation peut conduire à choisir un modèle par hasard.

Les modèles doivent être comparés avec les mêmes folds.

---

# 27. Tests à effectuer

## Tests des données

```python
assert X_train["Index"].is_unique
assert y_train["Index"].is_unique
assert X_test["Index"].is_unique
```

## Tests de la fusion

```python
assert len(visits) == len(X_train)
assert len(visits) == len(y_train)
assert visits["target"].notna().all()
```

## Tests de GroupKFold

```python
for train_indices, valid_indices in cv_splits:
    train_patients = set(
        groups.iloc[train_indices]
    )

    valid_patients = set(
        groups.iloc[valid_indices]
    )

    assert train_patients.isdisjoint(
        valid_patients
    )
```

## Tests des prédictions

```python
assert len(predictions) == len(X_test)
assert pd.Series(predictions).notna().all()
```

## Tests de la soumission

```python
assert list(submission.columns) == [
    "Index",
    "target",
]

assert len(submission) == len(X_test)

assert submission["Index"].equals(
    X_test["Index"]
)

assert submission["target"].notna().all()
```

## Validation des notebooks

Chaque notebook est valide uniquement si :

- il peut être exécuté après redémarrage du noyau ;
- toutes les cellules s’exécutent ;
- aucun chemin absolu personnel n’est utilisé ;
- les résultats sont affichés ;
- les conclusions sont présentes ;
- il ne dépend pas d’un autre notebook.

---

# 28. Tableau de suivi des expériences

Maintenir un tableau dans le notebook 15 :

| Expérience | Notebook | Modèle | Validation | RMSE | Score Kaggle | Statut |
|---|---|---|---|---:|---:|---|
| Baseline | `02_baseline.ipynb` | DummyRegressor | à préciser | à compléter | à compléter | à faire |
| Ridge | `03_ridge.ipynb` | Ridge | à préciser | à compléter | à compléter | à faire |
| Ridge groupé | `04_grouped_validation.ipynb` | Ridge | GroupKFold | à compléter | à compléter | à faire |
| HGBR | `05_hist_gradient_boosting.ipynb` | HGBR | GroupKFold | à compléter | à compléter | à faire |
| Tabulaire | `06_tabular_pipeline.ipynb` | skrub | GroupKFold | à compléter | à compléter | à faire |
| Features | `07_feature_engineering.ipynb` | modèle amélioré | GroupKFold | à compléter | à compléter | à faire |
| Analyse des erreurs | `08_error_analysis.ipynb` | analyse du meilleur modèle initial | GroupKFold | à compléter | — | documenté |
| Premier final | `09_final_model.ipynb` | meilleur modèle initial | GroupKFold | à compléter | à compléter | jalon |
| Amélioration | `10_improved_model.ipynb` | HGBR/skrub optimisé | GroupKFold | à compléter | à compléter | à faire |
| Trajectoires | `11_lgbm_advanced.ipynb` | LightGBM avancé | GroupKFold | à compléter | à compléter | à faire |
| Optimisation | `12_optimized.ipynb` | LightGBM + Optuna + stacking | GroupKFold | à compléter | à compléter | à faire |
| Ensemble XGB | `13_xgb_ensemble.ipynb` | LightGBM + XGBoost | GroupKFold | à compléter | à compléter | à faire |
| Ensemble niveau 2 | `14_ensemble_l2.ipynb` | ensemble sans fuite + lissage | GroupKFold | à compléter | à compléter | à faire |
| Final retenu | `15_multi_models.ipynb` | ensemble multi-algorithmes | GroupKFold | à compléter | à compléter | retenu |

---

# 29. Ordre de travail complet

Suivre l’ordre suivant :

1. créer `notebooks/01_exploration.ipynb` ;
2. charger les fichiers CSV ;
3. fusionner `X_train` et `y_train` ;
4. vérifier les dimensions ;
5. analyser les colonnes ;
6. analyser les valeurs manquantes ;
7. analyser la cible ;
8. analyser les patients ;
9. documenter les risques de fuite ;
10. créer `notebooks/02_baseline.ipynb` ;
11. construire le DummyRegressor ;
12. évaluer la baseline avec Skore ;
13. créer une première soumission ;
14. créer `notebooks/03_ridge.ipynb` ;
15. construire le pipeline Ridge ;
16. tester plusieurs valeurs de `alpha` ;
17. comparer Ridge à la baseline ;
18. créer `notebooks/04_grouped_validation.ipynb` ;
19. mettre en place GroupKFold ;
20. réévaluer les modèles avec les groupes patients ;
21. créer `notebooks/05_hist_gradient_boosting.ipynb` ;
22. tester HGBR ;
23. comparer HGBR et Ridge ;
24. créer `notebooks/06_tabular_pipeline.ipynb` ;
25. intégrer les colonnes catégorielles ;
26. créer `notebooks/07_feature_engineering.ipynb` ;
27. tester les nouvelles variables ;
28. créer `notebooks/08_error_analysis.ipynb` ;
29. analyser les erreurs ;
30. créer `notebooks/09_final_model.ipynb` ;
31. comparer les résultats ;
32. sélectionner le meilleur modèle ;
33. entraîner le modèle final sur toutes les données ;
34. produire le fichier CSV final ;
35. vérifier le format ;
36. envoyer le rapport à Skore ;
37. soumettre le premier fichier sur Kaggle ;
38. documenter le score et les limites du premier modèle final ;
39. créer `notebooks/10_improved_model.ipynb` et tuner le pipeline HGBR/skrub ;
40. comparer l'effet de `rater_id` et produire `submissions/improved.csv` ;
41. créer `notebooks/11_lgbm_advanced.ipynb` et construire les features de trajectoire ;
42. entraîner LightGBM avec early stopping et produire `submissions/lgbm_advanced.csv` ;
43. créer `notebooks/12_optimized.ipynb` et optimiser LightGBM avec Optuna ;
44. évaluer le stacking Ridge et produire `submissions/optimized.csv` ;
45. créer `notebooks/13_xgb_ensemble.ipynb` ;
46. comparer XGBoost, LightGBM et leur ensemble dans `submissions/xgb_ensemble.csv` ;
47. créer `notebooks/14_ensemble_l2.ipynb` ;
48. corriger le target encoding et valider l'ensemble de niveau 2 sans fuite ;
49. créer `notebooks/15_multi_models.ipynb` ;
50. comparer les familles de modèles supplémentaires et optimiser le blend final ;
51. produire `submissions/15_multi_models.csv` ;
52. sélectionner cette soumission sur la base de la validation honnête et du
    score Kaggle ;
53. documenter le score final, le rapport Skore et les limites.

---

# 30. Validation finale

## Données

- [ ] Les fichiers CSV sont correctement chargés.
- [ ] La fusion est réalisée sur `Index`.
- [ ] Aucune ligne n’est perdue.
- [ ] Les valeurs manquantes sont analysées.
- [ ] Les patients et leurs visites sont compris.
- [ ] L’ordre de `X_test` est conservé.

## Notebooks

- [ ] Les 15 notebooks sont présents dans l’ordre.
- [ ] Chaque notebook possède un objectif clair.
- [ ] Chaque notebook peut être exécuté de haut en bas.
- [ ] Le noyau a été redémarré avant validation.
- [ ] Les notebooks ne dépendent pas de variables en mémoire.
- [ ] Les conclusions sont rédigées.

## Modèles

- [ ] Une baseline existe.
- [ ] Ridge a été testé.
- [ ] GroupKFold a été utilisé.
- [ ] HistGradientBoosting a été évalué.
- [ ] Les variables catégorielles ont été étudiées.
- [ ] Le feature engineering a été contrôlé.
- [ ] Le modèle final est sélectionné sur des critères justifiés.
- [ ] Les notebooks 10 à 15 ont été comparés au jalon du notebook 09.
- [ ] Le notebook 15 produit la soumission retenue.
- [ ] Aucune fuite de données n’est présente.

## Skore

- [ ] Chaque expérience possède une clé unique.
- [ ] Les rapports ont été envoyés à Skore Hub.
- [ ] Les RMSE sont documentées.
- [ ] Les URLs des rapports sont conservées.
- [ ] Le rapport final correspond au modèle soumis.

## Kaggle

- [ ] Le fichier de soumission est généré.
- [ ] Les colonnes sont exactement `Index,target`.
- [ ] Le nombre de lignes correspond à `X_test`.
- [ ] Les `Index` sont identiques et dans le même ordre.
- [ ] Aucune prédiction n’est manquante.
- [ ] Le fichier est accepté par Kaggle.
- [ ] L’URL Skore est indiquée dans la description.
- [ ] Le score Kaggle est documenté.

---

# 31. Définition de terminé

Le projet est terminé lorsque :

- les notebooks ont été réalisés dans Bob IDE ;
- les données ont été comprises ;
- une baseline a été construite ;
- plusieurs modèles ont été comparés ;
- la validation par patient a été utilisée ;
- les valeurs manquantes ont été traitées ;
- les catégories ont été étudiées ;
- les erreurs ont été analysées ;
- le modèle final a été sélectionné ;
- le fichier final est valide ;
- Kaggle accepte la soumission ;
- le rapport Skore est associé à la soumission ;
- les résultats sont documentés ;
- le projet peut être repris et exécuté depuis le début dans Bob IDE.

Le projet final doit être compréhensible par une autre personne de l’équipe sans dépendre :

- d’un ancien chat ;
- d’un résultat conservé en mémoire ;
- d’un ancien fichier non documenté ;
- d’un notebook exécuté dans un ordre particulier ;
- d’un code extérieur à Bob IDE.

La règle fondamentale est :

```text
Chaque notebook doit être autonome.
Chaque étape doit être documentée.
Chaque résultat doit être reproductible.
Toute la réalisation doit être effectuée dans Bob IDE.
```