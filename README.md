# Projet Prédisinistre — Étape 1 (EDA) | Contribution Alex

⚠️ Ce README décrit le travail réalisé dans la branche **alex_eda**.
Il ne constitue pas encore la version consolidée du projet.

## Objectif de l’Étape 1
Réaliser une analyse exploratoire des données (EDA) afin de comprendre la structure du jeu de données,
évaluer sa qualité, identifier les variables pertinentes pour la prédiction de la fraude pré-sinistre
et poser des constats étayés par des visualisations et des validations statistiques.

## Contenu réalisé dans cette branche
- Audit de la qualité des données (valeurs manquantes, incohérences, distributions)
- Analyse de la structure et de la volumétrie
- Sélection a priori des variables pertinentes pour la fraude
- Réalisation de 5 visualisations exploratoires clés
- Interprétations métier des résultats
- Validation statistique des principaux constats

## Périmètre volontairement exclu
- Aucun preprocessing avancé
- Aucun feature engineering
- Aucun modèle de machine learning
- Aucun résultat de prédiction

## Fichiers concernés
- notebooks/01_EDA.ipynb
- reports/rapport_exploration_donnees
- Template_Rapport_exploration_rempli.xlsx

---

Project Name
==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
