
---

````markdown
# 🛡️ PrédiSinistre — Détection de Fraude Assurantielle

**Projet Data Science** | Détection automatique de sinistres frauduleux dans les réclamations d'assurance automobile

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Dataset](#dataset)
4. [Méthodologie](#méthodologie)
5. [Résultats](#résultats)
6. [Installation & Utilisation](#installation--utilisation)
7. [Architecture](#architecture)

---

## 🎯 Vue d'ensemble

**PrédiSinistre** est un projet de classification binaire visant à **détecter automatiquement les fraudes** dans les réclamations d'assurance automobile. 

**Contexte métier :**
- Les fraudes assurantielles représentent des pertes massives pour les assureurs
- La détection manuelle est coûteuse et non-scalable
- Un modèle ML peut flaguer les sinistres suspects pour audit humain

**Défi technique :**
- Dataset **déséquilibré** : ~75 % légitime / ~25 % fraude
- 1 000 sinistres avec 40 variables hétérogènes (numériques + catégorielles)
- Nécessité de gérer **data leakage** lors du preprocessing
- Interprétabilité requise pour justifier les décisions

---

## 📁 Structure du projet

```
sep25cds_predisinistre/
├── data/
│   |── raw/
│   |   └── insurance_claims.csv          # Dataset brut (1 000 × 40)
|   └── processed/
|       └── X_train.csv         
|       └── y_test.pkl          
|       └── insurance_claims_encoded.csv
|       └── insurance_claims_clean.csv
|
├── models/
|   └── best_pipeline_gb.pkl 
|   └── best_pipeline_hgb.pkl 
|   └── best_pipeline_xgb.pkl 
|   └── cat_modalities.pkl 
|   └── feature_names.pkl 
|   └── model_metadata.json 
|   └── num_stats.json 
|   └── y_prob_gb.pkl 
|   └── y_prob_hgb.pkl 
|   └── y_prob_xgb.pkl 
|
├── notebooks/
│   ├── PrediSinistre_EDA.ipynb                          # EDA & Preprocessing
│   └── PrediSinistre_modelisation_de_donnees.ipynb      # Modélisation de données & Optimisation
|   └── PrediSinistre_streamlit_app.py                   # Application python pour la démo Streamlit
|
├── reports/                                # Rapports et livrables
|   ├── PrediSinistre_EDA_Report.html       # Rendu 1 en HTML
|   ├── Présentation_PrédiSinistre.pptx  
|   ├── Rapport_PrédiSinistre.docx  
|     
└── README.md                              # Ce fichier
```

---

## 📊 Dataset

### Source et Dimensions

| Élément | Détail |
|--------|--------|
| **Observations** | 1 000 sinistres |
| **Variables** | 40 colonnes (brutes) |
| **Cible** | `fraud_reported` (Y=fraude / N=légitime) |
| **Déséquilibre** | 753 légitimes (75.3 %) vs 247 fraudes (24.7 %) — Ratio 3:1 |

### Variables Principales

#### 📌 Données Assuré
- `months_as_customer` : ancienneté du client (mois)
- `age` : âge de l'assuré
- `insured_sex`, `insured_education_level`, `insured_occupation`
- `insured_relationship` : statut relationnel

#### 📌 Police d'Assurance
- `policy_state`, `policy_deductable`, `policy_annual_premium`
- `umbrella_limit` : limite de couverture complémentaire
- `capital-gains`, `capital-loss` : revenus/pertes en capital

#### 📌 Sinistre
- `incident_type` : type d'incident (collision simple, multi-véhicule, vol, etc.)
- `incident_severity` : gravité (Minor, Major, Trivial, Total Loss)
- `collision_type`, `property_damage`, `authorities_contacted`
- `total_claim_amount`, `injury_claim`, `property_claim`, `vehicle_claim`

#### ⚠️ Valeurs Manquantes Identifiées
- `authorities_contacted` : 91 NaN (9.1 %)
- `collision_type` : 178 '?' (17.8 %)
- `property_damage` : 360 '?' (36 %)
- `police_report_available` : 343 '?' (34.3 %)

---

## 🔬 Méthodologie

### Phase 1 : Exploration & Nettoyage (Rendu 1)

#### ✓ Analyse Univariée
- Distributions des variables numériques (moyenne, écart-type, quartiles)
- Modalités des variables catégorielles
- Identification des valeurs aberrantes et manquantes

#### ✓ Data Visualization
6 graphiques commentés :
1. Distribution de la cible (déséquilibre)
2. Distribution d'âge par fraude
3. Montant du sinistre vs prime annuelle
4. Type d'incident et taux de fraude
5. Sévérité vs montant réclamé
6. Localisation géographique des sinistres

#### ✓ Nettoyage & Imputation
- **Colonnes supprimées** : `_c39` (vide), `policy_number` (ID), `incident_location` (1000 uniques), `insured_zip` (redondant)
- **Remplacement '?' → NaN** : uniformisation des valeurs manquantes
- **Imputation intelligente** :
  - `authorities_contacted` → 'None' (valeur neutre)
  - `collision_type`, `property_damage` → **mode conditionnel par fraude** (préserve la dépendance statistique)
  - `police_report_available` → **mode conditionnel par fraude**

### Phase 2 : Feature Engineering & Modélisation (Rendu 2)

#### ✓ Feature Engineering

**Transformations temporelles (sans fuite) :**
```python
policy_age_years = (incident_date - policy_bind_date) / 365.25
incident_month = incident_date.month
incident_dayofweek = incident_date.dayofweek
```

**Ratios de sinistre (déterministes) :**
```python
claim_ratio = total_claim_amount / policy_annual_premium
injury_share = injury_claim / total_claim_amount
vehicle_share = vehicle_claim / total_claim_amount
is_high_deductable = (policy_deductable > 1000).astype(int)
```

**Résultat :** 40 features finales (23 numériques + 17 catégorielles)

#### ✓ Prévention du Data Leakage

**Problème :** Si on encode X entier avant split → le test set contamine l'encodage du train set

**Solution :** Pipeline sklearn garantissant l'ordre correct

```
✗ MAUVAIS ordre (data leakage)
   encodage(X_entier) → split(train/test)
   → Les stats d'encodage s'apprennent sur train+test

✓ BON ordre (Pipeline sklearn)
   split(train/test) → Pipeline.fit(X_train) → Pipeline.predict(X_test)
   → Aucune statistique du test n'influence l'apprentissage
```

#### ✓ Architecture du Pipeline

```python
ImbPipeline([
    ("preprocessor", ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(...), cat_cols)
    ])),
    ("smote", SMOTE(random_state=42)),  # Rééquilibrage train-only
    ("classifier", ModeleML)             # Modèle final
])
```

#### ✓ Gestion du Déséquilibre

| Technique | Détail |
|-----------|--------|
| **SMOTE** | Surêchantillonnage synthétique → crée 404 exemples fraude (train seulement) |
| **class_weight** | Pondération : légitime=0.664, fraude=2.020 |
| **Métriques** | ROC-AUC & F1-score au lieu d'accuracy |

### Phase 3 : Comparaison des Modèles

| Modèle | ROC-AUC | F1-Score | Recall | Notes |
|--------|---------|----------|--------|-------|
| **Logistic Regression** | 0.597 ± 0.050 | 0.385 ± 0.036 | 0.577 ± 0.084 | Baseline (sous-performant) |
| **Random Forest** | 0.913 ± 0.016 | 0.647 ± 0.037 | 0.556 ± 0.060 | Bon, mais rappel limité |
| **AdaBoost** | 0.921 ± 0.018 | 0.726 ± 0.034 | 0.702 ± 0.074 | Performance solide |
| **Gradient Boosting** | 0.921 ± 0.012 | 0.739 ± 0.042 | 0.712 ± 0.060 | Excellent équilibre |
| **HistGradientBoosting** | 0.918 ± 0.018 | 0.741 ± 0.057 | 0.728 ± 0.079 | Stable & rapide |
| **XGBoost** | 0.913 ± 0.018 | 0.754 ± 0.052 | 0.794 ± 0.086 | ⭐ **Meilleur Recall** |

**Sélection :** **XGBoost** choisi pour son excellent rappel (79.4 %) — priorité : capturer max de fraudes

---

## 📈 Résultats

### Performances Finales (Test Set)

```
Accuracy  : 95.0%
Precision : 91.2%  ← sur les sinistres flagués comme fraude, 91% sont vrais positifs
Recall    : 79.4%  ← on détecte 79% des vraies fraudes
F1-Score  : 84.8%
ROC-AUC   : 0.934
```

### Interprétabilité

#### Permutation Importance (Top 10)
1. `total_claim_amount` — montant total réclamé
2. `claim_ratio` — ratio sinistre/prime
3. `policy_annual_premium` — prime annuelle
4. `injury_claim` — montant corporel réclamé
5. `vehicle_claim` — montant véhicule
6. `incident_severity` — gravité rapportée
7. `authorities_contacted` — autorités contactées
8. `incident_state` — lieu du sinistre
9. `property_damage` — dommages matériels
10. `police_report_available` — rapport de police disponible

#### Insights Métier

- **Fraudes tendent à :**
  - Déclarer des montants élevés (disproportionnés à la prime)
  - Réclamer sans rapport de police officiel
  - Se produire peu de temps après signature (police jeune)
  - Être classées en "gravité majeure" systématiquement

- **Signaux de légitimité :**
  - Présence d'un rapport de police
  - Cohérence montant/prime (ratio modéré)
  - Sinistres proches de la signature de police

---

## 🚀 Installation & Utilisation

### Prérequis

```bash
python==3.12.8
pandas==2.2.0
numpy==2.4.2
scikit-learn==1.8.0
imbalanced-learn==0.14.1
xgboost==3.2.0
matplotlib==3.9.0
seaborn==0.13.2
shap==0.51.0
joblib==1.5.3
scipy==1.13.1
plotly==6.5.2
streamlit==1.55.0
```

### Installation

```bash
git clone https://github.com/christperyclais/sep25cds_predisinistre.git
cd sep25cds_predisinistre

# Créer un environnement virtuel (venv)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt


# Créer un environnement virtuel (Pyenv & Poetry)
pyenv install 3.12.8 # installation de la verision de Python
pyenv local 3.12.8 # définition de la version python à utiliser pour le projet
poetry env use $(pyenv which python) # utilisation la version local définie
poetry add $(cat requirements.txt) # installation des packages depuis le fichier "requirements.txt"

```



### Utilisation

#### 1. Rendu 1 : Exploration & Visualisation
```bash
jupyter notebook notebooks/PrediSinstre_EDA.ipynb
```
- Charge `data/raw/insurance_claims.csv`
- Génère 6 graphiques exploratoires
- Produit dataset nettoyé pour modélisation

#### 2. Rendu 2 : Modélisation Avancée
```bash
jupyter notebook notebooks/PrediSinsitre_modelisation_de_donnees.ipynb
```
- Charge dataset nettoyé du Rendu 1
- Compare 6 algorithmes ML
- Optimise hyperparamètres du meilleur modèle
- Génère rapports d'interprétabilité
- Sauvegarde le modèle final

#### 3. Faire une Prédiction
```python
import joblib
import pandas as pd

# Préparer un nouveau sinistre
nouveau_sinistre = pd.DataFrame({
    'age': [45],
    'total_claim_amount': [50000],
    # ... autres features
})

# Prédiction
prob_fraude = pipeline.predict_proba(nouveau_sinistre)[0, 1]
prediction = "⚠️ FRAUDE" if prob_fraude > 0.5 else "✓ Légitime"
print(f"Probabilité fraude : {prob_fraude:.1%} → {prediction}")
```

---

## 🏗️ Architecture

### Flux de Traitement

```
insurance_claims.csv (brut)
         ↓
    [Rendu 1: EDA]
         ↓
   insurance_claims_clean.csv
         ↓
    [Rendu 2: Modélisation]
         ├─ Split train/test
         ├─ Pipeline (encode, SMOTE, classify)
         ├─ Validation croisée 5-fold
         ├─ Tuning hyperparamètres
         ├─ Évaluation finale
         └─ Interprétabilité
         ↓
   best_pipeline_xgb.pkl, best_pipeline_hgb.pkl, best_pipeline_gb.pkl + rapports
```

### Fichiers Clés

| Fichier | Contenu |
|---------|---------|
| `PrediSinistre_EDA.ipynb` | 9 sections : imports, chargement, analyse univariée, visualisation, nettoyage, encodage, export, baseline, conclusions |
| `PrediSinistre_modelisation_de_donnees.ipynb` | 12 sections : preprocessing, feature eng, split, pipeline, déséquilibre, comparaison modèles, tuning, seuil, interprétabilité, évaluation, sauvegarde |
| `insurance_claims.csv` | Dataset source (1 000 × 40) |
| `insurance_claims_clean.csv` | Dataset nettoyé (1 000 × 37) — entrée modélisation |

---

## 🎓 Concepts Clés

### 1. Data Leakage
**Problème :** Si on norme/encode avant split → le test set influence l'apprentissage des statistiques  
**Solution :** Pipeline sklearn qui fit uniquement sur train, transforme train+test uniformément

### 2. Déséquilibre des Classes
**Problème :** 75 % légitime / 25 % fraude → un modèle naïf obtient 75 % d'accuracy  
**Solution :** SMOTE (synthétique) + pondération des classes + ROC-AUC comme métrique

### 3. Overfitting vs Underfitting
- **Validation croisée stratifiée** : préserve le ratio fraude/légitime dans chaque fold
- **Cross-val score sur Pipeline** : refitte le préprocessor correctement à chaque fold
- **Hyperparamètres tunés** : GridSearchCV/RandomizedSearchCV

### 4. Interprétabilité
- **Permutation Importance** : impact de chaque feature sur la performance
- **SHAP values** : contribution locale de chaque feature par prédiction

---

## 📚 Bibliographie & Ressources

- **scikit-learn Pipeline :** [docs](https://scikit-learn.org/stable/modules/compose.html)
- **imbalanced-learn SMOTE :** [docs](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- **XGBoost :** [docs](https://xgboost.readthedocs.io/)
- **SHAP :** [GitHub](https://github.com/slundberg/shap)

---

## 👤 Auteur

**Christ Peryclais NGOLO et Souhaibou Seynou** — Projet DataScientest (Juillet 2026)

---

## 📄 Licence

Ce projet est fourni à titre éducatif. Reproduction et adaptation autorisées avec attribution.

---

## 🔗 Liens Utiles

- **Repository :** https://github.com/christperyclais/sep25cds_predisinistre
- **Jupyter Notebooks :** Voir dossier `notebooks/`
- **Dataset brut :** `data/raw/insurance_claims.csv`

---

**Dernière mise à jour :** Juillet 2026  
**Version :** 2.0 (Rendu 2 — Modélisation complète)
````

****