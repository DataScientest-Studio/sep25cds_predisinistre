"""
=============================================================
PrédiSinistre — Application Streamlit de Détection de Fraude
=============================================================

Lancement :
    streamlit run streamlit_app.py

Dépendances :
    pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib shap imbalanced-learn xgboost

Fichiers requis (générés par modelisation_de_donnees.ipynb) :
    best_pipeline.pkl   — ImbPipeline complet (preprocessor + SMOTE + modèle champion)
    cat_modalities.pkl  — Modalités catégorielles extraites de X_train uniquement
    model_metadata.json — Métriques, seuil optimal et configuration du modèle
    num_stats.json      — Stats numériques du train set (min/max/mean/std)

Modèles candidats (notebook) : HistGradientBoosting, GradientBoosting, XGBoost
Champion sélectionné par : F1-score sur X_test (pas AUC)
Déséquilibre classes : SMOTE via ImbPipeline sur tous les modèles
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib, json, os, warnings
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, roc_curve

warnings.filterwarnings('ignore')

# SHAP — optionnel, nécessite : pip install shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# =============================================================
# CONFIGURATION DE LA PAGE
# =============================================================
st.set_page_config(
    page_title="PrédiSinistre — Détection de Fraude",
    page_icon="PS",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 22px 32px; border-radius: 14px;
        margin-bottom: 22px; color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #a8d8ea; margin: 6px 0 0 0; font-size: 1rem; }
    .prediction-badge {
        font-size: 1.8rem; font-weight: bold;
        padding: 15px 30px; border-radius: 15px;
        text-align: center; margin: 10px 0;
    }
    .fraud-badge { background:#fde8e8; color:#c0392b; border:3px solid #e74c3c; }
    .legit-badge { background:#e8fde8; color:#1a7a4a; border:3px solid #27ae60; }
    .metric-card {
        border-radius: 12px; padding: 18px 22px;
        margin: 8px 0; border-left: 6px solid;
        box-shadow: 0 3px 10px rgba(0,0,0,0.25);
    }
    .metric-card-auc       { background:#1a3a5c; border-color:#3498db; color:white; }
    .metric-card-f1        { background:#1a4a2e; border-color:#2ecc71; color:white; }
    .metric-card-recall    { background:#4a2a10; border-color:#e67e22; color:white; }
    .metric-card-precision { background:#3a1a3a; border-color:#9b59b6; color:white; }
    .metric-card .label    { font-size:0.9rem; opacity:0.85; font-weight:500; }
    .metric-card .value    { font-size:2.4rem; font-weight:800; margin-top:4px; letter-spacing:1px; }
    .metric-card .sub      { font-size:0.78rem; opacity:0.75; margin-top:3px; }
    .model-config-card {
        background:#1e2a3a; border-radius:12px;
        padding:16px 20px; margin:8px 0;
        color:white; box-shadow:0 2px 8px rgba(0,0,0,0.2);
    }
    .model-config-card .config-title {
        font-size:1rem; font-weight:700; color:#a8d8ea; margin-bottom:10px;
        border-bottom:1px solid rgba(255,255,255,0.15); padding-bottom:6px;
    }
    .model-config-card .config-row {
        display:flex; justify-content:space-between;
        padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.07); font-size:0.88rem;
    }
    .model-config-card .config-key   { color:#bdc3c7; }
    .model-config-card .config-value { color:white; font-weight:600; }
    .sidebar-info {
        background:#1e3a5a; border-radius:10px;
        padding:14px; margin:8px 0;
        font-size:0.85rem; color:white;
        border-left:4px solid #3498db;
    }
    .sidebar-info b { color:#a8d8ea; }
    .guide-card {
        background:#1a2a3a; border-radius:10px;
        padding:14px 18px; margin:6px 0;
        color:white; border-left:4px solid #e67e22;
        font-size:0.88rem; line-height:1.6;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# PREPROCESSING — identique au notebook Rendu_2
# =============================================================

def preprocess_batch(df_raw):
    """
    Applique EXACTEMENT le même preprocessing que modelisation_de_donnees.ipynb.
    Appelé sur le CSV uploadé dans l'onglet 'Analyse par lots'.

    Ordre identique au notebook (section 2 et 3) :
      1. Suppression colonnes inutiles : _c39, policy_number, incident_location, insured_zip
      2. Encodage cible fraud_reported (si présente) : Y=1, N=0
      3. Transformation dates → policy_age_years, incident_month, incident_dayofweek
      4. Remplacement '?' → NaN
      5. Correction umbrella_limit (valeur absolue)
      6. Imputation : authorities_contacted → 'None',
                      collision_type / property_damage / police_report_available → mode conditionnel
      7. Feature Engineering : claim_ratio, injury_share, vehicle_share, is_high_deductable
      8. Suppression de fraud_reported si présente
    """
    df = df_raw.copy()

    # 1. Suppression colonnes sans valeur prédictive (identique au notebook)
    cols_to_drop = ['_c39', 'policy_number', 'incident_location', 'insured_zip']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 2. Encodage cible si présente
    if 'fraud_reported' in df.columns and df['fraud_reported'].dtype == object:
        df['fraud_reported'] = (df['fraud_reported'] == 'Y').astype(int)

    # 3. Transformation des dates → features numériques (identique au notebook)
    if 'policy_bind_date' in df.columns and 'incident_date' in df.columns:
        df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
        df['incident_date']    = pd.to_datetime(df['incident_date'],    errors='coerce')
        df['policy_age_years']   = ((df['incident_date'] - df['policy_bind_date']).dt.days / 365.25).round(2)
        df['incident_month']     = df['incident_date'].dt.month
        df['incident_dayofweek'] = df['incident_date'].dt.dayofweek
        df.drop(columns=['policy_bind_date', 'incident_date'], inplace=True)
    else:
        # Valeurs par défaut si les colonnes de dates sont absentes du CSV
        if 'policy_age_years'   not in df.columns: df['policy_age_years']   = 5.0
        if 'incident_month'     not in df.columns: df['incident_month']     = 6
        if 'incident_dayofweek' not in df.columns: df['incident_dayofweek'] = 2

    # 4. Remplacement '?' → NaN (les '?' encodent des valeurs manquantes dans ce dataset)
    df.replace('?', np.nan, inplace=True)

    # 5. Correction umbrella_limit : valeur absolue (notebook section 2)
    if 'umbrella_limit' in df.columns:
        df['umbrella_limit'] = df['umbrella_limit'].abs()

    # 6. Imputation (même logique que mode_impute_conditional du notebook)
    #    authorities_contacted : NaN → 'None'
    if 'authorities_contacted' in df.columns:
        df['authorities_contacted'] = df['authorities_contacted'].fillna('None')
    #    collision_type, property_damage, police_report_available : mode global
    #    (en batch on ne dispose pas de la cible → mode global plutôt que conditionnel)
    for col in ['collision_type', 'property_damage', 'police_report_available']:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val):
                df[col] = df[col].fillna(mode_val[0])

    # 7. Feature Engineering (identique au notebook section 3)
    if 'total_claim_amount' in df.columns:
        df['claim_ratio']   = (df['total_claim_amount']
                               / df['policy_annual_premium'].replace(0, np.nan).fillna(1))
        df['injury_share']  = (df['injury_claim']
                               / df['total_claim_amount'].replace(0, np.nan).fillna(1))
        df['vehicle_share'] = (df['vehicle_claim']
                               / df['total_claim_amount'].replace(0, np.nan).fillna(1))
    if 'policy_deductable' in df.columns:
        df['is_high_deductable'] = (df['policy_deductable'] > 1000).astype(int)

    # 8. Suppression de la cible si présente
    X = df.drop(columns=['fraud_reported'], errors='ignore')
    return X


def build_input_row(input_data):
    """
    Construit un DataFrame d'une ligne depuis les saisies du formulaire.
    Les features engineered sont calculées automatiquement.
    """
    row = dict(input_data)
    row['claim_ratio']        = row['total_claim_amount'] / max(row['policy_annual_premium'], 1)
    row['injury_share']       = row['injury_claim']  / max(row['total_claim_amount'], 1)
    row['vehicle_share']      = row['vehicle_claim'] / max(row['total_claim_amount'], 1)
    row['is_high_deductable'] = int(row['policy_deductable'] > 1000)
    return pd.DataFrame([row])


# =============================================================
# CHARGEMENT DU PIPELINE
# =============================================================

@st.cache_resource
def load_artifacts():
    """
    Charge les pipelines sauvegardés depuis le notebook modelisation_de_donnees.ipynb.

    CONVENTION DE NOMMAGE DES FICHIERS :
    ─────────────────────────────────────────────────────────────────
    Le Streamlit détecte automatiquement tous les fichiers .pkl
    présents dans le dossier courant selon ces patterns :

      Noms acceptés pour XGBoost          : best_pipeline_xgb.pkl
      Noms acceptés pour HistGradientBoosting : best_pipeline_hgb.pkl
      Noms acceptés pour GradientBoosting     : best_pipeline_gb.pkl

    Ainsi qu'une liste d'alias pour s'adapter à différentes conventions :
      pipeline_XGBoost.pkl, pipeline_HistGradientBoosting.pkl, etc.

    L'ordre de chargement garantit que XGBoost apparait en premier
    dans le sélecteur (= modèle par défaut).
    ─────────────────────────────────────────────────────────────────
    """

    # ── Artefacts communs ─────────────────────────────────────────
    required_common = ['cat_modalities.pkl', 'model_metadata.json', 'num_stats.json']
    if not all(os.path.exists(f) for f in required_common):
        return _train_demo_pipeline()

    cat_modalities = joblib.load('cat_modalities.pkl')
    with open('model_metadata.json') as f: metadata  = json.load(f)
    with open('num_stats.json')      as f: num_stats = json.load(f)

    # ── Correspondance nom affiché → liste de fichiers .pkl à tester ──
    # Pour chaque modèle, on essaie les noms de fichiers dans l'ordre.
    # Le premier fichier trouvé sur le disque est chargé.
    # XGBoost est en premier → apparait par défaut dans le sélecteur.
    PIPELINE_CANDIDATES = {
        'XGBoost': [
            'best_pipeline_xgb.pkl',       # convention du notebook 
            'pipeline_XGBoost.pkl',         # convention pipeline_{nom}.pkl
            'best_pipeline_xgboost.pkl',    # variante orthographe
        ],
        'HistGradientBoosting': [
            'best_pipeline_hgb.pkl',        # convention du notebook 
            'pipeline_HistGradientBoosting.pkl',
            'best_pipeline_histgb.pkl',
        ],
        'GradientBoosting': [
            'best_pipeline_gb.pkl',         # convention du notebook 
            'pipeline_GradientBoosting.pkl',
            'best_pipeline_gradientboosting.pkl',
        ],
    }

    # ── Chargement — uniquement ce qui existe sur le disque ───────
    pipelines = {}
    for model_name, candidates in PIPELINE_CANDIDATES.items():
        for pkl_path in candidates:
            if os.path.exists(pkl_path):
                pipelines[model_name] = joblib.load(pkl_path)
                break  # Premier fichier trouvé suffit, on passe au modèle suivant

    # Si aucun pipeline trouvé du tout → mode démo
    if not pipelines:
        return _train_demo_pipeline()

    # ── Assemblage des artefacts ──────────────────────────────────
    artifacts = {
        'pipelines'     : pipelines,       # dict {nom_affiché: pipeline}
        'cat_modalities': cat_modalities,
        'metadata'      : metadata,
        'num_stats'     : num_stats,
    }

    # feature_names pour les axes SHAP
    if os.path.exists('feature_names.pkl'):
        artifacts['feature_names'] = joblib.load('feature_names.pkl')

    # X_test / y_test pour SHAP et Permutation Importance
    if os.path.exists('X_test.csv') and os.path.exists('y_test.csv'):
        artifacts['X_test'] = pd.read_csv('X_test.csv', index_col=0)
        artifacts['y_test'] = pd.read_csv('y_test.csv', index_col=0).squeeze()

    return artifacts

def _train_demo_pipeline():
    """
    Pipeline de démonstration si les .pkl sont absents.
    Reproduit EXACTEMENT le pipeline de modelisation_de_donnees.ipynb :
      - ImbPipeline avec SMOTE
      - HistGradientBoosting comme modèle de démonstration
      - Sélection du seuil par F1-score (pas AUC)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import precision_recall_curve

    # Chemin CSV identique au notebook : ../data/raw/insurance_claims.csv
    for path in ['../data/raw/insurance_claims.csv', 'insurance_claims.csv',
                 '/mnt/project/insurance_claims.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path); break
    else:
        st.error("insurance_claims.csv introuvable. Chemin attendu : ../data/raw/insurance_claims.csv")
        st.stop()

    # Preprocessing identique au notebook (sections 2 et 3)
    df = df.drop(columns=[c for c in ['_c39','policy_number','incident_location','insured_zip']
                           if c in df.columns])
    df['fraud_reported'] = (df['fraud_reported'] == 'Y').astype(int)
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
    df['incident_date']    = pd.to_datetime(df['incident_date'],    errors='coerce')
    df['policy_age_years']   = ((df['incident_date'] - df['policy_bind_date']).dt.days / 365.25).round(2)
    df['incident_month']     = df['incident_date'].dt.month
    df['incident_dayofweek'] = df['incident_date'].dt.dayofweek
    df.drop(columns=['policy_bind_date','incident_date'], inplace=True)
    df.replace('?', np.nan, inplace=True)
    df['umbrella_limit'] = df['umbrella_limit'].abs()
    df['authorities_contacted'] = df['authorities_contacted'].fillna('None')
    # Imputation conditionnelle (mode_impute_conditional du notebook)
    for col in ['collision_type','property_damage','police_report_available']:
        for g in [0,1]:
            mask_g = df['fraud_reported'] == g
            mask_n = df[col].isnull() & mask_g
            if mask_n.sum() > 0:
                mode_val = df.loc[mask_g, col].mode()
                if len(mode_val): df.loc[mask_n, col] = mode_val[0]

    X_raw = df.drop(columns=['fraud_reported'])
    y     = df['fraud_reported']
    # Feature Engineering (section 3 du notebook)
    X_raw['claim_ratio']        = X_raw['total_claim_amount'] / X_raw['policy_annual_premium'].replace(0,np.nan).fillna(1)
    X_raw['injury_share']       = X_raw['injury_claim'] / X_raw['total_claim_amount'].replace(0,np.nan).fillna(1)
    X_raw['vehicle_share']      = X_raw['vehicle_claim'] / X_raw['total_claim_amount'].replace(0,np.nan).fillna(1)
    X_raw['is_high_deductable'] = (X_raw['policy_deductable'] > 1000).astype(int)

    cat_cols = X_raw.select_dtypes(include='object').columns.tolist()
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    # Split avant encodage (section 4 du notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        # drop=None : conforme au notebook (arbres, pas de régression)
        ('cat', OneHotEncoder(drop=None, handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    # ImbPipeline avec SMOTE — identique à build_pipeline(clf, use_smote=True) du notebook
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=4,
                class_weight='balanced', random_state=42))
        ])
    except ImportError:
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=4,
                class_weight='balanced', random_state=42))
        ])

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:,1]
    precs, recs, threshs = precision_recall_curve(y_test, y_prob)
    f1v = 2*precs[:-1]*recs[:-1]/(precs[:-1]+recs[:-1]+1e-10)
    # Seuil sélectionné par F1-score — identique au notebook section 9
    opt = float(threshs[np.argmax(f1v)])
    yp  = (y_prob >= opt).astype(int)

    pp = preprocessor.fit(X_train)
    ohe_names = pp.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    # Modalités depuis X_train uniquement (section 5 du notebook)
    cat_modalities = {col: sorted(X_train[col].dropna().unique().tolist()) for col in cat_cols}
    num_stats = {col: {'min':float(X_train[col].min()),'max':float(X_train[col].max()),
                       'mean':float(X_train[col].mean()),'std':float(X_train[col].std())}
                 for col in num_cols if col in X_train.columns}
    metadata = {
        'model_name':'HistGradientBoosting (demo)','optimal_threshold':opt,
        'roc_auc_test':float(roc_auc_score(y_test,y_prob)),
        'f1_test':float(f1_score(y_test,yp)),'recall_test':float(recall_score(y_test,yp)),
        'precision_test':float(precision_score(y_test,yp,zero_division=0)),
        'features_count':len(num_cols)+len(ohe_names),
        'train_size':len(y_train),'test_size':len(y_test),'fraud_rate_train':float(y_train.mean()),
        'num_cols':num_cols,'cat_cols':cat_cols
    }
    return {'pipeline':pipeline,'cat_modalities':cat_modalities,'metadata':metadata,'num_stats':num_stats,
            'X_test':X_test,'y_test':y_test,'y_prob':y_prob,
            'feature_names': num_cols + pp.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()}


# =============================================================
# GRAPHIQUES
# =============================================================

def plot_gauge(probability, threshold):
    """Jauge circulaire de risque de fraude."""
    fig, ax = plt.subplots(figsize=(4, 3.5), subplot_kw=dict(polar=True))
    color = '#e74c3c' if probability >= threshold else \
            '#f39c12' if probability >= threshold*0.7 else '#2ecc71'
    label = 'FRAUDE PROBABLE' if probability >= threshold else \
            'RISQUE MODÉRÉ' if probability >= threshold*0.7 else 'FAIBLE RISQUE'
    theta = np.linspace(0, np.pi, 100)
    ax.fill_between(theta, 0.7, 1.0, color='#ecf0f1', alpha=0.8)
    ax.fill_between(np.linspace(0, probability*np.pi, 100), 0.7, 1.0, color=color, alpha=0.9)
    angle = probability * np.pi
    ax.plot([angle,angle],[0,0.85], color='black', linewidth=3, zorder=5)
    ax.scatter([angle],[0.85], color='black', s=50, zorder=6)
    ax.text(np.pi/2, 0.35, f'{probability:.1%}', ha='center', va='center',
            fontsize=20, fontweight='bold', color=color)
    ax.text(np.pi/2, 0.15, label, ha='center', va='center',
            fontsize=8, fontweight='bold', color=color)
    ax.axvline(x=threshold*np.pi, color='gray', ls='--', lw=2, alpha=0.7)
    ax.set_ylim(0,1.1); ax.set_axis_off()
    ax.set_theta_zero_location('W'); ax.set_theta_direction(-1)
    plt.tight_layout()
    return fig


def plot_local_explanation(pipeline, X_instance_df, X_ref_df, n_top=10):
    """Explication locale par perturbation vers la médiane/mode."""
    base_prob = pipeline.predict_proba(X_instance_df)[0, 1]
    contributions = []
    for col in X_instance_df.columns:
        X_pert = X_instance_df.copy()
        X_pert[col] = X_ref_df[col].mode()[0] if X_ref_df[col].dtype == object \
                      else X_ref_df[col].median()
        contributions.append(base_prob - pipeline.predict_proba(X_pert)[0, 1])
    contrib_df = pd.DataFrame({'feature':X_instance_df.columns,'contribution':contributions})\
                   .sort_values('contribution', key=abs, ascending=False).head(n_top)\
                   .sort_values('contribution')
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#1a2a3a'); ax.set_facecolor('#1a2a3a')
    ax.barh(contrib_df['feature'], contrib_df['contribution'],
            color=['#e74c3c' if v>0 else '#2ecc71' for v in contrib_df['contribution']],
            edgecolor='white', linewidth=0.5, alpha=0.9)
    ax.axvline(0, color='white', lw=1.5)
    ax.set_xlabel('Contribution à P(fraude)', fontsize=10, color='white')
    ax.set_title('Facteurs explicatifs de la prédiction', fontweight='bold', fontsize=11, color='white')
    ax.tick_params(colors='white'); ax.grid(axis='x', alpha=0.2, color='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    legend = [mpatches.Patch(color='#e74c3c', label='↑ Augmente le risque'),
              mpatches.Patch(color='#2ecc71', label='↓ Diminue le risque')]
    ax.legend(handles=legend, fontsize=9, facecolor='#1a2a3a', labelcolor='white')
    plt.tight_layout()
    return fig


# plot_global_importance intentionnellement supprimée
# (causait une erreur d'affichage dans Streamlit)


def get_model_metrics(pipeline, X_test, y_test, metadata_base, selected_model):
    """
    Calcule dynamiquement les métriques du modèle sélectionné sur X_test.

    Si X_test est disponible, on recalcule les vraies métriques du modèle actif.
    Sinon, on retourne les métriques du champion depuis model_metadata.json.

    Paramètres:
    -----------
    pipeline      : pipeline du modèle sélectionné
    X_test        : données de test brutes (peut être None)
    y_test        : étiquettes de test (peut être None)
    metadata_base : métriques du champion (depuis model_metadata.json)
    selected_model: nom du modèle sélectionné

    Retourne:
    ---------
    dict avec roc_auc_test, f1_test, recall_test, precision_test,
               optimal_threshold, model_name
    """
    from sklearn.metrics import precision_recall_curve

    if X_test is None or y_test is None:
        # Pas de données de test disponibles → métriques du champion
        return metadata_base

    try:
        # Recalcul des probabilités avec le pipeline actif
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Seuil optimal par maximisation du F1 (identique au notebook section 9)
        precs, recs, threshs = precision_recall_curve(y_test, y_prob)
        f1v = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-10)
        opt = float(threshs[np.argmax(f1v)])
        y_pred = (y_prob >= opt).astype(int)

        return {
            'model_name'       : selected_model,
            'optimal_threshold': opt,
            'roc_auc_test'     : float(roc_auc_score(y_test, y_prob)),
            'f1_test'          : float(f1_score(y_test, y_pred)),
            'recall_test'      : float(recall_score(y_test, y_pred)),
            'precision_test'   : float(precision_score(y_test, y_pred, zero_division=0)),
            # Conserver les infos de configuration du champion (identiques pour tous)
            'features_count'   : metadata_base.get('features_count', 0),
            'train_size'       : metadata_base.get('train_size', 0),
            'test_size'        : metadata_base.get('test_size', 0),
            'fraud_rate_train' : metadata_base.get('fraud_rate_train', 0),
            'num_cols'         : metadata_base.get('num_cols', []),
            'cat_cols'         : metadata_base.get('cat_cols', []),
        }
    except Exception:
        # En cas d'erreur → fallback sur les métriques du champion
        return metadata_base


# =============================================================
# APPLICATION PRINCIPALE
# =============================================================

def main():

    st.markdown("""
    <div class="main-header">
        <h1>PrédiSinistre</h1>
        <p>Système de Détection de Fraude dans les Réclamations d'Assurance</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Chargement du modèle..."):
        artifacts = load_artifacts()

    # ── Artefacts communs ──────────────────────────────────────────
    all_pipelines  = artifacts['pipelines']       # dict {nom: pipeline}
    cat_modalities = artifacts['cat_modalities']
    metadata_base  = artifacts['metadata']        # métriques du champion du notebook
    num_stats      = artifacts['num_stats']
    cat_cols       = metadata_base.get('cat_cols', [])
    num_cols       = metadata_base.get('num_cols', [])
    available_models = list(all_pipelines.keys())

    # ── Sidebar — Sélecteur de modèle + Seuil ──────────────────────
    with st.sidebar:
        st.markdown("## Configuration")

        # ── Choix du modèle ──────────────────────────────────────
        # XGBoost est le modèle par défaut (premier dans la liste)
        st.markdown("### Modèle")
        default_idx = available_models.index('XGBoost') if 'XGBoost' in available_models else 0
        selected_model = st.selectbox(
            "Modèle actif",
            options=available_models,
            index=default_idx,
            help=(
                "XGBoost est le modèle par défaut.\n"
                "Les trois modèles proviennent du notebook modelisation_de_donnees.ipynb :\n"
                "  - XGBoost (champion par défaut)\n"
                "  - HistGradientBoosting\n"
                "  - GradientBoosting"
            )
        )

        # Pipeline du modèle sélectionné
        pipeline = all_pipelines[selected_model]

        # ── Calcul dynamique des métriques du modèle actif ───────
        # get_model_metrics() recalcule les vrais indicateurs sur X_test
        # pour chaque modèle choisi dans le sélecteur.
        X_test_ref  = artifacts.get('X_test')
        y_test_ref  = artifacts.get('y_test')
        metadata = get_model_metrics(
            pipeline, X_test_ref, y_test_ref,
            metadata_base, selected_model
        )
        opt_thresh = metadata['optimal_threshold']

        st.divider()

        # ── Seuil de Décision ────────────────────────────────────
        st.markdown("### Seuil de Décision")
        threshold = st.slider(
            "Probabilité requise pour 'Fraude'",
            min_value=0.10, max_value=0.90,
            value=float(round(opt_thresh, 2)), step=0.05,
            help=f"Seuil optimal F1 pour {selected_model} : {opt_thresh:.2f}"
        )
        if threshold < 0.35:   st.warning("Seuil bas : beaucoup de faux positifs")
        elif threshold > 0.55: st.warning("Seuil élevé : risque de manquer des fraudes")
        else:                  st.success("Seuil dans la plage recommandee")

        st.divider()

        # ── Performances dynamiques du modèle sélectionné ────────
        # Ces métriques sont recalculées sur X_test pour le modèle actif.
        # Changer de modèle dans le sélecteur met à jour ces valeurs.
        st.markdown("### Performances")
        st.markdown(f"""
        <div class="sidebar-info">
            <b>Modele actif :</b> {selected_model}<br>
            <b>ROC-AUC :</b> {metadata['roc_auc_test']:.3f}<br>
            <b>F1 (fraude) :</b> {metadata['f1_test']:.3f}<br>
            <b>Recall :</b> {metadata['recall_test']:.3f}<br>
            <b>Precision :</b> {metadata['precision_test']:.3f}<br>
            <b>Seuil optimal :</b> {opt_thresh:.2f}
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.info("Pipeline ImbPipeline complet.\nSMOTE + encodage geres en interne.")

    # Onglets 
    tab1, tab2, tab3 = st.tabs([
        "Prédiction individuelle",
        "Performances et Interpretabilite",
        "Analyse par lots (CSV)"
    ])

    # ===========================================================
    # ONGLET 1 — PRÉDICTION INDIVIDUELLE
    # ===========================================================
    with tab1:
        st.markdown("## Formulaire de Réclamation")
        col1, col2, col3 = st.columns(3)
        input_data = {}

        with col1:
            st.markdown("#### Profil Assuré")
            input_data['age']                = st.number_input("Âge", 16, 100, 40)
            input_data['months_as_customer'] = st.number_input("Ancienneté (mois)", 0, 600, 120)
            for field, label in [
                ('insured_sex','Sexe'),('insured_education_level',"Niveau d'études"),
                ('insured_occupation','Profession'),('insured_hobbies','Loisirs'),
                ('insured_relationship','Lien avec le souscripteur'),
            ]:
                if field in cat_modalities:
                    input_data[field] = st.selectbox(label, cat_modalities[field])

        with col2:
            st.markdown("#### Police d'Assurance")
            for field, label in [('policy_state',"État de la police"),('policy_csl','Limite CSL')]:
                if field in cat_modalities:
                    input_data[field] = st.selectbox(label, cat_modalities[field])
            input_data['policy_deductable']     = st.selectbox("Franchise ($)", [500,1000,2000], index=1)
            input_data['policy_annual_premium'] = st.number_input("Prime annuelle ($)", 500.0, 5000.0, 1400.0, 50.0)
            input_data['umbrella_limit']        = st.selectbox("Limite parapluie ($)",
                [0,1000000,2000000,3000000,4000000,5000000,6000000,7000000,8000000,9000000,10000000])
            input_data['capital-gains'] = st.number_input("Plus-values ($)",  0, 100000, 0, 1000)
            input_data['capital-loss']  = st.number_input("Moins-values ($)", -100000, 0, 0, 1000)
            st.markdown("#### Incident")
            for field, label in [
                ('incident_type',"Type d'incident"),('collision_type','Type de collision'),
                ('incident_severity','Sévérité'),('incident_state','État (incident)'),('incident_city','Ville'),
            ]:
                if field in cat_modalities:
                    input_data[field] = st.selectbox(label, cat_modalities[field])

        with col3:
            st.markdown("#### Véhicule")
            for field, label in [('auto_make','Marque'),('auto_model','Modèle')]:
                if field in cat_modalities:
                    input_data[field] = st.selectbox(label, cat_modalities[field])
            input_data['auto_year'] = st.number_input("Année véhicule", 1990, 2025, 2010)
            st.markdown("#### Montants Réclamés")
            input_data['total_claim_amount'] = st.number_input("Total ($)",     0, 200000, 30000, 1000)
            input_data['injury_claim']       = st.number_input("Blessures ($)", 0,  50000,  5000,  500)
            input_data['property_claim']     = st.number_input("Propriété ($)", 0,  50000,  5000,  500)
            input_data['vehicle_claim']      = st.number_input("Véhicule ($)",  0, 100000, 20000, 1000)
            st.markdown("#### Compléments")
            input_data['incident_hour_of_the_day']    = st.slider("Heure de l'incident", 0, 23, 14)
            input_data['number_of_vehicles_involved'] = st.number_input("Véhicules impliqués", 1, 4, 1)
            for field, label in [
                ('property_damage','Dommages tiers'),
                ('police_report_available','Rapport de police'),
                ('authorities_contacted','Autorités contactées'),
            ]:
                if field in cat_modalities:
                    input_data[field] = st.selectbox(label, cat_modalities[field])
            input_data['bodily_injuries'] = st.number_input("Blessés", 0, 5, 0)
            input_data['witnesses']       = st.number_input("Témoins", 0, 5, 1)
            input_data['policy_age_years'] = st.number_input(
                "Ancienneté police (années)", 0.0, 30.0, 5.0, 0.5,
                help="Durée entre souscription et sinistre"
            )
            input_data['incident_month'] = st.slider("Mois de l'incident", 1, 12, 6)
            input_data['incident_dayofweek'] = st.selectbox(
                "Jour de la semaine", [0,1,2,3,4,5,6],
                format_func=lambda x: ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche'][x]
            )

        st.divider()
        _, btn_col, _ = st.columns([1,2,1])
        with btn_col:
            predict_btn = st.button("Analyser le Risque de Fraude", type="primary", use_container_width=True)

        if predict_btn:
            X_input    = build_input_row(input_data)
            fraud_prob = pipeline.predict_proba(X_input)[0, 1]
            is_fraud   = fraud_prob >= threshold

            st.divider()
            st.markdown("## Résultat de l'Analyse")
            res1, res2 = st.columns([1, 2])

            with res1:
                st.pyplot(plot_gauge(fraud_prob, threshold), use_container_width=True); plt.close()
                badge_class = "fraud-badge" if is_fraud else "legit-badge"
                badge_icon  = "FRAUDE PROBABLE" if is_fraud else "RÉCLAMATION LÉGITIME"
                st.markdown(f'<div class="prediction-badge {badge_class}">{badge_icon}<br>'
                            f'<small>P = {fraud_prob:.1%}</small></div>', unsafe_allow_html=True)

            with res2:
                st.markdown("#### Indicateurs de Risque")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("P(fraude)", f"{fraud_prob:.1%}",
                           f"{fraud_prob-threshold:+.1%} vs seuil", delta_color="inverse")
                mc2.metric("Ratio sinistre/prime",
                           f"{input_data['total_claim_amount']/max(input_data['policy_annual_premium'],1):.1f}x")
                mc3.metric("Part corporelle",
                           f"{input_data['injury_claim']/max(input_data['total_claim_amount'],1):.0%}")
                st.markdown("#### Facteurs explicatifs")
                # X_test sert de référence pour calculer médiane/mode de chaque feature.
                # Cela mesure l'écart entre la réclamation analysée et une réclamation
                # "typique" du dataset → contributions réelles et non nulles.
                X_ref = artifacts.get('X_test', X_input)
                st.pyplot(plot_local_explanation(pipeline, X_input, X_ref, n_top=10),
                          use_container_width=True); plt.close()

            st.divider()
            if fraud_prob >= 0.7:
                st.error("**Risque ÉLEVÉ** — Transmettre immédiatement à l'unité anti-fraude.")
            elif is_fraud:
                st.warning("**Risque MODÉRÉ** — Vérification complémentaire recommandée.")
            elif fraud_prob >= threshold * 0.7:
                st.info("**Risque FAIBLE** — Traitement standard avec attention particulière.")
            else:
                st.success("**Faible risque** — Traitement standard recommandé.")

    # ===========================================================
    # ONGLET 2 — PERFORMANCES & INTERPRÉTABILITÉ
    # ===========================================================
    with tab2:
        st.markdown("## Performances et Interprétabilité du Modèle")

        # ── Métriques colorées ──────────────────────────────────
        st.markdown("### Métriques clés sur le Test Set")
        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, css_class, desc in [
            (m1, "ROC-AUC",          metadata['roc_auc_test'],   "metric-card-auc",      "Discrimination globale"),
            (m2, "F1-Score (fraude)", metadata['f1_test'],        "metric-card-f1",       "Equilibre precision/recall"),
            (m3, "Recall (fraude)",   metadata['recall_test'],    "metric-card-recall",   "Fraudes detectees / total"),
            (m4, "Precision",         metadata['precision_test'], "metric-card-precision","Alertes vraiment frauduleuses"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card {css_class}">
                    <div class="label">{label}</div>
                    <div class="value">{val:.3f}</div>
                    <div class="sub">{val:.1%} — {desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Config + Courbe ROC ─────────────────────────────────
        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown("### Configuration du Modèle")
            st.markdown(f"""
            <div class="model-config-card">
                <div class="config-title">Paramètres du Pipeline</div>
                <div class="config-row">
                    <span class="config-key">Algorithme</span>
                    <span class="config-value">{metadata['model_name']}</span>
                </div>
                <div class="config-row">
                    <span class="config-key">Donnees d'entrainement</span>
                    <span class="config-value">{metadata['train_size']} réclamations</span>
                </div>
                <div class="config-row">
                    <span class="config-key">Donnees de test</span>
                    <span class="config-value">{metadata['test_size']} réclamations</span>
                </div>
                <div class="config-row">
                    <span class="config-key">Taux fraude (train)</span>
                    <span class="config-value">{metadata['fraud_rate_train']:.1%}</span>
                </div>
                <div class="config-row">
                    <span class="config-key">Nombre de features</span>
                    <span class="config-value">{metadata['features_count']}</span>
                </div>
                <div class="config-row">
                    <span class="config-key">Seuil optimal F1</span>
                    <span class="config-value">{metadata['optimal_threshold']:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with pc2:
            if 'y_test' in artifacts and X_test_ref is not None:
                st.markdown("### Courbe ROC")
                # Recalcul des probabilités avec le pipeline actif (pas le champion fixe)
                y_prob_active = pipeline.predict_proba(X_test_ref)[:, 1]
                fpr, tpr, _ = roc_curve(artifacts['y_test'], y_prob_active)
                auc = roc_auc_score(artifacts['y_test'], y_prob_active)
                fig_roc, ax = plt.subplots(figsize=(6, 4))
                fig_roc.patch.set_facecolor('#1a2a3a'); ax.set_facecolor('#1a2a3a')
                ax.plot(fpr, tpr, color='#3498db', lw=2.5, label=f'ROC (AUC={auc:.3f})')
                ax.plot([0,1],[0,1],'w--', alpha=0.4, label='Aleatoire')
                ax.fill_between(fpr, tpr, alpha=0.15, color='#3498db')
                ax.set_xlabel('Taux Faux Positifs', color='white')
                ax.set_ylabel('Recall', color='white')
                ax.set_title('Courbe ROC — Test Set', fontweight='bold', color='white')
                ax.tick_params(colors='white')
                ax.legend(facecolor='#1a2a3a', labelcolor='white')
                ax.grid(alpha=0.2, color='white')
                for spine in ax.spines.values(): spine.set_edgecolor('#444')
                plt.tight_layout()
                st.pyplot(fig_roc, use_container_width=True); plt.close()

        st.divider()

        # ── SECTION SHAP ────────────────────────────────────────
        # Reproduit exactement la section 10 du notebook :
        # X_test_tr = best_pipeline["preprocessor"].transform(X_test)
        # shap_explainer = shap.TreeExplainer(model=best_pipeline["classifier"])
        # shap_values = shap_explainer(X_test_tr)
        # ────────────────────────────────────────────────────────
        st.markdown("### Interprétabilité SHAP")

        if not SHAP_AVAILABLE:
            st.warning(
                "SHAP non installe. Installer avec : pip install shap\n\n"
                "Les graphiques SHAP seront disponibles apres installation."
            )
        else:
            # Récupérer X_test et feature_names depuis les artefacts
            X_test_ref  = artifacts.get('X_test')
            y_test_ref  = artifacts.get('y_test')
            feat_names  = artifacts.get('feature_names', [])

            if X_test_ref is None:
                st.info(
                    "X_test non disponible pour les graphiques SHAP.\n"
                    "Ajouter 'X_test' et 'feature_names' dans load_artifacts() "
                    "pour activer cette section."
                )
            else:
                # Transformation de X_test via le préprocesseur du Pipeline
                # Identique à : X_test_tr = best_pipeline["preprocessor"].transform(X_test)
                @st.cache_data
                def compute_shap_values(_pipeline, _X_test_ref):
                    """
                    Calcule les valeurs SHAP une seule fois et les met en cache.
                    Utilise TreeExplainer — optimisé pour les modèles d'arbres
                    (XGBoost, GradientBoosting, HistGradientBoosting).
                    """
                    X_test_tr = _pipeline["preprocessor"].transform(_X_test_ref)
                    explainer  = shap.TreeExplainer(model=_pipeline["classifier"])
                    sv = explainer(X_test_tr)
                    return X_test_tr, sv

                with st.spinner("Calcul des valeurs SHAP..."):
                    try:
                        X_test_tr, shap_values = compute_shap_values(pipeline, X_test_ref)

                        # ── 1. SHAP Summary Plot (beeswarm) ──────────────────
                        # Identique au notebook : shap.summary_plot(shap_values, X_test_tr, ...)
                        # Chaque point = une réclamation
                        # Position horizontale = impact sur P(fraude)
                        # Couleur = valeur de la feature (rouge=haute, bleue=basse)
                        st.markdown("#### SHAP Summary Plot")
                        st.caption(
                            "Chaque point represente une reclamation. "
                            "La position indique l'impact sur la probabilite de fraude. "
                            "La couleur indique la valeur de la feature (rouge = élevée, bleu = faible)."
                        )
                        plt.close('all')
                        shap.summary_plot(
                            shap_values=shap_values,
                            features=X_test_tr,
                            feature_names=feat_names if feat_names else None,
                            max_display=15,
                            show=False,
                            plot_size=(9, 6)
                        )
                        fig_summary = plt.gcf()
                        fig_summary.patch.set_facecolor('#1a2a3a')
                        st.pyplot(fig_summary, use_container_width=True)
                        plt.close('all')

                        st.divider()

                        # ── 2. SHAP Bar Plot (importance globale) ─────────────
                        # Importance moyenne absolue des valeurs SHAP sur tout X_test
                        st.markdown("#### SHAP Bar Plot — Importance Globale")
                        st.caption(
                            "Importance moyenne absolue des valeurs SHAP sur l'ensemble du test set. "
                            "Equivalent à feature_importance mais plus fiable car basé sur les predictions."
                        )
                        plt.close('all')
                        shap.summary_plot(
                            shap_values=shap_values,
                            features=X_test_tr,
                            feature_names=feat_names if feat_names else None,
                            plot_type='bar',
                            max_display=15,
                            show=False,
                            plot_size=(9, 5)
                        )
                        fig_bar = plt.gcf()
                        fig_bar.patch.set_facecolor('#1a2a3a')
                        st.pyplot(fig_bar, use_container_width=True)
                        plt.close('all')

                        st.divider()

                        # ── 3. SHAP Waterfall & Force — par réclamation ───────
                        # Identique au notebook section 10 :
                        # shap.plots.waterfall(shap_values[i])
                        # shap.plots.force(shap_values[i])
                        st.markdown("#### SHAP par réclamation individuelle")
                        st.caption(
                            "Selectionnez une reclamation pour voir quelles features "
                            "ont pousse la prediction vers fraude ou legitime."
                        )

                        # Sélecteur d'instance
                        n_instances = len(X_test_tr)
                        col_sel1, col_sel2 = st.columns([2, 1])
                        with col_sel1:
                            if y_test_ref is not None:
                                instance_type = st.radio(
                                    "Type de reclamation a analyser",
                                    ["Fraude (y=1)", "Legitime (y=0)", "Index manuel"],
                                    horizontal=True
                                )
                            else:
                                instance_type = "Index manuel"

                        with col_sel2:
                            if instance_type == "Index manuel" or y_test_ref is None:
                                chosen_idx = st.number_input(
                                    "Index (0 a N-1)", 0, n_instances - 1, 0
                                )
                            elif instance_type == "Fraude (y=1)":
                                fraud_list = [
                                    i for i, v in enumerate(y_test_ref.values) if v == 1
                                ][:10]
                                chosen_idx = st.selectbox("Exemple de fraude", fraud_list)
                            else:
                                legit_list = [
                                    i for i, v in enumerate(y_test_ref.values) if v == 0
                                ][:10]
                                chosen_idx = st.selectbox("Exemple legitime", legit_list)

                        chosen_idx = int(chosen_idx)

                        col_wf, col_force = st.columns(2)

                        with col_wf:
                            # Waterfall plot — décomposition de la prédiction
                            st.markdown(f"**Waterfall — instance #{chosen_idx}**")
                            plt.close('all')
                            shap.plots.waterfall(shap_values[chosen_idx], max_display=12, show=False)
                            fig_wf = plt.gcf()
                            fig_wf.patch.set_facecolor('#1a2a3a')
                            st.pyplot(fig_wf, use_container_width=True)
                            plt.close('all')

                        with col_force:
                            # Bar plot individuel (équivalent force plot en matplotlib)
                            # Identique à shap.plots.force(shap_values[i]) du notebook
                            st.markdown(f"**Contribution par feature — instance #{chosen_idx}**")
                            plt.close('all')
                            shap.plots.bar(shap_values[chosen_idx], max_display=12, show=False)
                            fig_force = plt.gcf()
                            fig_force.patch.set_facecolor('#1a2a3a')
                            st.pyplot(fig_force, use_container_width=True)
                            plt.close('all')

                    except Exception as e:
                        st.error(f"Erreur lors du calcul SHAP : {e}")
                        st.exception(e)

        st.divider()

        # ── Permutation Importance ──────────────────────────────
        # Identique au notebook section 10 : permutation_importance(best_pipeline, X_test, ...)
        st.markdown("### Permutation Importance")
        st.caption(
            "Mesure la degradation du ROC-AUC quand on melange les valeurs d'une feature. "
            "Plus robuste que la feature importance native car base sur les performances reelles."
        )

        X_test_perm = artifacts.get('X_test')
        y_test_perm = artifacts.get('y_test')

        if X_test_perm is not None and y_test_perm is not None:
            if st.button("Calculer la Permutation Importance (peut prendre 1-2 min)"):
                from sklearn.inspection import permutation_importance as perm_imp_fn
                with st.spinner("Calcul en cours (n_repeats=20)..."):
                    perm = perm_imp_fn(
                        pipeline, X_test_perm, y_test_perm,
                        n_repeats=20, random_state=42,
                        scoring='roc_auc', n_jobs=-1
                    )
                perm_df = (
                    pd.DataFrame({
                        'feature'        : X_test_perm.columns.tolist(),
                        'importance_mean': perm.importances_mean,
                        'importance_std' : perm.importances_std
                    })
                    .sort_values('importance_mean', ascending=False)
                    .head(20)
                    .sort_values('importance_mean')
                )
                fig_perm, ax_perm = plt.subplots(figsize=(9, 7))
                fig_perm.patch.set_facecolor('#1a2a3a'); ax_perm.set_facecolor('#1a2a3a')
                ax_perm.barh(
                    perm_df['feature'], perm_df['importance_mean'],
                    xerr=perm_df['importance_std'],
                    color='darkorange', edgecolor='white', alpha=0.85, capsize=4
                )
                ax_perm.axvline(x=0, color='#e74c3c', ls='--', alpha=0.5)
                ax_perm.set_xlabel('Delta ROC-AUC apres permutation', color='white')
                ax_perm.set_title(
                    f'Permutation Importance (n=20) — {metadata["model_name"]}',
                    fontweight='bold', color='white'
                )
                ax_perm.tick_params(colors='white')
                ax_perm.grid(axis='x', alpha=0.2, color='white')
                for spine in ax_perm.spines.values(): spine.set_edgecolor('#444')
                plt.tight_layout()
                st.pyplot(fig_perm, use_container_width=True); plt.close()
        else:
            st.info("X_test non disponible pour la Permutation Importance.")

        st.divider()

        # ── Guide d'interprétation ──────────────────────────────
        st.markdown("### Guide d'Interprétation")
        g1, g2 = st.columns(2)
        with g1:
            for title, items in [
                ("Signaux financiers suspects",
                 ["Ratio sinistre/prime **> 20x**",
                  "Part corporelle **> 50%** sans hospitalisation",
                  "Montant total incoherent avec les dommages"]),
                ("Signaux temporels suspects",
                 ["Police **tres recente** avant le sinistre",
                  "Incident en **nuit ou week-end** sans temoins",
                  "Anciennete police **< 1 an** (policy_age_years faible)"]),
            ]:
                items_html = "".join(f"• {i}<br>" for i in items)
                st.markdown(
                    f'<div class="guide-card"><b>{title}</b><br><br>{items_html}</div>',
                    unsafe_allow_html=True
                )
        with g2:
            for title, items in [
                ("Signaux administratifs suspects",
                 ["Rapport de police **absent** pour sinistre important",
                  "Autorites **non contactees**",
                  "Dommages tiers **non documentes**"]),
                ("Rappel RGPD",
                 ["Toute decision automatique doit etre **explicable**",
                  "Droit au **recours humain** (article 22 RGPD)"]),
            ]:
                items_html = "".join(f"• {i}<br>" for i in items)
                st.markdown(
                    f'<div class="guide-card"><b>{title}</b><br><br>{items_html}</div>',
                    unsafe_allow_html=True
                )

    # ===========================================================
    # ONGLET 3 — ANALYSE PAR LOTS (CSV)
    # ===========================================================
    with tab3:
        st.markdown("## Analyse par Lots")
        st.markdown("Chargez un fichier CSV au format `insurance_claims.csv` pour analyser l'ensemble des dossiers.")

        uploaded = st.file_uploader("Fichier CSV", type=['csv'])

        if uploaded:
            try:
                # Chargement brut
                df_raw = pd.read_csv(uploaded)
                st.success(f"{len(df_raw)} réclamations chargées")

                with st.expander("Colonnes détectées dans le fichier"):
                    st.write(df_raw.columns.tolist())

                # Vérification IMMÉDIATE après chargement, avant tout preprocessing
                required_cols = [
                    'total_claim_amount', 'policy_annual_premium',
                    'injury_claim', 'vehicle_claim', 'policy_deductable'
                ]
                missing = [c for c in required_cols if c not in df_raw.columns]
                if missing:
                    st.error(
                        f"Colonnes manquantes : **{missing}**\n\n"
                        "Vérifiez que le CSV est au format `insurance_claims.csv`."
                    )
                    st.stop()

                # Preprocessing complet (dates, imputation, ratios)
                X_batch = preprocess_batch(df_raw)

                # Prédiction via le Pipeline — X brut, encodage géré en interne
                probs = pipeline.predict_proba(X_batch)[:, 1]
                preds = (probs >= threshold).astype(int)

                n_fraud = preds.sum()
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total dossiers", len(preds))
                b2.metric("Fraudes",      n_fraud, f"{n_fraud/len(preds):.1%}")
                b3.metric("Légitimes",     len(preds)-n_fraud)
                b4.metric("Score moyen",    f"{probs.mean():.3f}")

                fig_d, ax_d = plt.subplots(figsize=(8, 4))
                fig_d.patch.set_facecolor('#1a2a3a'); ax_d.set_facecolor('#1a2a3a')
                ax_d.hist(probs, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
                ax_d.axvline(threshold, color='#e74c3c', ls='--', lw=2, label=f'Seuil ({threshold:.2f})')
                ax_d.set(xlabel='P(fraude)', ylabel='Dossiers', title='Distribution des scores')
                ax_d.title.set_color('white')
                ax_d.xaxis.label.set_color('white'); ax_d.yaxis.label.set_color('white')
                ax_d.tick_params(colors='white')
                ax_d.legend(facecolor='#1a2a3a', labelcolor='white')
                ax_d.grid(alpha=0.2, color='white')
                for spine in ax_d.spines.values(): spine.set_edgecolor('#444')
                plt.tight_layout()
                st.pyplot(fig_d, use_container_width=True); plt.close()

                df_out = df_raw.copy()
                df_out['P_fraude']   = probs.round(3)
                df_out['Prédiction'] = np.where(preds==1,'FRAUDE','LÉGITIME')
                df_out['Risque']     = pd.cut(probs, bins=[0,.3,.5,.7,1.],
                                              labels=['Faible','Modéré','Élevé','Très élevé'])
                cols_show = [c for c in ['P_fraude','Prédiction','Risque','total_claim_amount',
                                         'policy_annual_premium','incident_type','incident_severity']
                             if c in df_out.columns]
                st.markdown("#### Dossiers classés par risque décroissant")
                st.dataframe(df_out[cols_show].sort_values('P_fraude', ascending=False),
                             use_container_width=True, height=400)

                st.download_button("Télécharger les résultats (CSV)",
                                   df_out.to_csv(index=False).encode('utf-8'),
                                   "predictions_fraude.csv", "text/csv")

            except Exception as e:
                st.error(f" Erreur : {e}")
                st.exception(e)
        else:
            st.info(
                "Déposez un fichier CSV au format `insurance_claims.csv`.\n\n"
                "Le preprocessing complet (dates, imputation, ratios) est appliqué automatiquement."
            )


if __name__ == "__main__":
    main()