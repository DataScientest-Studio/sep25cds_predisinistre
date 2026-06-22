"""
=============================================================
PrédiSinistre — Application Streamlit de Détection de Fraude
=============================================================
 
Lancement :
    poetry run streamlit run streamlit_app.py
 
Dépendances :
    poetry add streamlit scikit-learn pandas numpy matplotlib seaborn joblib
 
Fichiers requis (dans le même dossier) :
    best_pipeline.pkl   — Pipeline complet (préprocessing + modèle)
    cat_modalities.pkl  — Modalités catégorielles (depuis X_train)
    model_metadata.json — Métriques et configuration
    num_stats.json      — Stats numériques du train set
"""
 
# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import joblib, json, os, warnings
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
 
warnings.filterwarnings('ignore')
 
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
# PREPROCESSING — identique au notebook modelisation
# =============================================================
 
def preprocess_batch(df_raw):
    """
    Applique le même preprocessing que le notebook modelisation.
    Appelé sur le CSV uploadé dans l'onglet 'Analyse par lots'.
 
    Ordre identique au notebook :
      1. Suppression colonnes inutiles
      2. Transformation dates → policy_age_years, incident_month, incident_dayofweek
      3. Remplacement '?' → NaN
      4. Correction umbrella_limit (abs)
      5. Imputation
      6. Feature Engineering (ratios)
      7. Suppression de fraud_reported si présente
    """
    df = df_raw.copy()
 
    # 1. Suppression colonnes inutiles
    cols_to_drop = ['_c39', 'policy_number', 'incident_location', 'insured_zip']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
 
    # 2. Transformation des dates
    # Si les colonnes de dates brutes sont présentes, on les convertit en features numériques.
    # Si elles sont absentes (CSV déjà préprocessé), on crée des colonnes à 0 par défaut
    # pour que le pipeline reçoive toujours les mêmes features.
    if 'policy_bind_date' in df.columns and 'incident_date' in df.columns:
        df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
        df['incident_date']    = pd.to_datetime(df['incident_date'],    errors='coerce')
        df['policy_age_years']   = ((df['incident_date'] - df['policy_bind_date']).dt.days / 365.25).round(2)
        df['incident_month']     = df['incident_date'].dt.month
        df['incident_dayofweek'] = df['incident_date'].dt.dayofweek
        df.drop(columns=['policy_bind_date', 'incident_date'], inplace=True)
    else:
        # Valeurs par défaut si les dates sont absentes
        if 'policy_age_years'   not in df.columns: df['policy_age_years']   = 5.0
        if 'incident_month'     not in df.columns: df['incident_month']     = 6
        if 'incident_dayofweek' not in df.columns: df['incident_dayofweek'] = 2
 
    # 3. Remplacement '?' → NaN
    df.replace('?', np.nan, inplace=True)
 
    # 4. Correction umbrella_limit
    if 'umbrella_limit' in df.columns:
        df['umbrella_limit'] = df['umbrella_limit'].abs()
 
    # 5. Imputation
    if 'authorities_contacted' in df.columns:
        df['authorities_contacted'] = df['authorities_contacted'].fillna('None')
    for col in ['collision_type', 'property_damage', 'police_report_available']:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val):
                df[col] = df[col].fillna(mode_val[0])
 
    # 6. Feature Engineering
    if 'total_claim_amount' in df.columns:
        df['claim_ratio']  = df['total_claim_amount'] / df['policy_annual_premium'].replace(0, np.nan).fillna(1)
        df['injury_share'] = df['injury_claim']  / df['total_claim_amount'].replace(0, np.nan).fillna(1)
        df['vehicle_share']= df['vehicle_claim'] / df['total_claim_amount'].replace(0, np.nan).fillna(1)
    if 'policy_deductable' in df.columns:
        df['is_high_deductable'] = (df['policy_deductable'] > 1000).astype(int)
 
    # 7. Suppression de la cible si présente
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
    Charge le Pipeline sklearn et les métadonnées.
    Si les fichiers .pkl sont absents, entraîne un pipeline de démonstration.
    """
    # à vérifier
    required = ['best_pipeline_xgb.pkl', 'best_pipeline_hgb.pkl', 'best_pipeline_gb.pkl', 
                'cat_modalities.pkl','model_metadata.json', 'num_stats.json', "X_train.csv",
                'y_test.pkl', 'y_prob_xgb.pkl', 'y_prob_hgb.pkl', 'y_prob_gb.pkl']
 
    if all(os.path.exists(f) for f in required):
        pipeline       = joblib.load('best_pipeline_xgb.pkl') # modèle champion (XGBoost)
        cat_modalities = joblib.load('cat_modalities.pkl')
        y_test = joblib.load('y_test.pkl')
        y_prob_xgb = joblib.load('y_prob_xgb.pkl')
        y_prob_hgb = joblib.load('y_prob_hgb.pkl')
        y_prob_gb = joblib.load('y_prob_gb.pkl')
        with open('model_metadata.json') as f: metadata  = json.load(f)
        with open('num_stats.json')      as f: num_stats = json.load(f)
        X_train = pd.read_csv('X_train.csv')
        return {'pipeline': pipeline, 'cat_modalities': cat_modalities,
                'metadata': metadata, 'num_stats': num_stats, 'X_train': X_train,
                'y_test': y_test, 'y_prob_xgb': y_prob_xgb, 'y_prob_hgb': y_prob_hgb, 'y_prob_gb': y_prob_gb}
    else:
        return _train_demo_pipeline()
 
 
 # Pipeline de démonstration si les .pkl sont absents. Entraîne un modèle simple sur le dataset original.
def _train_demo_pipeline():
    """Pipeline de démonstration si les .pkl sont absents."""
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import precision_recall_curve
 
    for path in ['insurance_claims.csv', '/mnt/project/insurance_claims.csv',
                 '../data/raw/insurance_claims.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path); break
    else:
        st.error("insurance_claims.csv introuvable.")
        st.stop()
 
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
    for col in ['collision_type','property_damage','police_report_available']:
        for g in [0,1]:
            mask_g = df['fraud_reported'] == g
            mask_n = df[col].isnull() & mask_g
            if mask_n.sum() > 0:
                mode_val = df.loc[mask_g, col].mode()
                if len(mode_val): df.loc[mask_n, col] = mode_val[0]
 
    X_raw = df.drop(columns=['fraud_reported'])
    y     = df['fraud_reported']
    X_raw['claim_ratio']        = X_raw['total_claim_amount'] / X_raw['policy_annual_premium'].replace(0,np.nan).fillna(1)
    X_raw['injury_share']       = X_raw['injury_claim'] / X_raw['total_claim_amount'].replace(0,np.nan).fillna(1)
    X_raw['vehicle_share']      = X_raw['vehicle_claim'] / X_raw['total_claim_amount'].replace(0,np.nan).fillna(1)
    X_raw['is_high_deductable'] = (X_raw['policy_deductable'] > 1000).astype(int)
 
    cat_cols = X_raw.select_dtypes(include='object').columns.tolist()
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
 
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(drop=None, handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                       max_depth=4, class_weight='balanced', random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:,1]
    precs, recs, threshs = precision_recall_curve(y_test, y_prob)
    f1v = 2*precs[:-1]*recs[:-1]/(precs[:-1]+recs[:-1]+1e-10)
    opt = float(threshs[np.argmax(f1v)])
    yp  = (y_prob >= opt).astype(int)
    pp  = preprocessor.fit(X_train)
    ohe_names = pp.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    cat_modalities = {col: sorted(X_train[col].dropna().unique().tolist()) for col in cat_cols}
    num_stats = {col: {'min':float(X_train[col].min()),'max':float(X_train[col].max()),
                       'mean':float(X_train[col].mean()),'std':float(X_train[col].std())}
                 for col in num_cols if col in X_train.columns}
    metadata = {
        'model_name':'HistGradientBoosting','optimal_threshold':opt,
        'roc_auc_test':float(roc_auc_score(y_test,y_prob)),
        'f1_test':float(f1_score(y_test,yp)),'recall_test':float(recall_score(y_test,yp)),
        'precision_test':float(precision_score(y_test,yp,zero_division=0)),
        'features_count':len(num_cols)+len(ohe_names),
        'train_size':len(y_train),'test_size':len(y_test),'fraud_rate_train':float(y_train.mean()),
        'num_cols':num_cols,'cat_cols':cat_cols
    }
    return {'pipeline':pipeline,'cat_modalities':cat_modalities,'metadata':metadata,'num_stats':num_stats,
            'X_test':X_test,'y_test':y_test,'y_prob':y_prob}
 
 
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
                   .sort_values('contribution', key=abs, ascending=False).head(n_top).sort_values('contribution')
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
 
def plot_shap_waterfall(pipeline, X_input):
    """
    Explication SHAP d'une prédiction individuelle.

    Compatible :
        - XGBoost
        - HistGradientBoosting
        - GradientBoosting
    """

    try:
        # Préprocessing identique à l'entraînement
        X_transformed = pipeline["preprocessor"].transform(X_input)
        
        # Capture des noms de features après OneHotEncoding
        feature_names = (pipeline["preprocessor"].get_feature_names_out())
        
        # Nettoyage des noms de features pour SHAP (remplace 'cat__' et 'num__' par '')
        feature_names = [c.replace("num__", "").replace("cat__", "") for c in feature_names]
        
        # Transformation des colonnes en DataFrame pour SHAP
        X_transformed = pd.DataFrame(X_transformed,columns=feature_names)

        # Modèle final
        model = pipeline["classifier"]

        # Explainer SHAP
        explainer = shap.Explainer(model)

        shap_values = explainer(X_transformed)

        fig = plt.figure(figsize=(10, 6))

        shap.plots.waterfall(
            shap_values[0],
            max_display=12,
            show=False
        )

        return fig

    except Exception as e:
        st.warning(
            f"Impossible de calculer SHAP pour ce modèle : {e}"
        )
        return None
# plot_global_importance intentionnellement supprimée
# (causait une erreur d'affichage dans Streamlit)

def plot_confusion_thresholds(
        y_true,
        y_prob,
        optimal_threshold,
        model_name):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5)
    )

    thresholds = [
        (0.50, "Seuil par défaut"),
        (optimal_threshold, "Seuil optimisé")
    ]

    for ax, (thr, title) in zip(axes, thresholds):

        y_pred = (y_prob >= thr).astype(int)

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        f1 = f1_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        prec = precision_score(
            y_true,
            y_pred
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            cbar=True
        )

        ax.set_title(
            f"{title} ({thr:.2f})\n"
            f"F1={f1:.3f} | "
            f"Recall={rec:.3f} | "
            f"Précision={prec:.3f}",
            fontweight="bold"
        )

        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")

    fig.suptitle(
        f"Matrices de Confusion — {model_name}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    return fig
 
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
 
    
    metadata = artifacts['metadata']

    MODEL_REGISTRY = {
    "XGBoost": {
        "file": "best_pipeline_xgb.pkl",
        "roc_auc": metadata["roc_auc_test_xgb"],
        "f1": metadata["f1_test_xgb"],
        "recall": metadata["recall_test_xgb"],
        "precision": metadata["precision_test_xgb"],
        "threshold": metadata["optimal_threshold_xgb"]
    },
    "HistGradientBoosting": {
        "file": "best_pipeline_hgb.pkl",
        "roc_auc": metadata["roc_auc_test_hgb"],
        "f1": metadata["f1_test_hgb"],
        "recall": metadata["recall_test_hgb"],
        "precision": metadata["precision_test_hgb"],
        "threshold": metadata["optimal_threshold_hgb"]
    },
    "GradientBoosting": {
        "file": "best_pipeline_gb.pkl",
        "roc_auc": metadata["roc_auc_test_gb"],
        "f1": metadata["f1_test_gb"],
        "recall": metadata["recall_test_gb"],
        "precision": metadata["precision_test_gb"],
        "threshold": metadata["optimal_threshold_gb"]
    }}

    cat_modalities = artifacts['cat_modalities']
    num_stats      = artifacts['num_stats']
    cat_cols       = metadata.get('cat_cols', [])
    num_cols       = metadata.get('num_cols', [])
 
    # Sidebar 
    with st.sidebar:
        st.markdown("## Configuration")
        selected_model = st.selectbox("Modèle de prédiction",
        list(MODEL_REGISTRY.keys()),
        index=0
        )

        cfg = MODEL_REGISTRY[selected_model]
        
        if selected_model == "XGBoost":
            y_prob = artifacts["y_prob_xgb"]

        elif selected_model == "HistGradientBoosting":
            y_prob = artifacts["y_prob_hgb"]

        else:
            y_prob = artifacts["y_prob_gb"]

        y_test = artifacts["y_test"]

        pipeline = joblib.load(cfg["file"])
        opt_thresh = cfg["threshold"]
        st.markdown("### Seuil de Décision")
        threshold = st.slider(
            "Probabilité requise pour 'Fraude'",
            min_value=0.10, max_value=0.90,
            value=float(round(opt_thresh, 2)), step=0.05,
            help=f"Seuil optimal F1 : {opt_thresh:.2f}"
        )
        if threshold < 0.35:   st.warning("Seuil bas : beaucoup de faux positifs")
        elif threshold > 0.55: st.warning("Seuil élevé : risque de manquer des fraudes")
        else:                  st.success("Seuil dans la plage recommandée")
        st.divider()
        st.markdown("### Performances")
        st.markdown(f"""
        <div class="sidebar-info">
            <b>Modèle :</b> {selected_model}<br>
            <b>ROC-AUC :</b> {cfg['roc_auc']:.3f}<br>
            <b>F1 (fraude) :</b> {cfg['f1']:.3f}<br>
            <b>Recall :</b> {cfg['recall']:.3f}<br>
            <b>Précision :</b> {cfg['precision']:.3f}<br>
            <b>Seuil optimal :</b> {cfg['threshold']:.2f}
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.info("Pipeline sklearn complet.\nDonnées brutes passées directement.")
 
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
            
            st.session_state["last_prediction_input"] = X_input.copy()
            
            # Sauvegarde de la dernière observation
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
                st.pyplot(plot_local_explanation(pipeline, X_input, artifacts['X_train'], n_top=10),
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

        
        st.markdown("### Matrice de confusion")

        fig_cm = plot_confusion_thresholds(y_test, y_prob, cfg["threshold"], selected_model)
        st.pyplot(fig_cm,use_container_width=True)
        
        st.markdown("### Métriques clés sur le Test Set")
        
        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, css_class, desc in [
            (m1, "ROC-AUC",          cfg['roc_auc'],   "metric-card-auc",      "Discrimination globale"),
            (m2, "F1-Score (fraude)", cfg['f1'],        "metric-card-f1",       "Équilibre précision/recall"),
            (m3, "Recall (fraude)",   cfg['recall'],    "metric-card-recall",   "Fraudes détectées / total"),
            (m4, "Précision",         cfg['precision'], "metric-card-precision","Alertes vraiment frauduleuses"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card {css_class}">
                    <div class="label">{label}</div>
                    <div class="value">{val:.3f}</div>
                    <div class="sub">{val:.1%} — {desc}</div>
                </div>""", unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        pc1, pc2 = st.columns(2)
 
        
        st.markdown("### Importance Globale des Variables (SHAP)")
        def plot_global_shap(pipeline, X_reference):

            X_tr = pipeline["preprocessor"].transform(X_reference)

            feature_names = (
                pipeline["preprocessor"]
                .get_feature_names_out()
            )

            feature_names = [
                c.replace("num__", "")
                .replace("cat__", "")
                for c in feature_names
            ]

            X_tr = pd.DataFrame(
                X_tr,
                columns=feature_names
            )

            model = pipeline["classifier"]

            explainer = shap.Explainer(model)

            shap_values = explainer(X_tr)

            fig = plt.figure(figsize=(10, 6))

            shap.summary_plot(
                shap_values,
                X_tr,
                show=False,
                max_display=20
            )

            return fig
        
        if 'X_train' in artifacts:

            X_ref = artifacts["X_train"].sample(
                min(200, len(artifacts["X_train"])),
                random_state=42
            )

            fig_global = plot_global_shap(
                pipeline,
                X_ref
            )

            st.pyplot(
                fig_global,
                use_container_width=True
            )
            
        with pc1:
            st.markdown("### Configuration du Modèle")
            st.markdown(f"""
            <div class="model-config-card">
                <div class="config-title">Paramètres du Pipeline</div>
                <div class="config-row">
                    <span class="config-key">Algorithme</span>
                    <span class="config-value">{selected_model}</span>
                </div>
                <div class="config-row">
                    <span class="config-key">Données d'entraînement</span>
                    <span class="config-value">{metadata['train_size']} réclamations</span>
                </div>
                <div class="config-row">
                    <span class="config-key">Données de test</span>
                    <span class="config-value">{metadata['test_size']} réclamations</span>
                </div>
                <div class="config-row">
                    <span class="config-key">Taux fraude (train)</span>
                    <span class="config-value">{metadata['fraud_rate_train']:.1%}</span>
                </div>
                <div class="config-row">
                    <span class="config-key">Nombre de features</span>
                </div>
                <div class="config-row">
                    <span class="config-key">Seuil optimal F1</span>
                    <span class="config-value">{cfg['threshold']:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
 
            if 'y_test' in artifacts and 'y_prob' in artifacts:
                st.markdown("### Courbe ROC")
                fpr, tpr, _ = roc_curve(artifacts['y_test'], artifacts['y_prob'])
                auc = roc_auc_score(artifacts['y_test'], artifacts['y_prob'])
                fig_roc, ax = plt.subplots(figsize=(6, 4))
                fig_roc.patch.set_facecolor('#1a2a3a'); ax.set_facecolor('#1a2a3a')
                ax.plot(fpr, tpr, color='#3498db', lw=2.5, label=f'ROC (AUC={auc:.3f})')
                ax.plot([0,1],[0,1],'w--', alpha=0.4, label='Aléatoire')
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
 
        with pc2:
            st.markdown("### Guide d'Interprétation")
            for title, icon, items in [
                ("Signaux financiers suspects", "",
                 ["Ratio sinistre/prime **> 20x**",
                  "Part corporelle **> 50%** sans hospitalisation",
                  "Montant total incohérent avec les dommages"]),
                ("Signaux temporels suspects", "",
                 ["Police **très récente** avant le sinistre",
                  "Incident en **nuit ou week-end** sans témoins",
                  "Ancienneté client **< 1 an**"]),
                ("Signaux administratifs suspects", "",
                 ["Rapport de police **absent** pour sinistre important",
                  "Autorités **non contactées**",
                  "Dommages tiers **non documentés**"]),
                ("Rappel RGPD", "",
                 ["Toute décision automatique doit être **explicable**",
                  "Droit au **recours humain** (article 22 RGPD)"]),
            ]:
                items_html = "".join(f"• {i}<br>" for i in items)
                st.markdown(f"""
                <div class="guide-card">
                    <b>{icon} {title}</b><br><br>{items_html}
                </div>""", unsafe_allow_html=True)
            
            st.markdown("---")

            st.markdown("### Explication SHAP d'une prédiction")

            if "last_prediction_input" in st.session_state:

                fig_shap = plot_shap_waterfall(
                    pipeline,
                    st.session_state["last_prediction_input"]
                )

                if fig_shap:
                    st.pyplot(
                        fig_shap,
                        use_container_width=True
                    )

            else:

                st.info(
                    "Effectuez d'abord une prédiction dans l'onglet "
                    "'Prédiction individuelle' pour visualiser "
                    "l'explication SHAP."
                )   
                
            
 
    # ===========================================================
    # ONGLET 3 — ANALYSE PAR LOTS (CSV)
    # ===========================================================
    with tab3:
        st.markdown("## Analyse par Lots")
        st.markdown(
            "Chargez un fichier CSV au format `insurance_claims.csv` pour analyser l'ensemble des dossiers. "
            "Cette fonctionnalité permet de traiter en une fois tous les sinistres reçus "
            "(ex. file d'attente quotidienne) au lieu de les saisir un par un."
        )

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

                # ── Validation de la qualité des données ────────────────
                # Détecte les valeurs aberrantes ou incohérentes avant la
                # prédiction, pour éviter de fausser silencieusement les scores
                quality_issues = []
                if 'age' in df_raw.columns:
                    n_bad_age = ((df_raw['age'] < 16) | (df_raw['age'] > 100)).sum()
                    if n_bad_age > 0:
                        quality_issues.append(f"{n_bad_age} dossier(s) avec un âge hors plage (16-100 ans)")
                if 'total_claim_amount' in df_raw.columns:
                    n_neg_claim = (df_raw['total_claim_amount'] < 0).sum()
                    if n_neg_claim > 0:
                        quality_issues.append(f"{n_neg_claim} dossier(s) avec un montant réclamé négatif")
                if 'policy_annual_premium' in df_raw.columns:
                    n_zero_prem = (df_raw['policy_annual_premium'] <= 0).sum()
                    if n_zero_prem > 0:
                        quality_issues.append(f"{n_zero_prem} dossier(s) avec une prime annuelle nulle ou négative")
                if 'incident_hour_of_the_day' in df_raw.columns:
                    n_bad_hour = ((df_raw['incident_hour_of_the_day'] < 0) | (df_raw['incident_hour_of_the_day'] > 23)).sum()
                    if n_bad_hour > 0:
                        quality_issues.append(f"{n_bad_hour} dossier(s) avec une heure d'incident invalide")

                if quality_issues:
                    with st.expander("Alertes qualité des données détectées", expanded=False):
                        for issue in quality_issues:
                            st.warning(issue)
                        st.caption("Ces dossiers sont tout de même analysés, mais leur score peut être moins fiable.")

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

                # ── Évaluation si le CSV contient les vraies étiquettes ──
                # Utile pour auditer le modèle sur un nouveau lot annoté
                # (ex. dossiers déjà traités manuellement par les enquêteurs)
                if 'fraud_reported' in df_raw.columns:
                    y_true_batch = (df_raw['fraud_reported'] == 'Y').astype(int)
                    st.divider()
                    st.markdown("#### Évaluation sur ce lot (étiquettes réelles détectées)")
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("ROC-AUC",   f"{roc_auc_score(y_true_batch, probs):.3f}")
                    e2.metric("F1-Score",  f"{f1_score(y_true_batch, preds):.3f}")
                    e3.metric("Recall",    f"{recall_score(y_true_batch, preds):.3f}")
                    e4.metric("Précision", f"{precision_score(y_true_batch, preds, zero_division=0):.3f}")

                st.divider()

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

                # ── Répartition du risque par catégorie ──────────────────
                # Aide à repérer si la fraude se concentre sur un type
                # d'incident, une sévérité ou un état particulier
                st.markdown("#### Répartition du risque par catégorie")
                cat_options = [c for c in ['incident_type', 'incident_severity',
                                           'incident_state', 'auto_make', 'collision_type']
                               if c in df_out.columns]
                if cat_options:
                    chosen_cat = st.selectbox("Regrouper par", cat_options)
                    summary = (
                        df_out.groupby(chosen_cat)
                        .agg(Dossiers=('P_fraude', 'count'),
                             Score_moyen=('P_fraude', 'mean'),
                             Fraudes=('Prédiction', lambda s: (s == 'FRAUDE').sum()))
                        .assign(Taux_fraude=lambda d: (d['Fraudes'] / d['Dossiers'] * 100).round(1))
                        .sort_values('Score_moyen', ascending=False)
                    )
                    fig_cat, ax_cat = plt.subplots(figsize=(8, max(3, 0.4*len(summary))))
                    fig_cat.patch.set_facecolor('#1a2a3a'); ax_cat.set_facecolor('#1a2a3a')
                    ax_cat.barh(summary.index.astype(str), summary['Score_moyen'],
                                color='#e67e22', edgecolor='white', alpha=0.85)
                    ax_cat.set(xlabel='Score de fraude moyen', title=f'Score moyen par {chosen_cat}')
                    ax_cat.title.set_color('white'); ax_cat.xaxis.label.set_color('white')
                    ax_cat.tick_params(colors='white')
                    ax_cat.grid(axis='x', alpha=0.2, color='white')
                    for spine in ax_cat.spines.values(): spine.set_edgecolor('#444')
                    plt.tight_layout()
                    st.pyplot(fig_cat, use_container_width=True); plt.close()
                    st.dataframe(summary, use_container_width=True)

                st.divider()

                # ── Filtrage avant affichage / export ────────────────────
                st.markdown("#### Dossiers classés par risque décroissant")
                risk_filter = st.multiselect(
                    "Filtrer par niveau de risque",
                    options=['Faible','Modéré','Élevé','Très élevé'],
                    default=['Faible','Modéré','Élevé','Très élevé']
                )
                df_filtered = df_out[df_out['Risque'].isin(risk_filter)]

                cols_show = [c for c in ['P_fraude','Prédiction','Risque','total_claim_amount',
                                         'policy_annual_premium','incident_type','incident_severity']
                             if c in df_filtered.columns]
                st.dataframe(df_filtered[cols_show].sort_values('P_fraude', ascending=False),
                             use_container_width=True, height=400)
                st.caption(f"{len(df_filtered)} dossier(s) affiché(s) sur {len(df_out)} au total.")

                # ── Export — lot complet ou lot filtré ───────────────────
                exp1, exp2 = st.columns(2)
                with exp1:
                    st.download_button("Télécharger tous les résultats (CSV)",
                                       df_out.to_csv(index=False).encode('utf-8'),
                                       "predictions_fraude_complet.csv", "text/csv")
                with exp2:
                    st.download_button("Télécharger la sélection filtrée (CSV)",
                                       df_filtered.to_csv(index=False).encode('utf-8'),
                                       "predictions_fraude_filtre.csv", "text/csv")

            except Exception as e:
                st.error(f" Erreur : {e}")
                st.exception(e)
        else:
            st.info(
                "Déposez un fichier CSV au format `insurance_claims.csv`.\n\n"
                "Le preprocessing complet (dates, imputation, ratios) est appliqué automatiquement.\n\n"
                "Astuce : si le fichier contient déjà la colonne `fraud_reported`, "
                "le modèle sera automatiquement évalué sur ce lot (ROC-AUC, F1, Recall, Précision)."
            )
 
 
if __name__ == "__main__":
    main()