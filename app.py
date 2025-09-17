
import streamlit as st, pandas as pd, joblib, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

st.set_page_config(page_title="Predicción Compra Lotes", layout="wide")

@st.cache_data
def load_data(path): return pd.read_csv(path, encoding='utf-8-sig')

@st.cache_data
def load_model(path): return joblib.load(path)

DATA_PATH = "data_sample.csv"
MODEL_PATH = "model.pkl"
df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

menu = st.sidebar.radio("Navegación", ["Vista General", "Análisis Exploratorio", "Modelo de Clasificación", "Importancia de Variables", "Predicciones"])

if menu == "Vista General":
    st.title("Vista General - Predicción Compra de Lotes")
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", df.shape[0])
    col2.metric("Lotes únicos", int(df['lote_id'].nunique()) if 'lote_id' in df.columns else 'N/A')
    col3.metric("Tasa compra (promedio)", f"{df['compra_final'].mean():.2f}" if 'compra_final' in df.columns else 'N/A')
    st.dataframe(df.head(100))

elif menu == "Análisis Exploratorio":
    st.title("Análisis Exploratorio (EDA)")
    st.write(df.describe(include='all'))
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    sel = st.multiselect("Selecciona variables numéricas para graficar", num_cols, default=num_cols[:3])
    for col in sel:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20)
        st.pyplot(fig)

elif menu == "Modelo de Clasificación":
    st.title("Modelo de Clasificación - Resultados")
    # prepare features similar to training
    X = df.drop(columns=[c for c in ['cliente_id','lote_id','compra_final'] if c in df.columns], errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    # align features with model
    if hasattr(model,'feature_names_in_'):
        for col in model.feature_names_in_:
            if col not in X.columns:
                X[col]=0
        X = X[model.feature_names_in_]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]
    if 'compra_final' in df.columns:
        st.write("AUC aprox (sobre dataset incluido):", float(roc_auc_score(df['compra_final'], probs)) if len(df['compra_final'].unique())>1 else "N/A")
        st.write("Accuracy:", float(accuracy_score(df['compra_final'], preds)))
        st.write("Precision:", float(precision_score(df['compra_final'], preds, zero_division=0)))
        st.write("Recall:", float(recall_score(df['compra_final'], preds, zero_division=0)))
    fig, ax = plt.subplots()
    ax.hist(probs, bins=20)
    st.pyplot(fig)

elif menu == "Importancia de Variables":
    st.title("Importancia de Variables")
    try:
        imp = model.feature_importances_
        feat = model.feature_names_in_
        imp_df = pd.DataFrame({'feature': feat, 'importance': imp}).sort_values('importance', ascending=False)
        st.table(imp_df.head(20))
        fig, ax = plt.subplots(figsize=(6,6))
        ax.barh(imp_df['feature'].head(15)[::-1], imp_df['importance'].head(15)[::-1])
        st.pyplot(fig)
    except Exception as e:
        st.write("No se pudo mostrar importancias:", e)

elif menu == "Predicciones":
    st.title("Predicciones - Puntaje de compra por cliente")
    uploaded = st.file_uploader("Sube un CSV con registros a predecir (misma estructura)", type=["csv"])
    sample_select = st.selectbox("O seleccionar un cliente del dataset de ejemplo", df['cliente_id'].tolist() if 'cliente_id' in df.columns else [])
    if uploaded is not None:
        new_df = pd.read_csv(uploaded, encoding='utf-8-sig')
    else:
        new_df = df[df['cliente_id']==sample_select] if 'cliente_id' in df.columns else df.head(1)
    st.dataframe(new_df)
    X_new = new_df.drop(columns=[c for c in ['cliente_id','lote_id','compra_final'] if c in new_df.columns], errors='ignore')
    X_new = pd.get_dummies(X_new, drop_first=True)
    if hasattr(model,'feature_names_in_'):
        for col in model.feature_names_in_:
            if col not in X_new.columns:
                X_new[col]=0
        X_new = X_new[model.feature_names_in_]
    probs = model.predict_proba(X_new)[:,1]
    new_df['prob_compra'] = probs
    st.dataframe(new_df[['cliente_id','prob_compra']] if 'cliente_id' in new_df.columns else new_df.head())
