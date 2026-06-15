# ============================================================
# Proyecto Final: App Analizadora de Datasets con Streamlit
# Autor: Fernando Alberto Gaspar Callanaupa
# Año: 2026
# ============================================================

# -----------------------------
# Importando librerías necesarias
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# -----------------------------
# Configuración inicial de la App
# -----------------------------
st.set_page_config(page_title="App Analizadora de Datasets", layout="wide")
# -----------------------------
# Funciones auxiliares
# -----------------------------

@st.cache_data
def load_data(file):
    """Carga el dataset desde archivo CSV"""
    return pd.read_csv(file)

def detectar_tipos(df):
    """Detecta variables numéricas, categóricas y de fecha"""
    numericas = df.select_dtypes(include=np.number).columns.tolist()
    categoricas = df.select_dtypes(include="object").columns.tolist()
    fechas = df.select_dtypes(include="datetime").columns.tolist()
    return numericas, categoricas, fechas

def limpiar_columnas(df):
    """Estandariza nombres de columnas"""
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

def detectar_outliers(df, col):
    """Detecta outliers usando IQR"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    return outliers

# -----------------------------
# Sidebar principal
# -----------------------------
st.sidebar.title("Menú Principal")
seccion = st.sidebar.radio("Navegación", ["Home", "Carga y perfil del dataset", "Procesamiento de datos", "Análisis visual"])

# -----------------------------
# Módulo 1: Home
# -----------------------------
if seccion == "Home":
    st.title("📊 App Analizadora de Datasets con Streamlit")
    st.markdown("""
    **Objetivo del proyecto:**  
    Construir una aplicación interactiva capaz de cargar, procesar y analizar datasets de diferentes contextos.  

    **Datasets disponibles:**  
    - AI Impact on Jobs 2030  
    - Superstore  
    - Synthetic E-commerce Risk  
    - Teen Mental Health  

    **Tecnologías usadas:** Python, Pandas, Streamlit, Plotly, Matplotlib, Seaborn, GitHub.  

    ⚠️ *Nota de uso responsable:* Los resultados son exploratorios y no reemplazan validación técnica o profesional.
    """)

# -----------------------------
# Módulo 2: Carga y perfil del dataset
# -----------------------------
elif seccion == "Carga y perfil del dataset":
    st.header("📂 Carga y perfil del dataset")

    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state["df"] = df

        st.subheader("Vista previa del dataset")
        st.dataframe(df.head())

        st.write("Dimensiones:", df.shape)
        st.write("Columnas:", df.columns.tolist())
        st.write("Tipos de datos:", df.dtypes)

        numericas, categoricas, fechas = detectar_tipos(df)
        st.metric("Variables numéricas", len(numericas))
        st.metric("Variables categóricas", len(categoricas))
        st.metric("Variables de fecha", len(fechas))

        st.write("Valores nulos por columna:")
        st.bar_chart(df.isnull().sum())

    else:
        st.warning("Por favor carga un archivo CSV para continuar.")

# -----------------------------
# Módulo 3: Procesamiento de datos
# -----------------------------
elif seccion == "Procesamiento de datos":
    st.header("🛠 Procesamiento de datos")

    if "df" in st.session_state:
        df = st.session_state["df"]
        df = limpiar_columnas(df)

        st.subheader("Validación de datos")
        st.write("Duplicados:", df.duplicated().sum())
        st.write("Nulos:", df.isnull().sum().sum())

        col_num = st.selectbox("Selecciona columna numérica para detectar outliers", df.select_dtypes(include=np.number).columns)
        if col_num:
            outliers = detectar_outliers(df, col_num)
            st.write(f"Outliers detectados en {col_num}: {len(outliers)}")
            st.dataframe(outliers.head())

    else:
        st.error("No hay dataset cargado. Ve a la sección 'Carga y perfil del dataset'.")

# -----------------------------
# Módulo 4: Análisis visual
# -----------------------------
elif seccion == "Análisis visual":
    st.header("📈 Análisis visual")

    if "df" in st.session_state:
        df = st.session_state["df"]

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Resumen", "Univariado", "Bivariado", "Multivariado", "Temporal", "Insights"])

        with tab1:
            st.subheader("Resumen del dataset")
            st.write(df.describe(include="all"))

        with tab2:
            st.subheader("Análisis univariado")
            col = st.selectbox("Selecciona columna para histograma", df.columns)
            if col:
                fig = px.histogram(df, x=col)
                st.plotly_chart(fig)

        with tab3:
            st.subheader("Análisis bivariado")
            col_x = st.selectbox("Variable X", df.columns)
            col_y = st.selectbox("Variable Y", df.columns)
            if col_x and col_y:
                fig = px.scatter(df, x=col_x, y=col_y)
                st.plotly_chart(fig)

        with tab4:
            st.subheader("Análisis multivariado")
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        with tab5:
            st.subheader("Análisis temporal")
            fechas = df.select_dtypes(include="datetime").columns
            if len(fechas) > 0:
                fecha_col = st.selectbox("Selecciona columna de fecha", fechas)
                if fecha_col:
                    serie = df.groupby(fecha_col).size()
                    st.line_chart(serie)
            else:
                st.info("No hay columnas de fecha en este dataset.")

        with tab6:
            st.subheader("Insights")
            st.markdown("🔍 Aquí puedes escribir hallazgos clave y conclusiones relevantes del análisis.")
    else:
        st.error("No hay dataset cargado. Ve a la sección 'Carga y perfil del dataset'.")
