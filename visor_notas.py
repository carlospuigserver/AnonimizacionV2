import pandas as pd
import streamlit as st

PATH = "Resultados_Anonimizador_Hibrido_v3.xlsx"

@st.cache_data
def load_data(path: str):
    # Hoja que tú generas en el anonimizador
    df = pd.read_excel(path, sheet_name="Textos_Anonimizados")
    # Espera columnas: ID, Texto_original, Texto_anon
    df["ID"] = df["ID"].astype(int)
    df["Texto_original"] = df["Texto_original"].astype(str)
    df["Texto_anon"] = df["Texto_anon"].astype(str)
    return df

st.set_page_config(page_title="Visor de Anonimización", layout="wide")
st.title("🩺 Visor de notas clínicas (Original vs Anonimizada)")

df = load_data(PATH)

# ---- sidebar: búsqueda / selección ----
st.sidebar.header("Filtrar")
query = st.sidebar.text_input("Buscar (en original o anon):", "")
only_diff = st.sidebar.checkbox("Solo notas con cambios (original != anon)", value=True)

work = df.copy()
if only_diff:
    work = work[work["Texto_original"] != work["Texto_anon"]]

if query.strip():
    q = query.strip().lower()
    work = work[
        work["Texto_original"].str.lower().str.contains(q, na=False)
        | work["Texto_anon"].str.lower().str.contains(q, na=False)
    ]

ids = work["ID"].tolist()
if not ids:
    st.warning("No hay resultados con ese filtro.")
    st.stop()

selected_id = st.sidebar.selectbox("Selecciona ID de nota", ids)

row = df[df["ID"] == selected_id].iloc[0]

# ---- vista principal ----
c1, c2 = st.columns(2)

with c1:
    st.subheader("📄 Texto original")
    st.text_area("Original", row["Texto_original"], height=350)

with c2:
    st.subheader("🔒 Texto anonimizado")
    st.text_area("Anonimizado", row["Texto_anon"], height=350)

st.divider()

# Tabla “clickable” abajo (para navegar rápido)
st.subheader("📚 Tabla de notas (click en una fila para ver arriba)")
st.dataframe(
    work[["ID", "Texto_original", "Texto_anon"]].head(200),
    use_container_width=True,
    hide_index=True
)
