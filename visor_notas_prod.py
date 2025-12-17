import pandas as pd
import streamlit as st

DEFAULT_PATH = "Resultados_Anonimizador_Hibrido_v3.xlsx"

@st.cache_data
def load_anon(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Textos_Anonimizados")
    df["ID"] = df["ID"].astype(int)
    df["Texto_original"] = df["Texto_original"].astype(str)
    df["Texto_anon"] = df["Texto_anon"].astype(str)
    return df

@st.cache_data
def load_preds(path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name="Predicciones_Modelo")
    except Exception:
        return pd.DataFrame()
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(int)
    return df

@st.cache_data
def load_stats(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name="Stats_By_Note")
    except Exception:
        return pd.DataFrame()

st.set_page_config(page_title="Visor de Anonimización", layout="wide")
st.title("🩺 Visor de notas clínicas (Original vs Anonimizada)")

st.sidebar.header("Archivo")
path = st.sidebar.text_input("Ruta Excel resultados:", value=DEFAULT_PATH)

df = load_anon(path)
preds = load_preds(path)
stats = load_stats(path)

# ---- sidebar: filtros ----
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

# si hay stats, úsalo para ordenar/filtrar mejor
if not stats.empty and "ID" in stats.columns:
    work = work.merge(stats[["ID", "n_entidades_total", "len_original", "ratio_len_anon"]], on="ID", how="left")

sort_mode = st.sidebar.selectbox(
    "Ordenar IDs por:",
    ["ID", "Más entidades (n_entidades_total)", "Más largas (len_original)"],
    index=1 if "n_entidades_total" in work.columns else 0
)

if sort_mode == "Más entidades (n_entidades_total)" and "n_entidades_total" in work.columns:
    work = work.sort_values(["n_entidades_total", "ID"], ascending=[False, True])
elif sort_mode == "Más largas (len_original)" and "len_original" in work.columns:
    work = work.sort_values(["len_original", "ID"], ascending=[False, True])
else:
    work = work.sort_values("ID")

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
    st.text_area("Original", row["Texto_original"], height=320)

with c2:
    st.subheader("🔒 Texto anonimizado")
    st.text_area("Anonimizado", row["Texto_anon"], height=320)

# ---- Predicciones detectadas ----
st.subheader("🔎 Entidades detectadas en esta nota")
if preds.empty:
    st.info("No se encontró la hoja 'Predicciones_Modelo' en este Excel.")
else:
    p = preds[preds["ID"] == selected_id].copy()
    if p.empty:
        st.write("Sin entidades detectadas.")
    else:
        # columnas esperadas en tu output
        show_cols = [c for c in [
            "Entidad_pred_canon", "Entidad_pred", "Texto_pred", "Start_pred", "End_pred", "Score"
        ] if c in p.columns]
        if "Start_pred" in p.columns:
            p = p.sort_values(["Start_pred", "End_pred"])
        st.dataframe(p[show_cols], use_container_width=True, hide_index=True)

# ---- Flagging QA ----
st.subheader("✅ QA: marcar notas para revisión")
if "flagged_ids" not in st.session_state:
    st.session_state["flagged_ids"] = set()

colA, colB, colC = st.columns([1, 1, 2])

with colA:
    if st.button("🚩 Marcar esta nota"):
        st.session_state["flagged_ids"].add(int(selected_id))
with colB:
    if st.button("🧹 Limpiar marcadas"):
        st.session_state["flagged_ids"] = set()

with colC:
    st.write(f"Marcadas: {len(st.session_state['flagged_ids'])}")

flag_list = sorted(list(st.session_state["flagged_ids"]))
if flag_list:
    flagged_df = pd.DataFrame({"ID": flag_list})
    st.dataframe(flagged_df, use_container_width=True, hide_index=True)
    csv = flagged_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar IDs marcados (CSV)", data=csv, file_name="flagged_ids.csv", mime="text/csv")

st.divider()

# Tabla abajo para navegar rápido
st.subheader("📚 Tabla de notas (vista rápida)")
cols = ["ID", "Texto_original", "Texto_anon"]
extra = [c for c in ["n_entidades_total", "len_original", "ratio_len_anon"] if c in work.columns]
st.dataframe(
    work[cols + extra].head(200),
    use_container_width=True,
    hide_index=True
)
