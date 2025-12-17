import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

XLSX_PATH = os.path.join("Resultados_Sinteticas_1000.xlsx")


@st.cache_data
def load_all(path: str, file_mtime: float):
    df_anon = pd.read_excel(path, sheet_name="Textos_Anonimizados")
    df_qa = pd.read_excel(path, sheet_name="QA_Notas")
    df_miss = pd.read_excel(path, sheet_name="QA_Escapes_Detalle")
    df_ent = pd.read_excel(path, sheet_name="QA_Entidades_Fallan")
    df_metrics = pd.read_excel(path, sheet_name="Metricas")

    df_anon["ID"] = df_anon["ID"].astype(int)
    df_qa["ID"] = df_qa["ID"].astype(int)

    if not df_miss.empty:
        df_miss["ID"] = df_miss["ID"].astype(int)

    return df_anon, df_qa, df_miss, df_ent, df_metrics



def underline_misses_html(anon_text: str, misses: list[str]) -> str:
    """
    Subraya en rojo las cadenas 'misses' si aparecen en el texto anonimizado.
    Marcado simple (replace). No es perfecto con overlaps, pero funciona bien para QA visual.
    """
    # escapado mínimo para HTML
    safe = (
        anon_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    # evita duplicados y ordena por longitud (primero las largas)
    uniq = sorted({m for m in misses if isinstance(m, str) and m.strip()}, key=len, reverse=True)

    for m in uniq:
        m_safe = (
            m.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if m_safe in safe:
            safe = safe.replace(
                m_safe,
                f"<span class='miss'>{m_safe}</span>"
            )
    return safe


def pie_chart(values, labels, title):
    fig = plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title(title)
    return fig


st.set_page_config(page_title="QA Anonimización (Sintéticas)", layout="wide")
st.title("🧪 QA de anonimizador – 1000 notas sintéticas")
if st.button("🔄 Recargar (limpiar caché)"):
    st.cache_data.clear()
    st.rerun()


mtime = os.path.getmtime(XLSX_PATH) if os.path.exists(XLSX_PATH) else 0.0
df_anon, df_qa, df_miss, df_ent, df_metrics = load_all(XLSX_PATH, mtime)

# --- resumen arriba ---
c1, c2, c3 = st.columns(3)
ok = int((df_qa["Status"] == "OK").sum())
fail = int((df_qa["Status"] == "FAIL").sum())
total = len(df_qa)

with c1:
    st.metric("Total notas", total)
with c2:
    st.metric("OK (100% anon)", ok)
with c3:
    st.metric("FAIL (con escapes)", fail)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📌 Métricas", "✅ Notas OK", "❌ Notas con fallos", "📊 Gráficas"])

with tab1:
    st.subheader("Métricas (STRICT y LENIENT vs Gold_Silver_Regex)")
    st.dataframe(df_metrics, use_container_width=True)

with tab2:
    st.subheader("Notas 100% anonimizadas (OK)")
    ok_ids = df_qa[df_qa["Status"] == "OK"]["ID"].tolist()
    st.write(f"Total OK: {len(ok_ids)}")
    st.dataframe(df_qa[df_qa["Status"] == "OK"], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Notas con fallos (algo que debería anonimizarse sigue visible)")
    fail_df = df_qa[df_qa["Status"] == "FAIL"].sort_values("Misses", ascending=False)
    fail_ids = fail_df["ID"].tolist()

    if not fail_ids:
        st.success("No hay fallos 🎉")
        st.stop()

    left, right = st.columns([1, 2])

    with left:
        selected = st.selectbox("Selecciona una nota FAIL", fail_ids)
        st.dataframe(fail_df, use_container_width=True, hide_index=True)

    row = df_anon[df_anon["ID"] == selected].iloc[0]
    anon_text = str(row["Texto_anon"])
    orig_text = str(row["Texto_original"])

    misses_for_note = []
    if not df_miss.empty:
        misses_for_note = df_miss[df_miss["ID"] == selected]["Texto_que_deberia_anonimizarse"].astype(str).tolist()

    # HTML con subrayado
    styled = underline_misses_html(anon_text, misses_for_note)

    with right:
        st.markdown(
            """
            <style>
            .miss {
                text-decoration: underline;
                text-decoration-thickness: 3px;
                text-decoration-color: #d11;
                background: rgba(255, 0, 0, 0.08);
                border-radius: 4px;
                padding: 0 2px;
            }
            .box {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 12px;
                background: #fff;
                white-space: pre-wrap;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                line-height: 1.35;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("🔒 Texto anonimizado (subrayado = ESCAPE)")
        st.markdown(f"<div class='box'>{styled}</div>", unsafe_allow_html=True)

        st.subheader("📄 Texto original")
        st.text_area("Original", orig_text, height=260)

        st.subheader("📌 Escapes detectados (por regex-gold)")
        if misses_for_note:
            st.write(pd.DataFrame({"Escapes": misses_for_note}))
        else:
            st.write("No se han encontrado escapes (esto sería raro si está en FAIL).")

with tab4:
    st.subheader("Distribución OK vs FAIL")
    fig1 = pie_chart([ok, fail], ["OK", "FAIL"], "Notas OK vs FAIL")
    st.pyplot(fig1)

    st.subheader("Entidades que más fallan (solo FAIL)")
    if df_ent.empty:
        st.write("No hay fallos.")
    else:
        st.dataframe(df_ent, use_container_width=True, hide_index=True)

        # pie de top entidades (top 10)
        top = df_ent.head(10)
        fig2 = pie_chart(top["Count"].tolist(), top["Entidad"].tolist(), "Top entidades que fallan (Top 10)")
        st.pyplot(fig2)

st.caption(f"Fuente: {XLSX_PATH}")
