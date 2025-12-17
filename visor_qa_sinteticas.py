import os
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

XLSX_PATH = os.path.join("Resultados_Sinteticas_1000.xlsx")

# -------------------------
# Helpers: subrayado HTML
# -------------------------
def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def underline_spans_html(text: str, spans: list[tuple[int, int, str]]) -> str:
    """
    Subraya spans (start,end) en un texto. label se usa como tooltip.
    spans: [(start, end, label), ...]
    """
    safe = _escape_html(text)

    # Para poder subrayar por offsets, necesitamos trabajar sobre el texto original,
    # pero insertando tags de derecha a izquierda para no romper offsets.
    # OJO: como escapamos HTML, offsets cambiarían si escapamos antes.
    # Solución: NO escapamos primero. Insertamos tags en el texto original y luego escapamos “trozos”.
    # Implementación: construimos output por segmentos.
    if not spans:
        return f"<div class='box'>{_escape_html(text)}</div>"

    spans_sorted = sorted(spans, key=lambda x: x[0])
    out_parts = []
    cursor = 0
    for s, e, label in spans_sorted:
        s = max(0, min(len(text), int(s)))
        e = max(0, min(len(text), int(e)))
        if e <= s:
            continue
        if s < cursor:
            # overlap; lo saltamos para no liarla visualmente
            continue

        # texto normal
        out_parts.append(_escape_html(text[cursor:s]))

        # texto marcado
        seg = _escape_html(text[s:e])
        lab = _escape_html(str(label))
        out_parts.append(f"<span class='mark' title='{lab}'>{seg}</span>")

        cursor = e

    # resto
    out_parts.append(_escape_html(text[cursor:]))
    return "<div class='box'>" + "".join(out_parts) + "</div>"

def underline_misses_html(text: str, misses: list[str], css_class: str) -> str:
    """
    Subraya por reemplazo literal (sin offsets) una lista de strings.
    Útil para marcar escapes en el anonimizado.
    """
    safe = _escape_html(text)
    uniq = sorted({m for m in misses if isinstance(m, str) and m.strip()}, key=len, reverse=True)

    for m in uniq:
        m_safe = _escape_html(m)
        if m_safe in safe:
            safe = safe.replace(m_safe, f"<span class='{css_class}'>{m_safe}</span>")
    return f"<div class='box'>{safe}</div>"

def underline_placeholders_html(anon_text: str) -> str:
    """
    Subraya placeholders tipo <PATIENT_NAME>, <DATE>, etc. en verde.
    """
    safe = _escape_html(anon_text)

    # placeholders estándar: <...>
    # (evita comerse HTML porque ya está escapado)
    pat = re.compile(r"&lt;[A-Z0-9_]+&gt;")

    def repl(m):
        return f"<span class='okanon'>{m.group(0)}</span>"

    safe2 = pat.sub(repl, safe)
    return f"<div class='box'>{safe2}</div>"

# -------------------------
# Charts
# -------------------------
def pie_chart(values, labels, title):
    fig = plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title(title)
    return fig

# -------------------------
# Data load (cache with mtime)
# -------------------------
@st.cache_data
def load_all(path: str, file_mtime: float):
    df_anon = pd.read_excel(path, sheet_name="Textos_Anonimizados")
    df_qa = pd.read_excel(path, sheet_name="QA_Notas")
    df_miss = pd.read_excel(path, sheet_name="QA_Escapes_Detalle")
    df_ent = pd.read_excel(path, sheet_name="QA_Entidades_Fallan")

    df_anon["ID"] = df_anon["ID"].astype(int)
    df_qa["ID"] = df_qa["ID"].astype(int)

    if not df_miss.empty:
        df_miss["ID"] = df_miss["ID"].astype(int)

    return df_anon, df_qa, df_miss, df_ent

# -------------------------
# Build offsets for ORIGINAL highlighting
# -------------------------
def build_miss_spans_for_original(orig_text: str, misses: list[str]) -> list[tuple[int, int, str]]:
    """
    Intenta localizar cada 'miss' dentro del original para subrayar por offset.
    Si hay varias ocurrencias, marca todas.
    """
    spans = []
    if not orig_text or not misses:
        return spans

    # orden por longitud para encontrar primero lo largo
    uniq = sorted({m for m in misses if isinstance(m, str) and m.strip()}, key=len, reverse=True)

    for m in uniq:
        m = m.strip()
        if not m:
            continue
        start = 0
        while True:
            idx = orig_text.find(m, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(m), f"DEBERÍA ANON: {m}"))
            start = idx + 1

    # ordenar y quitar overlaps básicos
    spans.sort(key=lambda x: x[0])
    cleaned = []
    last_end = -1
    for s, e, lab in spans:
        if s >= last_end:
            cleaned.append((s, e, lab))
            last_end = e
    return cleaned

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="QA Anonimización (Sintéticas)", layout="wide")
st.title("🧪 QA de anonimizador – 1000 notas sintéticas")

if st.button("🔄 Recargar (limpiar caché)"):
    st.cache_data.clear()
    st.rerun()

mtime = os.path.getmtime(XLSX_PATH) if os.path.exists(XLSX_PATH) else 0.0
df_anon, df_qa, df_miss, df_ent = load_all(XLSX_PATH, mtime)

ok = int((df_qa["Status"] == "OK").sum())
fail = int((df_qa["Status"] == "FAIL").sum())
total = len(df_qa)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total notas", total)
with c2:
    st.metric("OK (100% anon)", ok)
with c3:
    st.metric("FAIL (con escapes)", fail)

st.divider()

# Tabs (sin métricas)
tab_ok, tab_fail, tab_graph = st.tabs(["✅ Notas OK", "❌ Notas con fallos", "📊 Gráficas"])

# CSS global
st.markdown(
    """
    <style>
    .box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 12px;
        background: #fff;
        white-space: pre-wrap;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        line-height: 1.35;
    }
    .miss {
        text-decoration: underline;
        text-decoration-thickness: 3px;
        text-decoration-color: #d11;
        background: rgba(255, 0, 0, 0.10);
        border-radius: 4px;
        padding: 0 2px;
    }
    .okanon {
        text-decoration: underline;
        text-decoration-thickness: 3px;
        text-decoration-color: #0a7;
        background: rgba(0, 180, 120, 0.10);
        border-radius: 4px;
        padding: 0 2px;
    }
    .mark {
        text-decoration: underline;
        text-decoration-thickness: 3px;
        text-decoration-color: #d11;
        background: rgba(255, 0, 0, 0.10);
        border-radius: 4px;
        padding: 0 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------
# TAB OK
# ---------------
with tab_ok:
    st.subheader("Notas 100% anonimizadas (OK)")
    ok_df = df_qa[df_qa["Status"] == "OK"].copy()
    ok_ids = ok_df["ID"].tolist()
    st.write(f"Total OK: {len(ok_ids)}")

    if not ok_ids:
        st.info("No hay notas OK.")
        st.stop()

    left, right = st.columns([1, 2])

    with left:
        selected_ok = st.selectbox("Selecciona una nota OK", ok_ids, key="sel_ok")
        st.dataframe(ok_df, use_container_width=True, hide_index=True)

    row = df_anon[df_anon["ID"] == selected_ok].iloc[0]
    anon_text = str(row["Texto_anon"])
    orig_text = str(row["Texto_original"])

    with right:
        st.subheader("🔒 Texto anonimizado (placeholders en verde)")
        st.markdown(underline_placeholders_html(anon_text), unsafe_allow_html=True)

        st.subheader("📄 Texto original")
        st.text_area("Original", orig_text, height=260, key=f"orig_ok_{selected_ok}")

# ---------------
# TAB FAIL
# ---------------
with tab_fail:
    st.subheader("Notas con fallos (algo que debería anonimizarse sigue visible)")
    fail_df = df_qa[df_qa["Status"] == "FAIL"].sort_values("Misses", ascending=False)
    fail_ids = fail_df["ID"].tolist()

    if not fail_ids:
        st.success("No hay fallos 🎉")
        st.stop()

    left, right = st.columns([1, 2])

    with left:
        selected_fail = st.selectbox("Selecciona una nota FAIL", fail_ids, key="sel_fail")
        st.dataframe(fail_df, use_container_width=True, hide_index=True)

    row = df_anon[df_anon["ID"] == selected_fail].iloc[0]
    anon_text = str(row["Texto_anon"])
    orig_text = str(row["Texto_original"])

    # misses del excel
    misses_for_note = []
    if not df_miss.empty:
        misses_for_note = (
            df_miss[df_miss["ID"] == selected_fail]["Texto_que_deberia_anonimizarse"]
            .astype(str)
            .tolist()
        )

    # ORIGINAL: subrayar lo que debería anonimizarse (por offsets)
    miss_spans_original = build_miss_spans_for_original(orig_text, misses_for_note)
    orig_styled = underline_spans_html(orig_text, miss_spans_original)

    # ANON: subrayar placeholders (verde) + escapes literales (rojo)
    # (primero ponemos placeholders, luego marcamos misses)
    anon_with_placeholders = _escape_html(anon_text)
    anon_with_placeholders = re.sub(
        r"&lt;[A-Z0-9_]+&gt;",
        lambda m: f"<span class='okanon'>{m.group(0)}</span>",
        anon_with_placeholders
    )
    anon_with_both = anon_with_placeholders
    uniq_miss = sorted({m for m in misses_for_note if isinstance(m, str) and m.strip()}, key=len, reverse=True)
    for m in uniq_miss:
        m_safe = _escape_html(m)
        if m_safe in anon_with_both:
            anon_with_both = anon_with_both.replace(m_safe, f"<span class='miss'>{m_safe}</span>")
    anon_styled = f"<div class='box'>{anon_with_both}</div>"

    with right:
        st.subheader("🔒 Texto anonimizado (verde = anon, rojo = ESCAPE)")
        st.markdown(anon_styled, unsafe_allow_html=True)

        st.subheader("📄 Texto original (rojo = debería anon)")
        st.markdown(orig_styled, unsafe_allow_html=True)

        st.subheader("📌 Escapes detectados (por regex-gold)")
        if uniq_miss:
            st.write(pd.DataFrame({"Escapes": uniq_miss}))
        else:
            st.write("No se han encontrado escapes (raro si está en FAIL).")

# ---------------
# TAB GRÁFICAS
# ---------------
with tab_graph:
    st.subheader("Distribución OK vs FAIL")
    fig1 = pie_chart([ok, fail], ["OK", "FAIL"], "Notas OK vs FAIL")
    st.pyplot(fig1)

    st.subheader("Entidades que más fallan (solo FAIL)")
    if df_ent.empty:
        st.write("No hay fallos.")
    else:
        st.dataframe(df_ent, use_container_width=True, hide_index=True)

        top = df_ent.head(10)
        fig2 = pie_chart(top["Count"].tolist(), top["Entidad"].tolist(), "Top entidades que fallan (Top 10)")
        st.pyplot(fig2)

st.caption(f"Fuente: {XLSX_PATH}")
