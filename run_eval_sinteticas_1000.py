import os
import re
import pandas as pd


from anonimizador import (
    # config / loaders
    load_guidelines,
    build_all_rules,
    load_ner_pipeline,

    # detección (modelo)
    detect_all_texts,
    preds_to_dataframe,

    # detección (regex-only) -> para gold_silver
    detect_entities_regex,

    # anonimización
    anonymize_all_texts,

    # evaluación
    evaluate_strict,
    evaluate_lenient,

    # utils
    canonicalize_entity_name,
)

# =========================
# CONFIG
# =========================
INPUT_CSV = os.path.join("notas_sinteticas_1000.csv")
OUTPUT_XLSX = os.path.join("Resultados_Sinteticas_1000.xlsx")

CSV_ID_COL = "ID"
CSV_TEXT_COL = "Texto"

# Si tu CSV tiene otras columnas, ajusta aquí:
# CSV_ID_COL = "id"
# CSV_TEXT_COL = "text"


# =========================
# GOLD SILVER (regex-only)
# =========================
def build_gold_silver_from_regex(df_textos: pd.DataFrame, rules) -> pd.DataFrame:
    """
    Genera un 'Gold_Standard' aproximado (silver) a partir SOLO de regex.
    Formato igual a tu gold: ID, Start, End, Entidad, Texto
    """
    rows = []

    for _, r in df_textos.iterrows():
        tid = int(r["ID"])
        texto = str(r["Texto"])

        preds_regex = detect_entities_regex(tid, texto, rules)
        for p in preds_regex:
            rows.append(
                {
                    "ID": tid,
                    "Start": int(p.start),
                    "End": int(p.end),
                    "Entidad": canonicalize_entity_name(p.entity),
                    "Texto": p.text,
                }
            )

    df_gold = pd.DataFrame(rows)
    if df_gold.empty:
        # por si el regex no detecta nada (raro)
        df_gold = pd.DataFrame(columns=["ID", "Start", "End", "Entidad", "Texto"])

   


    # ordena para estabilidad
    df_gold = df_gold.sort_values(["ID", "Start", "End"]).reset_index(drop=True)
    return df_gold


# =========================
# QA whitelist: constantes clínicas (NO PHI)
# =========================
CLINICAL_WHITELIST_PATTERNS = [
    r"\bTA\s*\d{2,3}\s*/\s*\d{2,3}\b",     # TA 136/62
    r"\bFC\s*\d{2,3}\b",                   # FC 107
    r"\b\d{2,3}\s*lpm\b",                  # 107 lpm
    r"\b\d{2,3}\s*mmhg\b",                 # 136 mmHg
    r"\bSatO2\s*\d{2,3}\s*%?\b",           # SatO2 98 / 98%
    r"\bgluc(?:emia)?\s*\d{2,3}\b",        # glucemia 110
    r"\b\d+\s*horas?\b",                   # 7 horas
    r"\b\d+\s*min(?:utos)?\b",             # 30 min / minutos
]

_CLINICAL_RE = re.compile("|".join(f"(?:{p})" for p in CLINICAL_WHITELIST_PATTERNS), re.IGNORECASE)

def is_clinical_constant(span_txt: str) -> bool:
    s = str(span_txt).strip()
    if not s:
        return False
    # si contiene o coincide con patrón clínico -> no lo consideramos "escape"
    return bool(_CLINICAL_RE.fullmatch(s) or _CLINICAL_RE.search(s))


# =========================
# QA: comprobar escapes
# =========================
def compute_qa_misses(df_anon: pd.DataFrame, df_gold_silver: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Para cada nota:
      - FAIL si algún Texto (span gold_silver) sigue apareciendo literal en Texto_anon
      - OK si ninguno aparece

    FIXES:
      - Ignora spans “basura” demasiado cortos (p.ej. '24', '1') que generan falsos FAIL.
      - Ignora patrones clínicos típicos (TA/FC/mmHg/lpm/hora) si llegan como span.
    """

    # ---- helpers anti falsos-positivos ----
    def should_ignore_span(span_txt: str) -> bool:
        s = (span_txt or "").strip()
        if not s:
            return True

        # 1) demasiados cortos -> casi siempre falsos FAIL (24, 1, 20, etc.)
        if len(s) <= 2 and s.isdigit():
            return True

        # 2) números muy cortos (3 dígitos) también suelen ser constantes/vitales si están aislados
        # (si quieres ser más estricto, quita esta línea)
        if len(s) == 3 and s.isdigit():
            return True

        # 3) whitelist clínica (por si algún regex te mete aquí cosas que NO son PHI)
        # ejemplos: TA 136/62, FC 107, mmHg, lpm, 09:00, 8:43, etc.
        if re.fullmatch(r"\d{1,2}:\d{2}", s):  # hora
            return True
        if re.fullmatch(r"\d{2,3}/\d{2,3}", s):  # TA 136/62
            return True
        if s.lower() in {"mmhg", "lpm"}:
            return True

        return False

    # index gold por ID
    gold_by_id = {}
    for _, g in df_gold_silver.iterrows():
        gold_by_id.setdefault(int(g["ID"]), []).append(g)

    qa_rows = []
    miss_rows = []

    for _, row in df_anon.iterrows():
        tid = int(row["ID"])
        anon = str(row["Texto_anon"])

        gold_spans = gold_by_id.get(tid, [])
        total_should = len(gold_spans)

        misses = 0

        for g in gold_spans:
            ent = str(g["Entidad"])
            span_txt = str(g["Texto"])

            # ✅ nuevo: ignora spans basura antes de marcar FAIL
            if should_ignore_span(span_txt):
                continue

            # criterio: si el texto exacto sigue en el anon -> escape
            if span_txt and (span_txt in anon):
                misses += 1
                miss_rows.append(
                    {
                        "ID": tid,
                        "Entidad": ent,
                        "Texto_que_deberia_anonimizarse": span_txt,
                    }
                )

        status = "OK" if misses == 0 else "FAIL"

        qa_rows.append(
            {
                "ID": tid,
                "Status": status,
                "Total_should_anon": total_should,
                "Misses": misses,
            }
        )

    df_qa_notas = pd.DataFrame(qa_rows).sort_values(["Status", "Misses"], ascending=[True, False])
    df_qa_misses = pd.DataFrame(miss_rows)
    if df_qa_misses.empty:
        df_qa_misses = pd.DataFrame(columns=["ID", "Entidad", "Texto_que_deberia_anonimizarse"])

    return df_qa_notas, df_qa_misses



def build_entity_fail_breakdown(df_qa_misses: pd.DataFrame) -> pd.DataFrame:
    if df_qa_misses.empty:
        return pd.DataFrame(columns=["Entidad", "Count"])

    out = (
        df_qa_misses.groupby("Entidad", as_index=False)
        .size()
        .rename(columns={"size": "Count"})
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    return out


# =========================
# MAIN
# =========================
def main():
    os.makedirs("outputs", exist_ok=True)

    # 1) Cargar notas sintéticas
    df = pd.read_csv(INPUT_CSV)
    df = df.rename(columns={CSV_ID_COL: "ID", CSV_TEXT_COL: "Texto"})
    df["ID"] = df["ID"].astype(int)
    df["Texto"] = df["Texto"].astype(str)
    df_textos = df[["ID", "Texto"]].copy()

    # 2) Guidelines + reglas
    df_guidelines = load_guidelines()
    rules = build_all_rules(df_guidelines)

    # 3) Gold_silver (regex-only)
    df_gold_silver = build_gold_silver_from_regex(df_textos, rules)

    # 4) Modelo híbrido (NER + regex)
    ner_pipe = load_ner_pipeline()
    all_preds = detect_all_texts(df_textos, rules, ner_pipe, merge_address=False)  # para evaluación
    df_preds = preds_to_dataframe(all_preds)

    # 5) Métricas strict/lenient vs gold_silver
    matching_strict, metrics_strict = evaluate_strict(df_gold_silver, df_preds)
    matching_lenient, metrics_lenient = evaluate_lenient(df_gold_silver, df_preds)

    df_metrics = pd.DataFrame(
        [
            {"Tipo": "STRICT", "Métrica": "TP", "Valor": metrics_strict["TP"]},
            {"Tipo": "STRICT", "Métrica": "FP", "Valor": metrics_strict["FP"]},
            {"Tipo": "STRICT", "Métrica": "FN", "Valor": metrics_strict["FN"]},
            {"Tipo": "STRICT", "Métrica": "Precision", "Valor": metrics_strict["Precision"]},
            {"Tipo": "STRICT", "Métrica": "Recall", "Valor": metrics_strict["Recall"]},
            {"Tipo": "STRICT", "Métrica": "F1", "Valor": metrics_strict["F1"]},
            {"Tipo": "LENIENT", "Métrica": "TP", "Valor": metrics_lenient["TP"]},
            {"Tipo": "LENIENT", "Métrica": "FP", "Valor": metrics_lenient["FP"]},
            {"Tipo": "LENIENT", "Métrica": "FN", "Valor": metrics_lenient["FN"]},
            {"Tipo": "LENIENT", "Métrica": "Precision", "Valor": metrics_lenient["Precision"]},
            {"Tipo": "LENIENT", "Métrica": "Recall", "Valor": metrics_lenient["Recall"]},
            {"Tipo": "LENIENT", "Métrica": "F1", "Valor": metrics_lenient["F1"]},
        ]
    )

    # 6) Anonimización (producción)
    all_preds_anon = detect_all_texts(df_textos, rules, ner_pipe, merge_address=True)
    df_anon = anonymize_all_texts(df_textos, all_preds_anon, rules)

    # 7) QA: notas OK vs FAIL + desglose de entidades fallidas
    df_qa_notas, df_qa_misses = compute_qa_misses(df_anon, df_gold_silver)
    df_entity_breakdown = build_entity_fail_breakdown(df_qa_misses)

    # 8) Guardar Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_textos.to_excel(writer, sheet_name="Notas_Sinteticas", index=False)
        df_guidelines.to_excel(writer, sheet_name="Guidelines", index=False)
        df_gold_silver.to_excel(writer, sheet_name="Gold_Silver_Regex", index=False)

        df_preds.to_excel(writer, sheet_name="Predicciones_Modelo", index=False)
        matching_strict.to_excel(writer, sheet_name="Validacion_Strict", index=False)
        matching_lenient.to_excel(writer, sheet_name="Validacion_Lenient", index=False)
        df_metrics.to_excel(writer, sheet_name="Metricas", index=False)

        df_anon.to_excel(writer, sheet_name="Textos_Anonimizados", index=False)
        df_qa_notas.to_excel(writer, sheet_name="QA_Notas", index=False)
        df_qa_misses.to_excel(writer, sheet_name="QA_Escapes_Detalle", index=False)
        df_entity_breakdown.to_excel(writer, sheet_name="QA_Entidades_Fallan", index=False)

    print(f"\n✔️ Listo. Excel generado en: {OUTPUT_XLSX}")
    print("Para ver el visor QA:")
    print("  streamlit run visor_qa_sinteticas.py")


if __name__ == "__main__":
    main()
