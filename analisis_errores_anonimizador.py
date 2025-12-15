import pandas as pd

# Ajusta estos nombres si tu Excel se llama distinto
RESULTS_PATH = "Resultados_Anonimizador_Hibrido_v3.xlsx"

SHEET_GOLD = "Gold_Standard"
SHEET_PREDS = "Predicciones_Modelo"
SHEET_VAL_STRICT = "Validacion_Strict"
SHEET_VAL_LENIENT = "Validacion_Lenient"


def canonicalize_entity_name(name: str) -> str:
    t_raw = str(name).strip()
    t = t_raw.lower()

    if ("nombre" in t and "paciente" in t) or "nombre_sujeto_asistencia" in t:
        return "NOMBRE_PACIENTE"
    if ("nombre" in t and ("profesional" in t or "médico" in t or "medico" in t)) \
       or "nombre_personal_sanitario" in t:
        return "NOMBRE_PROFESIONAL"

    if "hospital" in t or "clínica" in t or "clinica" in t or "institución" in t or "institucion" in t:
        return "HOSPITAL"

    if "dirección" in t or "direccion" in t or "calle" in t or "territorio" in t \
       or "código postal" in t or "codigo postal" in t or "ciudad" in t or "provincia" in t:
        return "DIRECCION"

    if "teléfono" in t or "telefono" in t or "numero_telefono" in t:
        return "TELEFONO"

    if "mail" in t or "correo" in t or "email" in t:
        return "EMAIL"

    if "edad" in t:
        return "EDAD"

    if "nacimiento" in t:
        return "FECHA_NACIMIENTO"
    if "ingreso" in t or "admisión" in t or "admision" in t:
        return "FECHA_INGRESO"
    if "alta" in t:
        return "FECHA_ALTA"
    if "fecha" in t or t_raw.upper() == "FECHAS":
        return "FECHA"

    if "dni" in t or "nif" in t or "id_sujeto_asistencia" in t:
        return "DNI_NIF"
    if "historia clínica" in t or "historia clinica" in t or "nhc" in t:
        return "NHC"
    if "aseguramiento" in t or "seguro" in t:
        return "INSURANCE_ID"
    if t_raw.upper() in {"IP", "DIRECCION IP", "DIRECCIÓN IP"}:
        return "IP"

    return t_raw.upper()


def load_data():
    print(f"Cargando {RESULTS_PATH} ...")
    xl = pd.ExcelFile(RESULTS_PATH)

    gold = pd.read_excel(xl, SHEET_GOLD)
    preds = pd.read_excel(xl, SHEET_PREDS)
    val_strict = pd.read_excel(xl, SHEET_VAL_STRICT)
    val_lenient = pd.read_excel(xl, SHEET_VAL_LENIENT)

    # Canonicalizamos etiquetas para análisis
    gold["Entidad_canon"] = gold["Entidad"].map(canonicalize_entity_name)
    preds["Entidad_pred_canon"] = preds["Entidad_pred"].map(canonicalize_entity_name)

    if "Entidad_Gold_canon" not in val_strict.columns:
        val_strict["Entidad_Gold_canon"] = val_strict["Entidad_Gold"].map(canonicalize_entity_name)
    if "Entidad_Pred_canon" not in val_strict.columns:
        val_strict["Entidad_Pred_canon"] = val_strict["Entidad_Pred"].map(canonicalize_entity_name)

    if "Entidad_Gold_canon" not in val_lenient.columns:
        val_lenient["Entidad_Gold_canon"] = val_lenient["Entidad_Gold"].map(canonicalize_entity_name)
    if "Entidad_Pred_canon" not in val_lenient.columns:
        val_lenient["Entidad_Pred_canon"] = val_lenient["Entidad_Pred"].map(canonicalize_entity_name)

    return gold, preds, val_strict, val_lenient


def resumen_lenient_por_entidad(val_lenient: pd.DataFrame):
    print("\n===== RESUMEN LENIENT POR ENTIDAD CANÓNICA =====\n")

    # TP y FN se cuentan por entidad GOLD
    gold_part = val_lenient[val_lenient["Error_Type"].isin(["TP", "FN"])].copy()
    gold_part["Entidad_eval"] = gold_part["Entidad_Gold_canon"].fillna("")

    tp_gold = gold_part[gold_part["Error_Type"] == "TP"].groupby("Entidad_eval").size()
    fn_gold = gold_part[gold_part["Error_Type"] == "FN"].groupby("Entidad_eval").size()

    # FP se cuentan por entidad PRED
    pred_part = val_lenient[val_lenient["Error_Type"] == "FP"].copy()
    pred_part["Entidad_eval"] = pred_part["Entidad_Pred_canon"].fillna("")
    fp_pred = pred_part.groupby("Entidad_eval").size()

    entidades = sorted(set(tp_gold.index) | set(fn_gold.index) | set(fp_pred.index))

    rows = []
    for ent in entidades:
        TP = int(tp_gold.get(ent, 0))
        FN = int(fn_gold.get(ent, 0))
        FP = int(fp_pred.get(ent, 0))

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        rows.append(
            {
                "Entidad_canon": ent,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1": round(f1, 3),
            }
        )

    df_res = pd.DataFrame(rows).sort_values("F1", ascending=False)
    print(df_res.to_string(index=False))
    return df_res


def ejemplos_errores(val_lenient: pd.DataFrame, entidad_canon: str,
                     n_tp: int = 3, n_fp: int = 3, n_fn: int = 3):
    """
    Imprime algunos ejemplos de TP/FP/FN para una entidad concreta
    según la evaluación lenient.
    """
    print(f"\n===== EJEMPLOS PARA {entidad_canon} (LENIENT) =====")

    # True Positives
    tp = val_lenient[
        (val_lenient["Error_Type"] == "TP")
        & (val_lenient["Entidad_Gold_canon"] == entidad_canon)
    ].head(n_tp)

    print("\n--- TP (aciertos) ---")
    for _, r in tp.iterrows():
        print(f"ID {r['ID']}: GOLD='{r['Texto_Gold']}'  |  PRED='{r['Texto_Pred']}'")

    # False Negatives (faltan predicciones)
    fn = val_lenient[
        (val_lenient["Error_Type"] == "FN")
        & (val_lenient["Entidad_Gold_canon"] == entidad_canon)
    ].head(n_fn)

    print("\n--- FN (faltan) ---")
    for _, r in fn.iterrows():
        print(f"ID {r['ID']}: GOLD='{r['Texto_Gold']}'  (no se predijo nada)")

    # False Positives (sobran predicciones)
    fp = val_lenient[
        (val_lenient["Error_Type"] == "FP")
        & (val_lenient["Entidad_Pred_canon"] == entidad_canon)
    ].head(n_fp)

    print("\n--- FP (sobran) ---")
    for _, r in fp.iterrows():
        print(f"ID {r['ID']}: PRED='{r['Texto_Pred']}'  (sin GOLD correspondiente)")


if __name__ == "__main__":
    gold, preds, val_strict, val_lenient = load_data()

    # 1) Resumen global LENIENT por entidad
    df_res = resumen_lenient_por_entidad(val_lenient)

    # 2) Imprimir ejemplos para las 3 peores entidades (por F1)
    worst = df_res.sort_values("F1").head(3)["Entidad_canon"].tolist()
    for ent in worst:
        ejemplos_errores(val_lenient, ent, n_tp=2, n_fp=3, n_fn=3)
