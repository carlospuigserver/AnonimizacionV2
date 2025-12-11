import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# ========= CONFIGURACIÓN GENERAL =========

EXCEL_PATH = "Gold_Standard_Anonimizacion.xlsx"

SHEET_GUIDELINES = "Guidelines"        # reglas, regex, sustituciones
SHEET_TEXTOS = "Datos_Originales"      # textos originales (ID, Texto)
SHEET_GOLD = "Gold_Standard"           # anotaciones gold (ID, Start, End, Entidad, Texto)

OUTPUT_EXCEL = "Resultados_Anonimizador_Hibrido_v2.xlsx"

MODEL_NAME = "BSC-NLP4BIA/bsc-bio-ehr-es-meddocan"


# ========= MODELOS DE DATOS =========

@dataclass
class Rule:
    entity: str        # nombre tal como viene en Guidelines
    pattern: str       # regex
    replacement: str   # texto de sustitución
    obligatorio: bool  # si la entidad es obligatoria o no (por si quieres usarlo después)


@dataclass
class Prediction:
    id_texto: int
    start: int
    end: int
    entity: str         # nombre de entidad "humano" (Guidelines-like)
    text: str
    score: float = 1.0


# ========= CARGA DE DATOS =========

def load_guidelines(path: str = EXCEL_PATH, sheet_name: str = SHEET_GUIDELINES) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)

    df["Entidad"] = df["Entidad"].astype(str).str.strip()

    if "Obligatorio" in df.columns:
        df["Obligatorio"] = (
            df["Obligatorio"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"sí": "si"})
        )
    else:
        df["Obligatorio"] = "si"

    return df


def load_textos(path: str = EXCEL_PATH, sheet_name: str = SHEET_TEXTOS) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    df["ID"] = df["ID"].astype(int)
    df["Texto"] = df["Texto"].astype(str)
    return df


def load_gold(path: str = EXCEL_PATH, sheet_name: str = SHEET_GOLD) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    df["ID"] = df["ID"].astype(int)
    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)
    df["Entidad"] = df["Entidad"].astype(str).str.strip()
    df["Texto"] = df["Texto"].astype(str)
    return df


# ========= UTILIDADES DE NLP LIGERO (NORMALIZACIÓN, CONTEXTO) =========

def canonicalize_entity_name(name: str) -> str:
    """
    Normaliza nombres de entidades (tanto del GOLD como de las predicciones)
    a una taxonomía común tipo MEDDOCAN/HIPAA.
    Esto armoniza GOLD ↔ MEDDOCAN y mejora F1.
    """
    t = name.strip().lower()

    # Nombres
    if "nombre" in t and "paciente" in t:
        return "NOMBRE_PACIENTE"
    if "nombre" in t and ("profesional" in t or "médico" in t or "medico" in t):
        return "NOMBRE_PROFESIONAL"

    # Hospital / institución
    if "hospital" in t or "clínica" in t or "clinica" in t or "institución" in t or "institucion" in t:
        return "HOSPITAL"

    # Direcciones, territorio, ciudad, provincia, código postal
    if "dirección" in t or "direccion" in t:
        return "DIRECCION"
    if "código postal" in t or "codigo postal" in t or "cp" == t:
        return "CODIGO_POSTAL"
    if "ciudad" in t:
        return "CIUDAD"
    if "provincia" in t:
        return "PROVINCIA"

    # Teléfono
    if "teléfono" in t or "telefono" in t or "tel" == t:
        return "TELEFONO"

    # Correo
    if "mail" in t or "email" in t or "correo" in t:
        return "EMAIL"

    # Edad
    if "edad" in t:
        return "EDAD"

    # Fechas específicas
    if "nacimiento" in t:
        return "FECHA_NACIMIENTO"
    if "ingreso" in t or "admisión" in t or "admision" in t:
        return "FECHA_INGRESO"
    if "alta" in t:
        return "FECHA_ALTA"

    # Fechas genéricas
    if "fecha" in t or t == "fechas":
        return "FECHA"

    # Identificadores
    if "dni" in t or "nif" in t:
        return "DNI_NIF"
    if "historia" in t or "hc" in t:
        return "NHC"
    if "aseguramiento" in t or "seguro" in t:
        return "ID_ASEGURAMIENTO"
    if "ip" in t:
        return "IP"

    # Fallback: devolver algo estable en mayúsculas
    return t.upper()


def classify_date_entity_by_context(text: str, start: int, end: int) -> str:
    """
    A partir del contexto local alrededor de una fecha, decide si es
    fecha de nacimiento / ingreso / alta / genérica.
    """
    window = 40
    left = max(0, start - window)
    right = min(len(text), end + window)
    ctx = text[left:right].lower()

    if "nacim" in ctx or "nació" in ctx or "nacio" in ctx:
        return "Fecha de nacimiento"
    if "ingres" in ctx or "admis" in ctx:
        return "Fecha de ingreso"
    if "alta" in ctx:
        return "Fecha de alta"
    # Si nada cuadra, dejamos genérica
    return "FECHAS"


SPANISH_PROVINCES = {
    "alava", "álava", "albacete", "alicante", "almeria", "almería", "asturias",
    "avila", "ávila", "badajoz", "barcelona", "burgos", "caceres", "cáceres",
    "cadiz", "cádiz", "cantabria", "castellon", "castellón", "ceuta", "ciudad real",
    "cordoba", "córdoba", "cuenca", "girona", "gerona", "granada", "guadalajara",
    "guipuzcoa", "gipuzkoa", "huelva", "huesca", "illes balears", "islas baleares",
    "jaen", "jaén", "la rioja", "las palmas", "leon", "león", "lleida", "lerida",
    "lugo", "madrid", "malaga", "málaga", "melilla", "murcia", "navarra", "ourense",
    "orense", "palencia", "pontevedra", "salamanca", "segovia", "sevilla", "soria",
    "tarragona", "santa cruz de tenerife", "teruel", "toledo", "valencia", "valladolid",
    "vizcaya", "bizkaia", "zamora", "zaragoza"
}


def refine_address_entity(text: str, start: int, end: int, base_entity: str) -> str:
    """
    Dado un span detectado como dirección (CALLE, TERRITORIO, etc.),
    intenta separar en Calle / Código Postal / Ciudad / Provincia usando heurísticas.
    """
    span = text[start:end].strip()
    span_l = span.lower()

    # Código postal español
    if re.fullmatch(r"\d{5}", span):
        return "Código Postal"

    # Provincia (si coincide exactamente con una provincia)
    if span_l in SPANISH_PROVINCES:
        return "Provincia"

    # Ciudad: heurística muy sencilla (podríamos enriquecerla con listas, embeddings, etc.)
    # Si viene después de un CP o en mayúsculas iniciales, etc.
    # Para v2, si no es CP ni Provincia, pero parece un nombre de lugar → Ciudad.
    if base_entity.lower() == "dirección" and len(span.split()) == 1 and span[0].isupper():
        return "Ciudad"

    # Calle (si contiene "c/", "calle", "av.", "avenida", "paseo"...)
    if any(pref in span_l for pref in ["c/", "calle", "av.", "avenida", "paseo", "plaza", "pza."]):
        return "Dirección"  # mantenemos "Dirección" como entidad visible

    # Fallback: dejamos la entidad de Guidelines
    return base_entity


# ========= REGLAS REGEX DESDE GUIDELINES =========

SKIP_REGEX_ENTITIES = {
    "Fecha de nacimiento",
    "Fecha de ingreso",
    "Fecha de alta",
}

def sanitize_pattern(entity: str, pattern: str) -> str:
    """
    Arregla patrones problemáticos del Excel y refina algunos casos
    para evitar FPs masivos.
    """
    pattern = str(pattern).strip()
    ent_up = entity.upper()

    # Email / correo electrónico
    if "MAIL" in ent_up or "CORREO" in ent_up:
        return r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    # Edad: número + "años" o "a."
    if entity.strip().lower() == "edad":
        return r"\b\d{1,3}\s*(?:años?|a\.?)\b"

    # Teléfono
    if "TELEFONO" in ent_up or "TELÉFONO" in ent_up:
        return r"\b(?:\+?\d{1,3}[ ]?)?(?:\d[ ]?){7,12}\b"

    # IP
    if entity.strip().lower() in {"ip", "dirección ip", "direccion ip"}:
        return r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

    # Nº Historia Clínica / NHC
    if "historia" in ent_up or "HC" == ent_up:
        # Captura patrones tipo "Nº Historia Clínica: HC-0099123" o "HC-0099123"
        return r"\b(?:N[º°]\s*)?Historia\s+Cl[ií]nica[:\s-]*[A-Z0-9\-]{4,}\b|\bHC-?[A-Z0-9]{4,}\b"

    return pattern


def build_rules_from_guidelines(df: pd.DataFrame) -> List[Rule]:
    col_entidad = "Entidad"
    col_regex = "Patrón (Regex opcional)"
    col_regla = "Regla (Sustitución)"
    col_ej_anon = "Ejemplo Anonimizado"
    col_oblig = "Obligatorio"

    rules: List[Rule] = []

    for _, row in df.iterrows():
        entity = str(row[col_entidad]).strip()

        # Saltar entidades de fecha específicas para no multiplicar detecciones
        if entity in SKIP_REGEX_ENTITIES:
            continue

        pattern = row.get(col_regex, None)
        if pd.isna(pattern) or not str(pattern).strip():
            # Sin regex → se cubrirá solo por NER
            continue

        pattern = sanitize_pattern(entity, str(pattern))

        replacement = row.get(col_regla, None)
        if pd.isna(replacement) or not str(replacement).strip():
            replacement = row.get(col_ej_anon, None)

        if pd.isna(replacement) or not str(replacement).strip():
            # Fallback HIPAA-like
            repl_canon = canonicalize_entity_name(entity)
            replacement = f"<{repl_canon}>"

        replacement = str(replacement).strip()

        obligatorio_raw = str(row.get(col_oblig, "")).strip().lower()
        obligatorio = obligatorio_raw in {"si", "sí", "yes", "y", "true", "1"}

        rules.append(
            Rule(
                entity=entity,
                pattern=pattern,
                replacement=replacement,
                obligatorio=obligatorio,
            )
        )

    return rules


# ========= NER MEDDOCAN =========

def load_ner_pipeline(model_name: str = MODEL_NAME):
    print(f"Cargando modelo NER MEDDOCAN: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    return ner


# Mapeo de labels MEDDOCAN → nombres de entidad "humanos"
NER_LABEL_MAP: Dict[str, str] = {
    "NOMBRE_SUJETO_ASISTENCIA": "Nombre Paciente",
    "OTROS_SUJETO_ASISTENCIA": "Nombre Paciente",
    "NOMBRE_PERSONAL_SANITARIO": "Nombre Profesional",
    "HOSPITAL": "Hospital",
    "INSTITUCION": "Hospital",
    "CALLE": "Dirección",
    "TERRITORIO": "Dirección",
    "NUMERO_TELEFONO": "Teléfono",
    "CORREO_ELECTRONICO": "Email",
    "EDAD_SUJETO_ASISTENCIA": "Edad",
    "ID_SUJETO_ASISTENCIA": "DNI/NIF",
    "ID_ASEGURAMIENTO": "ID Aseguramiento",
    "FECHAS": "FECHAS",
}


def map_ner_label_to_entity(label: str) -> str:
    clean = label.replace("B-", "").replace("I-", "").upper()
    return NER_LABEL_MAP.get(clean, clean)


def detect_entities_ner(text_id: int, text: str, ner_pipeline) -> List[Prediction]:
    preds: List[Prediction] = []
    outputs = ner_pipeline(text)

    for ent in outputs:
        start = int(ent["start"])
        end = int(ent["end"])
        raw_label = str(ent.get("entity_group", ent.get("entity", "")))
        span_text = text[start:end]
        score = float(ent.get("score", 1.0))

        mapped_entity = map_ner_label_to_entity(raw_label)

        # Clasificación contextual de fechas
        if mapped_entity == "FECHAS":
            mapped_entity = classify_date_entity_by_context(text, start, end)

        # Refinar direcciones (calle / ciudad / provincia / CP)
        if mapped_entity == "Dirección":
            mapped_entity = refine_address_entity(text, start, end, mapped_entity)

        preds.append(
            Prediction(
                id_texto=text_id,
                start=start,
                end=end,
                entity=mapped_entity,
                text=span_text,
                score=score,
            )
        )

    return preds


# ========= REGLAS REGEX PURAS =========

def detect_entities_regex(text_id: int, text: str, rules: List[Rule]) -> List[Prediction]:
    preds: List[Prediction] = []

    for rule in rules:
        regex = re.compile(rule.pattern, flags=re.IGNORECASE)
        for m in regex.finditer(text):
            start, end = m.start(), m.end()
            span_text = text[start:end]
            preds.append(
                Prediction(
                    id_texto=text_id,
                    start=start,
                    end=end,
                    entity=rule.entity,
                    text=span_text,
                    score=1.0,
                )
            )

    return preds


# ========= COMBINACIÓN NER + REGEX =========

def merge_predictions(preds_a: List[Prediction], preds_b: List[Prediction]) -> List[Prediction]:
    seen = set()
    merged: List[Prediction] = []

    for p in preds_a + preds_b:
        key = (p.id_texto, p.start, p.end, p.entity)
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)

    return merged


def detect_all_texts(df_textos: pd.DataFrame, rules: List[Rule], ner_pipeline) -> List[Prediction]:
    all_preds: List[Prediction] = []

    for _, row in df_textos.iterrows():
        tid = int(row["ID"])
        ttext = str(row["Texto"])

        preds_ner = detect_entities_ner(tid, ttext, ner_pipeline)
        preds_regex = detect_entities_regex(tid, ttext, rules)

        preds_combined = merge_predictions(preds_ner, preds_regex)
        all_preds.extend(preds_combined)

    return all_preds


# ========= ANONIMIZACIÓN (HIPAA-LIKE) =========

# Placeholders HIPAA-like por entidad canónica
HIPAA_PLACEHOLDERS = {
    "NOMBRE_PACIENTE": "<PATIENT_NAME>",
    "NOMBRE_PROFESIONAL": "<PROVIDER_NAME>",
    "HOSPITAL": "<HOSPITAL_NAME>",
    "DIRECCION": "<ADDRESS>",
    "CIUDAD": "<CITY>",
    "PROVINCIA": "<PROVINCE>",
    "CODIGO_POSTAL": "<ZIP_CODE>",
    "TELEFONO": "<PHONE>",
    "EMAIL": "<EMAIL>",
    "EDAD": "<AGE>",
    "FECHA_NACIMIENTO": "<DATE_BIRTH>",
    "FECHA_INGRESO": "<DATE_ADMISSION>",
    "FECHA_ALTA": "<DATE_DISCHARGE>",
    "FECHA": "<DATE>",
    "DNI_NIF": "<ID_DOC>",
    "NHC": "<MRN>",
    "ID_ASEGURAMIENTO": "<INSURANCE_ID>",
    "IP": "<IP_ADDRESS>",
}


def anonymize_text(text: str, preds: List[Prediction], rules_by_entity: Dict[str, Rule]) -> str:
    spans: List[Tuple[int, int, str]] = []

    for p in preds:
        # 1) Intentar usar Guidelines (si hay regla para esa entidad exacta)
        rule = rules_by_entity.get(p.entity)

        if rule is not None:
            replacement = rule.replacement
        else:
            # 2) Usar placeholder HIPAA-like basado en entidad canónica
            canon = canonicalize_entity_name(p.entity)
            replacement = HIPAA_PLACEHOLDERS.get(canon, f"<{canon}>")

        spans.append((p.start, p.end, replacement))

    # Sustituimos de derecha a izquierda
    spans.sort(key=lambda x: x[0], reverse=True)

    anonymized = text
    for start, end, repl in spans:
        anonymized = anonymized[:start] + repl + anonymized[end:]

    return anonymized


def anonymize_all_texts(df_textos: pd.DataFrame, all_preds: List[Prediction], rules: List[Rule]) -> pd.DataFrame:
    rules_by_entity = {r.entity: r for r in rules}

    preds_by_id: Dict[int, List[Prediction]] = {}
    for p in all_preds:
        preds_by_id.setdefault(p.id_texto, []).append(p)

    rows = []
    for _, row in df_textos.iterrows():
        tid = int(row["ID"])
        original = str(row["Texto"])
        preds = preds_by_id.get(tid, [])
        anon = anonymize_text(original, preds, rules_by_entity)
        rows.append({"ID": tid, "Texto_original": original, "Texto_anon": anon})

    return pd.DataFrame(rows)


# ========= EVALUACIÓN (ESTRICTO + LENIENT) =========

def preds_to_dataframe(preds: List[Prediction]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": p.id_texto,
                "Start_pred": p.start,
                "End_pred": p.end,
                "Entidad_pred": p.entity,
                "Entidad_pred_canon": canonicalize_entity_name(p.entity),
                "Texto_pred": p.text,
                "Score": p.score,
            }
            for p in preds
        ]
    )


def add_canonical_to_gold(df_gold: pd.DataFrame) -> pd.DataFrame:
    df = df_gold.copy()
    df["Entidad_canon"] = df["Entidad"].map(canonicalize_entity_name)
    return df


def evaluate_strict(gold: pd.DataFrame, pred: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluación estricta, pero usando entidad canónica.
    exac match en: ID, Start, End, Entidad_canon.
    """
    gold = add_canonical_to_gold(gold)
    pred = pred.copy()

    gold["key"] = (
        gold["ID"].astype(str)
        + "|"
        + gold["Start"].astype(str)
        + "|"
        + gold["End"].astype(str)
        + "|"
        + gold["Entidad_canon"].astype(str)
    )

    pred["key"] = (
        pred["ID"].astype(str)
        + "|"
        + pred["Start_pred"].astype(str)
        + "|"
        + pred["End_pred"].astype(str)
        + "|"
        + pred["Entidad_pred_canon"].astype(str)
    )

    gold_keys = set(gold["key"])
    pred_keys = set(pred["key"])

    tp_keys = gold_keys & pred_keys
    fn_keys = gold_keys - pred_keys
    fp_keys = pred_keys - gold_keys

    rows = []

    for k in tp_keys:
        g_row = gold[gold["key"] == k].iloc[0]
        p_row = pred[pred["key"] == k].iloc[0]
        rows.append(
            {
                "ID": g_row["ID"],
                "Entidad_Gold": g_row["Entidad"],
                "Entidad_Gold_canon": g_row["Entidad_canon"],
                "Texto_Gold": g_row["Texto"],
                "Entidad_Pred": p_row["Entidad_pred"],
                "Entidad_Pred_canon": p_row["Entidad_pred_canon"],
                "Texto_Pred": p_row["Texto_pred"],
                "Match": "✓",
                "Error_Type": "TP",
                "Eval": "strict",
            }
        )

    for k in fn_keys:
        g_row = gold[gold["key"] == k].iloc[0]
        rows.append(
            {
                "ID": g_row["ID"],
                "Entidad_Gold": g_row["Entidad"],
                "Entidad_Gold_canon": g_row["Entidad_canon"],
                "Texto_Gold": g_row["Texto"],
                "Entidad_Pred": "—",
                "Entidad_Pred_canon": "",
                "Texto_Pred": "",
                "Match": "✗",
                "Error_Type": "FN",
                "Eval": "strict",
            }
        )

    for k in fp_keys:
        p_row = pred[pred["key"] == k].iloc[0]
        rows.append(
            {
                "ID": p_row["ID"],
                "Entidad_Gold": "—",
                "Entidad_Gold_canon": "",
                "Texto_Gold": "",
                "Entidad_Pred": p_row["Entidad_pred"],
                "Entidad_Pred_canon": p_row["Entidad_pred_canon"],
                "Texto_Pred": p_row["Texto_pred"],
                "Match": "✗",
                "Error_Type": "FP",
                "Eval": "strict",
            }
        )

    matching_df = pd.DataFrame(rows)

    TP = len(matching_df[matching_df["Error_Type"] == "TP"])
    FN = len(matching_df[matching_df["Error_Type"] == "FN"])
    FP = len(matching_df[matching_df["Error_Type"] == "FP"])

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    return matching_df, metrics


def spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return not (e1 <= s2 or e2 <= s1)


def evaluate_lenient(gold: pd.DataFrame, pred: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluación lenient:
    - ID igual
    - Entidad canónica igual
    - Cualquier solape de span cuenta como acierto (TP).
    """
    g = add_canonical_to_gold(gold)
    p = pred.copy()

    g_rows = list(g.to_dict("records"))
    p_rows = list(p.to_dict("records"))

    used_pred = set()
    rows = []

    # Emparejar cada GOLD con el primer PRED compatible
    for gi, gr in enumerate(g_rows):
        matched = False
        for pj, pr in enumerate(p_rows):
            if pj in used_pred:
                continue
            if gr["ID"] != pr["ID"]:
                continue
            if gr["Entidad_canon"] != pr["Entidad_pred_canon"]:
                continue
            if not spans_overlap(gr["Start"], gr["End"], pr["Start_pred"], pr["End_pred"]):
                continue

            # TP
            used_pred.add(pj)
            matched = True
            rows.append(
                {
                    "ID": gr["ID"],
                    "Entidad_Gold": gr["Entidad"],
                    "Entidad_Gold_canon": gr["Entidad_canon"],
                    "Texto_Gold": gr["Texto"],
                    "Start_Gold": gr["Start"],
                    "End_Gold": gr["End"],
                    "Entidad_Pred": pr["Entidad_pred"],
                    "Entidad_Pred_canon": pr["Entidad_pred_canon"],
                    "Texto_Pred": pr["Texto_pred"],
                    "Start_Pred": pr["Start_pred"],
                    "End_Pred": pr["End_pred"],
                    "Match": "✓",
                    "Error_Type": "TP",
                    "Eval": "lenient",
                }
            )
            break

        if not matched:
            rows.append(
                {
                    "ID": gr["ID"],
                    "Entidad_Gold": gr["Entidad"],
                    "Entidad_Gold_canon": gr["Entidad_canon"],
                    "Texto_Gold": gr["Texto"],
                    "Start_Gold": gr["Start"],
                    "End_Gold": gr["End"],
                    "Entidad_Pred": "—",
                    "Entidad_Pred_canon": "",
                    "Texto_Pred": "",
                    "Start_Pred": None,
                    "End_Pred": None,
                    "Match": "✗",
                    "Error_Type": "FN",
                    "Eval": "lenient",
                }
            )

    # Cualquier pred no usado = FP
    for pj, pr in enumerate(p_rows):
        if pj in used_pred:
            continue
        rows.append(
            {
                "ID": pr["ID"],
                "Entidad_Gold": "—",
                "Entidad_Gold_canon": "",
                "Texto_Gold": "",
                "Start_Gold": None,
                "End_Gold": None,
                "Entidad_Pred": pr["Entidad_pred"],
                "Entidad_Pred_canon": pr["Entidad_pred_canon"],
                "Texto_Pred": pr["Texto_pred"],
                "Start_Pred": pr["Start_pred"],
                "End_Pred": pr["End_pred"],
                "Match": "✗",
                "Error_Type": "FP",
                "Eval": "lenient",
            }
        )

    matching_df = pd.DataFrame(rows)

    TP = len(matching_df[matching_df["Error_Type"] == "TP"])
    FN = len(matching_df[matching_df["Error_Type"] == "FN"])
    FP = len(matching_df[matching_df["Error_Type"] == "FP"])

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    return matching_df, metrics


# ========= MAIN =========

if __name__ == "__main__":
    # 1) Cargar datos
    df_guidelines = load_guidelines()
    df_textos = load_textos()
    df_gold = load_gold()

    print("Guidelines cargados:")
    print(df_guidelines.head())

    # 2) Construir reglas regex
    rules = build_rules_from_guidelines(df_guidelines)
    print(f"\nSe han construido {len(rules)} reglas regex.")

    # 3) Cargar modelo NER
    ner_pipe = load_ner_pipeline()

    # 4) Detectar entidades (NER + regex)
    print("\nLanzando detección híbrida (NER + regex)...")
    all_preds = detect_all_texts(df_textos, rules, ner_pipe)
    print(f"Total de entidades detectadas (combinadas): {len(all_preds)}")

    df_preds = preds_to_dataframe(all_preds)

    print("\nPrimeras predicciones combinadas:")
    print(df_preds.head(20))

    # 5) Anonimizar textos
    df_anon = anonymize_all_texts(df_textos, all_preds, rules)

    # 6) Evaluar (estricto + lenient)
    matching_strict, metrics_strict = evaluate_strict(df_gold, df_preds)
    matching_lenient, metrics_lenient = evaluate_lenient(df_gold, df_preds)

    print("\nMétricas STRICT:")
    for k, v in metrics_strict.items():
        print(f"{k}: {v}")

    print("\nMétricas LENIENT:")
    for k, v in metrics_lenient.items():
        print(f"{k}: {v}")

    # 7) Guardar todo a Excel
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df_guidelines.to_excel(writer, sheet_name="Guidelines", index=False)
        df_textos.to_excel(writer, sheet_name="Datos_Originales", index=False)
        df_gold.to_excel(writer, sheet_name="Gold_Standard", index=False)
        df_preds.to_excel(writer, sheet_name="Predicciones_Modelo", index=False)
        matching_strict.to_excel(writer, sheet_name="Validacion_Strict", index=False)
        matching_lenient.to_excel(writer, sheet_name="Validacion_Lenient", index=False)
        df_anon.to_excel(writer, sheet_name="Textos_Anonimizados", index=False)

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
        df_metrics.to_excel(writer, sheet_name="Metricas", index=False)

    print(f"\nResultados guardados en: {OUTPUT_EXCEL}")
