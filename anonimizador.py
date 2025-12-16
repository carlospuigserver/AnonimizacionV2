import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Configuración General

EXCEL_PATH = "Gold_Standard_Anonimizacion.xlsx"

SHEET_GUIDELINES = "Guidelines"       
SHEET_TEXTOS = "Datos_Originales"      
SHEET_GOLD = "Gold_Standard"           

OUTPUT_EXCEL = "Resultados_Anonimizador_Hibrido_v3.xlsx"

MODEL_NAME = "BSC-NLP4BIA/bsc-bio-ehr-es-meddocan"

# Umbral mínimo de confianza para aceptar entidades del NER
NER_SCORE_THRESHOLD = 0.80
NER_SCORE_BY_ENTITY = {
    
    "NOMBRE_PACIENTE": 0.50,
    "NOMBRE_PROFESIONAL": 0.70,
    "HOSPITAL": 0.60,
    "DIRECCION": 0.60,
    "DNI_NIF": 0.75,
    "NHC": 0.75,
    "TELEFONO": 0.70,
    "EMAIL": 0.70,
    "IP": 0.80,
    "FECHA": 0.60,
    "FECHA_NACIMIENTO": 0.60,
    "FECHA_INGRESO": 0.60,
    "FECHA_ALTA": 0.60,
    "EDAD": 0.60,
    "INSURANCE_ID": 0.75,
}


# Modelos de datos

@dataclass
class Rule:
    entity: str        # nombre tal como viene en Guidelines
    pattern: str       # regex
    replacement: str   # texto de sustitución
    obligatorio: bool  # si la entidad es obligatoria o no


@dataclass
class Prediction:
    id_texto: int
    start: int
    end: int
    entity: str         
    text: str
    score: float = 1.0


# Carga de datos

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


def sanity_check_gold_offsets(df_textos: pd.DataFrame, df_gold: pd.DataFrame, max_print: int = 20):
    textos = {int(r["ID"]): str(r["Texto"]) for _, r in df_textos.iterrows()}
    bad = []
    for _, g in df_gold.iterrows():
        tid = int(g["ID"])
        s = int(g["Start"])
        e = int(g["End"])
        gold_txt = str(g["Texto"])
        full = textos.get(tid, "")
        if not (0 <= s <= e <= len(full)):
            bad.append((tid, s, e, "OUT_OF_RANGE", gold_txt, len(full)))
            continue
        slice_txt = full[s:e]
        if slice_txt != gold_txt:
            bad.append((tid, s, e, "MISMATCH", gold_txt, slice_txt))
    print(f"\n[GOLD OFFSET CHECK] total={len(df_gold)} bad={len(bad)}")
    for row in bad[:max_print]:
        print(row)
    return bad



def _find_all_occurrences(text: str, sub: str) -> List[Tuple[int, int]]:
    """Devuelve todas las ocurrencias exactas de sub en text como (start,end)."""
    out = []
    if not sub:
        return out
    i = 0
    while True:
        j = text.find(sub, i)
        if j == -1:
            break
        out.append((j, j + len(sub)))
        i = j + 1
    return out


def _flexible_whitespace_pattern(s: str) -> str:
    """
    Convierte un texto literal en patrón tolerante a espacios/puntuación frecuente.
    """
    esc = re.escape(s.strip())
    esc = re.sub(r"\\\s+", r"\\s+", esc)  
    esc = esc.replace(r"\,", r"\s*,\s*")
    esc = esc.replace(r"\.", r"\s*\.\s*")
    esc = esc.replace(r"\-", r"\s*-\s*")
    esc = esc.replace(r"\/", r"\s*/\s*")
    return esc


def _date_candidates_from_excel_datetime(s: str) -> List[str]:
    """
    Si viene 'YYYY-MM-DD 00:00:00', genera candidatos típicos en ambos órdenes:
    - YYYY-MM-DD
    - DD/MM/YYYY y variantes
    - MM/DD/YYYY y variantes
    - con año 2 dígitos (YY)
    """
    s = str(s).strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})\s+\d{2}:\d{2}:\d{2}", s)
    if not m:
        return []
    y, mo, d = m.group(1), m.group(2), m.group(3)

    d_i = int(d)
    mo_i = int(mo)
    y2 = y[-2:]

    # DD/MM/YYYY
    ddmmYYYY = [
        f"{d}/{mo}/{y}",              
        f"{d_i}/{mo_i}/{y}",          
        f"{d_i}/{mo}/{y}",            
        f"{d}/{mo_i}/{y}",            
        f"{d}/{mo}/{y2}",             
        f"{d_i}/{mo_i}/{y2}",        
        f"{d_i}/{mo}/{y2}",           
        f"{d}/{mo_i}/{y2}",           
    ]

    # MM/DD/YYYY (día/mes invertidos)
    mmddYYYY = [
        f"{mo}/{d}/{y}",              
        f"{mo_i}/{d_i}/{y}",          
        f"{mo_i}/{d}/{y}",           
        f"{mo}/{d_i}/{y}",            
        f"{mo}/{d}/{y2}",             
        f"{mo_i}/{d_i}/{y2}",         
        f"{mo_i}/{d}/{y2}",           
        f"{mo}/{d_i}/{y2}",       
    ]

    cands = [f"{y}-{mo}-{d}"] + ddmmYYYY + mmddYYYY

    # Quita duplicados
    out = []
    seen = set()
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


import unicodedata

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def _find_accent_insensitive(full: str, needle: str) -> Tuple[int, int] | None:
    """
    Busca needle en full ignorando acento, devuelve (start,end) en el texto ORIGINAL si encuentra y 
    se implementa comparando sobre versiones sin acentos (misma longitud por char).
    """
    full_norm = _strip_accents(full)
    needle_norm = _strip_accents(needle)

    idx = full_norm.lower().find(needle_norm.lower())
    if idx == -1:
        return None
    return (idx, idx + len(needle))


def repair_gold_offsets(df_textos: pd.DataFrame, df_gold: pd.DataFrame, max_print: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    "Repara Start/End del gold contra el texto real actual (por ID): "
    textos = {int(r["ID"]): str(r["Texto"]) for _, r in df_textos.iterrows()}

    repaired_rows = []
    report = []

    for idx, g in df_gold.iterrows():
        tid = int(g["ID"])
        s0 = int(g["Start"])
        e0 = int(g["End"])
        gold_txt = str(g["Texto"])
        ent = str(g["Entidad"])

        full = textos.get(tid, "")
        status = "OK"
        s_new, e_new = s0, e0

        # Si slice exacto coincide, no tocamos
        if (0 <= s0 <= e0 <= len(full)) and full[s0:e0] == gold_txt:
            status = "OK"
        else:
            # Intento 1: búsqueda exacta del texto gold
            occ = _find_all_occurrences(full, gold_txt)
            if occ:
                if 0 <= s0 <= len(full):
                    occ.sort(key=lambda se: abs(se[0] - s0))
                s_new, e_new = occ[0]
                status = "REPAIRED_EXACT"
            else:
                # Intento 2: tolerante a espacios/puntuación
                pat = _flexible_whitespace_pattern(gold_txt)
                m = re.search(pat, full, flags=re.IGNORECASE)
                if m:
                    s_new, e_new = m.start(), m.end()
                    status = "REPAIRED_FLEX"
                else:
                    # --- Heurística 1: Fechas tipo 'YYYY-MM-DD 00:00:00' (PRIMERO, para que no lo “pise” PHONE) ---
                    cands = _date_candidates_from_excel_datetime(gold_txt)
                    found = False
                    for c in cands:
                        occ2 = _find_all_occurrences(full, c)
                        if occ2:
                            if 0 <= s0 <= len(full):
                                occ2.sort(key=lambda se: abs(se[0] - s0))
                            s_new, e_new = occ2[0]
                            status = "REPAIRED_DATE"
                            found = True
                            break

                    if not found:
                        # --- Heurística 2: Teléfono solo dígitos (solo si la ENTIDAD sugiere teléfono) ---
                        digits = re.sub(r"\D", "", gold_txt)
                        is_phone_entity = ("tel" in ent.lower()) or ("telefono" in ent.lower()) or ("teléfono" in ent.lower())

                        if is_phone_entity and len(digits) >= 9:
                            phone_pat = r"\b" + r"[\s\-\.]*".join(list(digits)) + r"\b"
                            m2 = re.search(phone_pat, full)
                            if m2:
                                s_new, e_new = m2.start(), m2.end()
                                status = "REPAIRED_PHONE"
                            else:
                                status = "UNRESOLVED"
                                s_new, e_new = s0, e0
                        else:
                            # --- Heurística 3: Email (buscar un email cerca del start original) ---
                            if "mail" in ent.lower() or "correo" in ent.lower() or "email" in ent.lower():
                                left = max(0, s0 - 80)
                                right = min(len(full), s0 + 120)
                                window = full[left:right]
                                m3 = re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", window)
                                if m3:
                                    s_new, e_new = left + m3.start(), left + m3.end()
                                    status = "REPAIRED_EMAIL_NEAR"
                                else:
                                    status = "UNRESOLVED"
                                    s_new, e_new = s0, e0
                            else:
                                # Heurística extra: Dirección sin acentos (Alcalá vs Alcala) + espacios
                                if "direc" in ent.lower() or "calle" in ent.lower() or "territorio" in ent.lower():
                                    # 1) intenta match sin acentos tal cual
                                    hit = _find_accent_insensitive(full, gold_txt)
                                    if hit:
                                        s_new, e_new = hit
                                        status = "REPAIRED_ADDR_NOACC"
                                    else:
                                        # 2) intenta con espacio opcional tras '/'
                                        gold_txt2 = gold_txt.replace("C/", "C/ ").replace("c/", "c/ ")
                                        hit2 = _find_accent_insensitive(full, gold_txt2)
                                        if hit2:
                                            s_new, e_new = hit2
                                            status = "REPAIRED_ADDR_NOACC"
                                        else:
                                            status = "UNRESOLVED"
                                            s_new, e_new = s0, e0
                                else:
                                    status = "UNRESOLVED"
                                    s_new, e_new = s0, e0



        slice_now = full[s_new:e_new] if (0 <= s_new <= e_new <= len(full)) else ""

        new_row = {**g.to_dict(), "Start": int(s_new), "End": int(e_new)}

        # si hemos conseguido ubicar el span, hacemos que Texto sea el slice real
        if status in {"REPAIRED_EXACT", "REPAIRED_FLEX", "REPAIRED_PHONE", "REPAIRED_DATE", "REPAIRED_EMAIL_NEAR", "REPAIRED_ADDR_NOACC"}:

            new_row["Texto"] = slice_now


        repaired_rows.append(new_row)


        slice_now = full[s_new:e_new] if (0 <= s_new <= e_new <= len(full)) else ""
        report.append(
            {
                "Row": idx,
                "ID": tid,
                "Entidad": ent,
                "Status": status,
                "Start_old": s0,
                "End_old": e0,
                "Start_new": s_new,
                "End_new": e_new,
                "Gold_Texto": gold_txt,
                "Slice_new": slice_now,
            }
        )

    df_gold_fixed = pd.DataFrame(repaired_rows)
    df_report = pd.DataFrame(report)

    repaired = df_report[df_report["Status"].str.startswith("REPAIRED")]
    unresolved = df_report[df_report["Status"] == "UNRESOLVED"]

    print(f"\n[GOLD REPAIR] total={len(df_report)} repaired={len(repaired)} unresolved={len(unresolved)}")
    if len(unresolved) > 0:
        print("\n[GOLD REPAIR] Ejemplos UNRESOLVED:")
        for _, r in unresolved.head(max_print).iterrows():
            print((r["ID"], r["Entidad"], r["Gold_Texto"]))

    return df_gold_fixed, df_report




# NLP ligero y taxonomía

def canonicalize_entity_name(name: str) -> str:
    """
    Normaliza nombres de entidades GOLD/MEDDOCAN/guidelines a una
    taxonomía estable.
    """
    t_raw = str(name).strip()
    t = t_raw.lower()

    # Nombres
    if ("nombre" in t and "paciente" in t) or "nombre_sujeto_asistencia" in t:
        return "NOMBRE_PACIENTE"

    if ("nombre" in t and ("profesional" in t or "médico" in t or "medico" in t)) \
       or "nombre_personal_sanitario" in t:
        return "NOMBRE_PROFESIONAL"
    
    if "historia clínica" in t or "historia clinica" in t or "nhc" in t or "mrn" in t:
        return "NHC"
    
    if t_raw.upper() in {"HC"}:
        return "NHC"

    

    # Hospital
    if t_raw.upper() in {"HOSPITL"}:
        return "HOSPITAL"


    if "hospital" in t or "clínica" in t or "clinica" in t \
       or "institución" in t or "institucion" in t:
        return "HOSPITAL"

    # Dirección
    if "dirección" in t or "direccion" in t or "calle" in t or "territorio" in t \
       or "código postal" in t or "codigo postal" in t or "ciudad" in t \
       or "provincia" in t:
        return "DIRECCION"

    # Teléfono
    if "teléfono" in t or "telefono" in t or "numero_telefono" in t:
        return "TELEFONO"

    # Email
    if "correo" in t or "email" in t or "mail" in t or t_raw.upper() == "EMALI":
        return "EMAIL"

    # Edad
    if "edad" in t:
        return "EDAD"

    # Fechas específicas
    if "nacimiento" in t:
        return "FECHA_NACIMIENTO"
    if "ingreso" in t or "admisión" in t or "admision" in t:
        return "FECHA_INGRESO"
    if "alta" in t and "fecha" in t:
        return "FECHA_ALTA"

    # Fechas genéricas (etiqueta MEDDOCAN)
    if "fecha" in t or t_raw.upper() == "FECHAS":
        return "FECHA"

    # Identificadores personales
    if "dni" in t or "nif" in t or "id_sujeto_asistencia" in t:
        return "DNI_NIF"

   


    if "aseguramiento" in t or "seguro" in t or t_raw.upper() == "ID_ASEGURAMIENTO":
        return "INSURANCE_ID"

    # IP explícita
    if t_raw.upper() in {"IP", "DIRECCION IP", "DIRECCIÓN IP"}:
        return "IP"
    # IDs técnicos
    if "device" in t or "dispositivo" in t or t_raw.upper() in {"DEVICE_ID"}:
        return "DEVICE_ID"
    
    if "pasaporte" in t or t_raw.upper() == "PASAPORTE":
        return "PASAPORTE"




    # El resto se devuelve en mayúsculas tal cual
    return t_raw.upper()


IP_REGEX = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")


def looks_like_ip(text: str) -> bool:
    return IP_REGEX.search(text) is not None


def classify_date_entity(full_text: str, start: int, end: int) -> str:
    window_left = full_text[max(0, start - 80):start].lower()
    window_right = full_text[end:end + 80].lower()
    ctx = window_left + " " + window_right

    # Nacimiento
    if any(k in ctx for k in [
        "nacim", "nació", "nacio", "fnac", "f. nac", "f nac",
        "fecha de nacimiento", "nacido", "nacida"
    ]):
        return "FECHA_NACIMIENTO"

    # Ingreso 
    if any(k in ctx for k in [
        "ingres", "ingreso", "ingresó", "ingreso el", "fecha de ingreso",
        "admis", "admisión", "admision", "admitido", "admitida", "admis.",
        "entrada", "hospitalización", "hospitalizacion"
    ]):
        return "FECHA_INGRESO"

    # Alta
    if any(k in ctx for k in [
        "alta", "alta el", "fecha de alta", "alta hospitalaria",
        "alta médica", "alta medica", "discharge"
    ]):
        return "FECHA_ALTA"

    return "FECHA"


def classify_person_role_by_context(full_text: str, start: int, end: int, default: str) -> str:
   
    left = full_text[max(0, start - 40):start].lower()
    right = full_text[end:end + 60].lower()
    ctx = left + " " + right

    # 1) Señales de profesional
    if any(k in ctx for k in ["dr.", "dra.", "doctor", "doctora", "médico", "medico"]):
        return "NOMBRE_PROFESIONAL"

    # 2) Señales de paciente 
    patient_markers = [
        " ingresó", " ingreso", " ingresa", " ingresaba",
        " admitido", " admitida", " admisión", " admision",
        " paciente", " se presenta", " acude", " acudió", " acudio"
    ]
    if any(k in ctx for k in patient_markers):
        return "NOMBRE_PACIENTE"

    return default



def classify_date_entity_by_context(text: str, start: int, end: int) -> str:
    
    window = 40
    left = max(0, start - window)
    right = min(len(text), end + window)
    ctx = text[left:right].lower()

    if "nacim" in ctx or "nació" in ctx or "nacio" in ctx or "fecha de nacimiento" in ctx:
        return "Fecha de nacimiento"
    if "ingres" in ctx or "admis" in ctx or "fecha de ingreso" in ctx:
        return "Fecha de ingreso"
    if "alta" in ctx and "fecha" in ctx:
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
    
    span = text[start:end].strip()
    span_l = span.lower()

    # Código postal español
    if re.fullmatch(r"\d{5}", span):
        return "Código Postal"

    # Provincia (si coincide exactamente con una provincia)
    if span_l in SPANISH_PROVINCES:
        return "Provincia"

    # Ciudad: heurística muy sencilla
    if base_entity.lower() == "dirección" and len(span.split()) == 1 and span[0].isupper():
        return "Ciudad"

    # Calle (si contiene "c/", "calle", "av.", "avenida", "paseo"...)
    if any(pref in span_l for pref in ["c/", "calle", "av.", "avenida", "paseo", "plaza", "pza."]):
        return "Dirección"

    # Fallback
    return base_entity


#  Validadores ligeros (Reducir FP)

def count_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


def is_valid_phone(text: str) -> bool:
    """
    Teléfono razonable:
      - Al menos 9 dígitos
      - No más de 15
    """
    n = count_digits(text)
    return 9 <= n <= 15


DNI_REGEX = re.compile(r"\b\d{8}[A-HJ-NP-TV-Z]\b", re.IGNORECASE)

def is_probable_dni_or_id(text: str) -> bool:
    # Evita que un pasaporte (ES1234567, ABC123456) se clasifique como DNI
    if re.search(r"\b[A-Z]{2,3}\d{6,9}\b", text):
        return False

    # DNI español estándar
    if DNI_REGEX.search(text):
        return True

    # Fallback genérico para otros IDs
    raw = re.sub(r"[^A-Za-z0-9]", "", text)
    return len(raw) >= 6


COMMON_NON_NAME_TOKENS = {
    "hospital", "clínica", "clinica", "servicio", "unidad", "urgencias",
    "consulta", "paciente", "doctor", "doctora", "dra", "dr",
    "fecha", "ingreso", "alta", "diagnóstico", "diagnostico"
}


def trim_person_name_span(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Recorta spans de nombre para eliminar colas típicas:
    ', de 52 años', ', de', ', 52 años', etc.
    """
    span = text[start:end]

    # corta por coma seguida de 'de', dígitos o edad
    m = re.search(r",\s*(de\b|\d|\d+\s*a[nñ]os?)", span, flags=re.IGNORECASE)
    if m:
        end = start + m.start()

    # recorte final de puntuación
    while end > start and text[end - 1] in " ,;:":
        end -= 1

    return start, end


NAME_CONNECTORS = {"de", "del", "la", "las", "los", "y", "da", "do", "dos"}

def is_probable_person_name(span: str) -> bool:
    """
    Heurística para bajar FP en NOMBRE_*:
    - sin dígitos
    - al menos 2 tokens (nombre + apellido)
    - tokens empiezan por mayúscula (permitiendo acentos)
    - evita palabras clínicas frecuentes
    """
    s = span.strip()
    if not s:
        return False
    if any(ch.isdigit() for ch in s):
        return False

    # normaliza separadores y quita títulos típicos
    s = re.sub(r"\s+", " ", s)
    s_low = s.lower().strip(".:,; ")

    # elimina prefijos Dr/Dra si vienen pegados
    s_low = re.sub(r"^(dr|dra)\.?\s+", "", s_low, flags=re.IGNORECASE)

    toks = [t for t in s.split() if t.strip()]
    if len(toks) < 2:
        return False

    # si contiene tokens clínicos típicos, descartamos
    if any(t.lower().strip(".:,;") in COMMON_NON_NAME_TOKENS for t in toks):
        return False

    for t in toks:
        t_clean = t.strip(".:,;")
        if not t_clean:
            return False

        t_low = t_clean.lower()
        if t_low in NAME_CONNECTORS:
            continue

        if not re.match(r"^[A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-']+$", t_clean):
            return False


    return True


TITLE_STRONG_LEFT = re.compile(
    r"(?:\bSr\.?|\bSra\.?|\bDr\.?|\bDra\.?)\s*$",
    re.IGNORECASE
)

def is_single_surname_with_title(text: str, start: int, span: str) -> bool:
    """
    Acepta apellidos únicos SOLO si van precedidos de un título fuerte:
    'Sr. Sánchez', 'Dr. Pérez', etc.
    """
    if len(span.split()) != 1:
        return False

    if not re.match(r"^[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ\-']+$", span):
        return False

    left = text[max(0, start - 10):start]
    return bool(TITLE_STRONG_LEFT.search(left))



# Normalización de spans

TRIM_CHARS = " \t\n\r,;:.-()[]{}\"'"

PREFIX_BY_ENTITY = {
    "TELEFONO": [r"tel(?:éfono)?[:\s]*", r"tlf[:\s]*", r"tfno[:\s]*"],
    "EMAIL": [r"correo(?:\s+electr[oó]nico)?[:\s]*", r"email[:\s]*", r"e-?mail[:\s]*"],
    "DNI_NIF": [r"dni[:\s]*", r"nif[:\s]*"],
    "NHC": [
    r"(?:n[º°]\s*)?historia\s+cl[ií]nica[:\s-]*",
    r"nhc[:\s-]*",
    
    ],
    "DEVICE_ID": [ r"dispositivo[:\s]*",r"device[:\s]*"],

    "IP": [r"ip[:\s]*", r"direcci[oó]n\s+ip[:\s]*"],
    "DIRECCION": [r"dir(?:ecci[oó]n)?[:\s]*", r"domicilio[:\s]*"],
}

def normalize_span(text: str, start: int, end: int, ent_canon: str) -> tuple[int, int]:
    """
    Ajusta start/end para quitar basura típica:
    - espacios/puntuación periférica
    - prefijos tipo "teléfono:", "DNI:", "HC:", "IP:", "correo:"
    """
    s, e = int(start), int(end)
    if s < 0: s = 0
    if e > len(text): e = len(text)
    if e <= s:
        return s, e

    # 1) recorte de caracteres periféricos
    while s < e and text[s] in TRIM_CHARS:
        s += 1
    while e > s and text[e - 1] in TRIM_CHARS:
        e -= 1

    if e <= s:
        return s, e

    # 2) quitar prefijo si quedó dentro del span
    span = text[s:e]
    for pref in PREFIX_BY_ENTITY.get(ent_canon, []):
        m = re.match(pref, span, flags=re.IGNORECASE)
        if m:
            s += m.end()
            break

    # 3) recorte final tras prefijo
    while s < e and text[s] in TRIM_CHARS:
        s += 1
    while e > s and text[e - 1] in TRIM_CHARS:
        e -= 1


    # 4) Ajuste específico EDAD: recorta sufijos tipo "años", "a.", "a"
    if ent_canon == "EDAD":
        span2 = text[s:e]
        m_age = re.match(r"^\s*(\d{1,3})\s*(?:años?|a\.?)?\s*$", span2, flags=re.IGNORECASE)
        if m_age:
            num = m_age.group(1)
            num_pos = span2.find(num)
            if num_pos != -1:
                s = s + num_pos
                e = s + len(num)


    return s, e



TITLE_LEFT_PAT = re.compile(r"(?:\bDr\.?|\bDra\.?|\bSr\.?|\bSra\.?)\s*$", re.IGNORECASE)

def expand_span_left_titles(text: str, start: int, end: int, ent_canon: str) -> tuple[int, int]:
    """
    Si la entidad es un nombre de profesional/paciente, incluye títulos tipo 'Dr.' si están pegados a la izquierda.
    Mejora STRICT cuando el GOLD incluye el título.
    """
    if ent_canon not in {"NOMBRE_PROFESIONAL", "NOMBRE_PACIENTE"}:
        return start, end

    s, e = start, end
    left = text[max(0, s - 10):s]  # mira un poco a la izquierda
    m = TITLE_LEFT_PAT.search(left)
    if m:
        s = max(0, s - (len(left) - m.start()))
        while s < e and text[s].isspace():
            s += 1
    return s, e



# Reglas regex desde guidelines

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

    # Email 
    if "MAIL" in ent_up or "CORREO" in ent_up or "EMAIL" in ent_up:
        return r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    # Edad: número + "años" o "a."
    if entity.strip().lower() == "edad":
        return r"\b\d{1,3}\s*(?:años?|a\.?)\b"

    # Teléfono
    if "TELEFONO" in ent_up or "TELÉFONO" in ent_up:
        # prefijo opcional +34, 9-12 dígitos reales
        return r"(?:\+?\d{1,3}[ ]?)?(?:\d[ ]?){9,12}"


    # IP
    if entity.strip().lower() in {"ip", "dirección ip", "direccion ip"}:
        return r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

    # Nº Historia Clínica / NHC
    if "HISTORIA" in ent_up or "NHC" in ent_up:
        # "Nº Historia Clínica: HC-0099123" o "HC-0099123"
        return r"\b(?:N[º°]\s*)?Historia\s+Cl[ií]nica[:\s-]*[A-Z0-9\-]{4,}\b|\bHC-?[A-Z0-9]{4,}\b"

    # DNI/NIF
    if "DNI" in ent_up or "NIF" in ent_up:
        return r"\b\d{8}[A-HJ-NP-TV-Z]\b"

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
            # Fallback HIPAA-like usando entidad canónica
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





def get_extra_rules() -> List[Rule]:
    """
    Reglas extra “mínimas” que complementan Guidelines.
    Úsalas tanto en el main como en notas sintéticas para consistencia.
    """
    return [
        # NHC / MRN
        Rule(
            entity="NHC",
            pattern=r"\b(?:N[º°]\s*)?Historia\s+Cl[ií]nica[:\s-]*((?:HC)-?[A-Z0-9]{4,})\b|\b((?:MRN)-?[A-Z0-9]{4,})\b|\b((?:HC)-?[A-Z0-9]{4,})\b",
            replacement="<MRN>",
            obligatorio=True
        ),

        # ID Seguro / INS
        Rule(
            entity="INSURANCE_ID",
            pattern=r"\bINS-?\s*\d{2,}(?:-\d{2,})+\b",
            replacement="<INSURANCE_ID>",
            obligatorio=True
        ),

        # IDs de dispositivo / internos (DEV-..., XK..., etc.)
        Rule(
            entity="DEVICE_ID",
            pattern=r"\bDEV-[A-Z0-9\-]{3,}\b|\bXK-[A-Z0-9\-]{3,}\b",
            replacement="<DEVICE_ID>",
            obligatorio=False
        ),

        # Fecha de nacimiento (captura fecha tras etiquetas típicas)
        Rule(
            entity="FECHA_NACIMIENTO",
            pattern=r"(?i)\b(?:fecha\s*de\s*nacimiento|nacimiento|fnac|f\.\s*nac|nac\.|naci[oó]|nacido|nacida)\b"
                    r"[^0-9]{0,15}"
                    r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b",
            replacement="<DATE_BIRTH>",
            obligatorio=False
        ),

        # Pasaporte (evita DNI)
        Rule(
            entity="PASAPORTE",
            pattern=r"\b(?!\d{8}[A-HJ-NP-TV-Z]\b)([A-Z]{2,3}\d{6,9}|\d{6,9}[A-Z]{2,3})\b",
            replacement="<PASSPORT>",
            obligatorio=False
        ),

        # Fecha de ingreso / admisión
        Rule(
            entity="FECHA_INGRESO",
            pattern=r"(?i)\b(?:fecha\s*de\s*ingreso|ingreso|admis(?:i[oó]n)?)\s*[:\-]?\s*"
                    r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b",
            replacement="<DATE_ADMISSION>",
            obligatorio=False
        ),



                # Sr./Sra. + Apellido (mención abreviada típica) -> paciente
        Rule(
            entity="NOMBRE_PACIENTE",
            pattern=r"\b(?:Sr\.?|Sra\.?)\s+([A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-']{2,})\b",
            replacement="<PATIENT_NAME>",
            obligatorio=False
        ),

        # Dr./Dra. + Apellido (mención abreviada típica) -> profesional
        Rule(
            entity="NOMBRE_PROFESIONAL",
            pattern=r"\b(?:Dr\.?|Dra\.?)\s+([A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-']{2,})\b",
            replacement="<PROVIDER_NAME>",
            obligatorio=False
        ),

    ]


def build_all_rules(df_guidelines: pd.DataFrame) -> List[Rule]:
    """
    Construye TODAS las reglas: las de Guidelines + EXTRA_RULES.
    """
    rules = build_rules_from_guidelines(df_guidelines)
    rules.extend(get_extra_rules())
    return rules






# NER MEDDOCAN 

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
    "ID_ASEGURAMIENTO": "Nº Seguro Médico",
    "FECHAS": "FECHAS",
}


def map_ner_label_to_entity(label: str) -> str:
    clean = label.replace("B-", "").replace("I-", "").upper()
    return NER_LABEL_MAP.get(clean, clean)

def detect_entities_ner(id_texto: int, texto: str, ner_pipeline) -> List[Prediction]:
    preds: List[Prediction] = []
    outputs = ner_pipeline(texto)

    for ent in outputs:
        start = int(ent["start"])
        end = int(ent["end"])
        score = float(ent.get("score", 1.0))
        raw_label = ent.get("entity_group") or ent.get("entity", "")

        # Map MEDDOCAN → etiqueta humana → canónica
        human_label = map_ner_label_to_entity(raw_label)
        ent_canon = canonicalize_entity_name(human_label)

        # Umbral por entidad
        thr = NER_SCORE_BY_ENTITY.get(ent_canon, NER_SCORE_THRESHOLD)
        if score < thr:
            continue

        span_raw = texto[start:end]

        # Edad: exige "años/a."
        if ent_canon == "EDAD" and not re.search(r"(?:años?|a\.?)\b", span_raw.lower()):
            continue

        # Filtro anti-FP para nombres
        if ent_canon in {"NOMBRE_PACIENTE", "NOMBRE_PROFESIONAL"}:
            if not is_probable_person_name(span_raw):
                # Permitimos apellido único SOLO si va con título (Sr./Dr.)
                if not is_single_surname_with_title(texto, start, span_raw):
                    continue


        # Resolver PACIENTE vs PROFESIONAL por contexto
        if ent_canon in {"NOMBRE_PACIENTE", "NOMBRE_PROFESIONAL"}:
            ent_canon = classify_person_role_by_context(texto, start, end, default=ent_canon)

        # Fechas genéricas → específicas por contexto
        if ent_canon in {"FECHA", "FECHAS"}:
            ent_canon = classify_date_entity(texto, start, end)

        # INSURANCE_ID mal clasificado como IP
        if ent_canon == "INSURANCE_ID" and looks_like_ip(span_raw):
            ent_canon = "IP"

        # Validadores
        if ent_canon == "TELEFONO" and not is_valid_phone(span_raw):
            continue
        if ent_canon == "DNI_NIF" and not is_probable_dni_or_id(span_raw):
            continue
        if ent_canon == "IP" and not looks_like_ip(span_raw):
            continue

        # Normaliza bordes
        ns, ne = normalize_span(texto, start, end, ent_canon)
        ns, ne = expand_span_left_titles(texto, ns, ne, ent_canon)
        if ne <= ns:
            continue

        span_text = texto[ns:ne]

        preds.append(
            Prediction(
                id_texto=id_texto,
                start=ns,
                end=ne,
                entity=ent_canon,
                text=span_text,
                score=score,
            )
        )

    # ------------------------------------------------------------------
    # FALLBACK: Nombre al inicio tipo "Roberto Sánchez, ..."
    # Solo si el NER NO detectó ningún nombre en este texto.
    # ------------------------------------------------------------------
    has_any_name = any(p.entity in {"NOMBRE_PACIENTE", "NOMBRE_PROFESIONAL"} for p in preds)
    if not has_any_name:
        m = re.search(
            r"^\s*([A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ'\-]+(?:\s+[A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ'\-]+)+)(?=,)",
            texto
        )
        if m:
            s, e = m.start(1), m.end(1)
            candidate = texto[s:e]
            if is_probable_person_name(candidate):
                ent_canon = classify_person_role_by_context(texto, s, e, default="NOMBRE_PACIENTE")

                ns, ne = normalize_span(texto, s, e, ent_canon)
                ns, ne = expand_span_left_titles(texto, ns, ne, ent_canon)
                if ne > ns:
                    preds.append(
                        Prediction(
                            id_texto=id_texto,
                            start=ns,
                            end=ne,
                            entity=ent_canon,
                            text=texto[ns:ne],
                            score=0.35,  # score “sintético” (fallback)
                        )
                    )

    return preds





def detect_entities_regex(id_texto: int, texto: str, rules: List[Rule]) -> List[Prediction]:
    preds: List[Prediction] = []

    for rule in rules:
        pattern = re.compile(rule.pattern, flags=re.IGNORECASE)

        for m in pattern.finditer(texto):
            # Si el regex tiene grupos, usa el PRIMERO que realmente haya matcheado (no siempre es el 1)
            if m.lastindex and m.lastindex >= 1:
                gidx = None
                for gi in range(1, m.lastindex + 1):
                    if m.group(gi) is not None:
                        gidx = gi
                        break
                if gidx is not None:
                    start, end = m.start(gidx), m.end(gidx)
                else:
                    start, end = m.start(), m.end()
            else:
                start, end = m.start(), m.end()


            ent_canon = canonicalize_entity_name(rule.entity)
            if ent_canon in {"NOMBRE_PACIENTE", "NOMBRE_PROFESIONAL"}:
                ent_canon = classify_person_role_by_context(texto, start, end, default=ent_canon)


            # Reclasificar FECHA por contexto
            if ent_canon == "FECHA":
                ent_canon = classify_date_entity(texto, start, end)

            # Normaliza bordes (quita "teléfono:", "DNI:", etc.)
            ns, ne = normalize_span(texto, start, end, ent_canon)
            ns, ne = expand_span_left_titles(texto, ns, ne, ent_canon)
            if ne <= ns:
                continue

            span_text = texto[ns:ne]

            # Validadores
            if ent_canon == "TELEFONO" and not is_valid_phone(span_text):
                continue
            if ent_canon == "DNI_NIF" and not is_probable_dni_or_id(span_text):
                continue
            if ent_canon == "IP" and not looks_like_ip(span_text):
                continue
            # ---- FIX: aceptar apellido único solo con título (Sr./Dr.) ----
            if ent_canon in {"NOMBRE_PACIENTE", "NOMBRE_PROFESIONAL"}:
                if not is_probable_person_name(span_text):
                    if not is_single_surname_with_title(texto, ns, span_text):
                        continue


            preds.append(
                Prediction(
                    id_texto=id_texto,
                    start=ns,
                    end=ne,
                    entity=ent_canon,
                    text=span_text,
                    score=1.0,
                )
            )

    return preds



# ========= COMBINACIÓN NER + REGEX =========

def spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return not (e1 <= s2 or e2 <= s1)


def merge_predictions(preds_a: List[Prediction], preds_b: List[Prediction]) -> List[Prediction]:
    """
    Fusión avanzada evitando duplicados:
      - Unifica predicciones de NER y regex.
      - Si hay mismo ID + entidad canónica + solape de span, se queda con la de mayor score.
    """
    merged: List[Prediction] = []

    all_preds = preds_a + preds_b

    # Ordenamos de mayor a menor score para conservar primero las "mejores"
    all_preds.sort(key=lambda p: p.score, reverse=True)

    for p in all_preds:
        keep = True
        p_canon = canonicalize_entity_name(p.entity)
        for q in merged:
            q_canon = canonicalize_entity_name(q.entity)
            if (
                p.id_texto == q.id_texto
                and p_canon == q_canon
                and spans_overlap(p.start, p.end, q.start, q.end)
            ):
                # Ya tenemos una detección para esa entidad canónica y span solapado en ese texto
                keep = False
                break
        if keep:
            merged.append(p)

    return merged


ENTITY_PRIORITY = {
    "INSURANCE_ID": 110,
    "DNI_NIF": 100,
    "PASAPORTE": 98,
    "NHC": 95,
    "DEVICE_ID": 90,
    "EMAIL": 85,
    "TELEFONO": 85,
    "IP": 80,
    "NOMBRE_PACIENTE": 70,
    "NOMBRE_PROFESIONAL": 70,
    "FECHA_NACIMIENTO": 60,
    "FECHA_INGRESO": 60,
    "FECHA_ALTA": 60,
    "FECHA": 50,
    "HOSPITAL": 40,
    "DIRECCION": 30,
}


def resolve_overlaps_global(preds: List[Prediction]) -> List[Prediction]:
    """
    Elimina solapes priorizando entidades críticas.
    Mantiene primero: prioridad entidad desc, score desc, longitud desc.
    """
    preds_sorted = sorted(
        preds,
        key=lambda p: (
            -ENTITY_PRIORITY.get(canonicalize_entity_name(p.entity), 10),
            -p.score,
            -(p.end - p.start),
            p.start
        )
    )

    kept: List[Prediction] = []
    for p in preds_sorted:
        if any(spans_overlap(p.start, p.end, q.start, q.end) for q in kept):
            continue
        kept.append(p)

    # devuelve ordenado por aparición en texto
    return sorted(kept, key=lambda x: x.start)



def detect_all_texts(df_textos: pd.DataFrame, rules: List[Rule], ner_pipeline, merge_address: bool = True) -> List[Prediction]:
    """
    Ejecuta detección híbrida (NER + regex) para todos los textos.
    Aplica:
      - merge NER/regex evitando duplicados
      - resolve_overlaps_global con prioridades
      - (opcional) merge_nearby_same_entity para DIRECCION si existe la función
    """
    all_preds: List[Prediction] = []

    for _, row in df_textos.iterrows():
        tid = int(row["ID"])
        ttext = str(row["Texto"])

        preds_ner = detect_entities_ner(tid, ttext, ner_pipeline)
        preds_regex = detect_entities_regex(tid, ttext, rules)

        preds_combined = merge_predictions(preds_ner, preds_regex)
        preds_combined = resolve_overlaps_global(preds_combined)
        if tid == 4:
            print("\n[DEBUG ID=4] preds_ner:")
            for p in preds_ner:
                print(p.entity, p.start, p.end, repr(p.text))

            print("\n[DEBUG ID=4] preds_regex:")
            for p in preds_regex:
                print(p.entity, p.start, p.end, repr(p.text))

            print("\n[DEBUG ID=4] preds_final (pre-merge-address):")
            for p in preds_combined:
                print(p.entity, p.start, p.end, repr(p.text))

        # Si tienes merge_nearby_same_entity definido, úsalo
        if merge_address:
            preds_combined = merge_nearby_same_entity(preds_combined, ttext, entity="DIRECCION", max_gap=3)

        if tid == 4 and merge_address:
            print("\n[DEBUG ID=4] preds_final (post-merge-address):")
            for p in preds_combined:
                print(p.entity, p.start, p.end, repr(p.text))


        all_preds.extend(preds_combined)

    return all_preds



#  ANONIMIZACIÓN (HIPAA-LIKE) 

# Placeholders HIPAA-like por entidad canónica
HIPAA_PLACEHOLDERS = {
    "NOMBRE_PACIENTE": "<PATIENT_NAME>",
    "NOMBRE_PROFESIONAL": "<PROVIDER_NAME>",
    "HOSPITAL": "<HOSPITAL_NAME>",
    "DIRECCION": "<ADDRESS>",
    "TELEFONO": "<PHONE>",
    "EMAIL": "<EMAIL>",
    "EDAD": "<AGE>",
    "FECHA_NACIMIENTO": "<DATE_BIRTH>",
    "FECHA_INGRESO": "<DATE_ADMISSION>",
    "FECHA_ALTA": "<DATE_DISCHARGE>",
    "FECHA": "<DATE>",
    "DNI_NIF": "<ID_DOC>",
    "NHC": "<MRN>",
    "INSURANCE_ID": "<INSURANCE_ID>",
    "IP": "<IP_ADDRESS>",
    "DEVICE_ID": "<DEVICE_ID>",
    "PASAPORTE": "<PASSPORT>",


}

def merge_nearby_same_entity(preds: List[Prediction], text: str, entity: str, max_gap: int = 2) -> List[Prediction]:
    """
    Une spans de la misma entidad si están muy cerca (gap <= max_gap),
    por ejemplo DIRECCION partida en 'C/..' + '28013' + 'Madrid'.
    """
    # separa los preds objetivo
    target = [p for p in preds if p.entity == entity]
    others = [p for p in preds if p.entity != entity]

    target.sort(key=lambda p: p.start)

    merged = []
    for p in target:
        if not merged:
            merged.append(p)
            continue
        last = merged[-1]

        gap = p.start - last.end
        between = text[last.end:p.start]

        # permitimos gaps cortos y separadores típicos
        if gap <= max_gap and re.fullmatch(r"[\s,;/\-]*", between or ""):
            new_start = last.start
            new_end = p.end
            new_text = text[new_start:new_end]
            new_score = max(last.score, p.score)
            merged[-1] = Prediction(
                id_texto=last.id_texto,
                start=new_start,
                end=new_end,
                entity=entity,
                text=new_text,
                score=new_score,
            )
        else:
            merged.append(p)

    return others + merged



def anonymize_text(text: str, preds: List[Prediction], rules_by_entity: Dict[str, Rule]) -> str:
    spans: List[Tuple[int, int, str]] = []

    for p in preds:
        # 1) Intentar usar Guidelines (si hay regla para esa entidad exacta)
        
        rule = rules_by_entity.get(canonicalize_entity_name(p.entity)) or rules_by_entity.get(p.entity)


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
    # Indexamos por entidad CANÓNICA y también por el nombre "humano" del Excel
    rules_by_entity = {}
    for r in rules:
        rules_by_entity[canonicalize_entity_name(r.entity)] = r
        rules_by_entity[r.entity] = r  # por si en algún caso llega "tal cual"


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
    exact match en: ID, Start, End, Entidad_canon.
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


def _overlap_len(a_s: int, a_e: int, b_s: int, b_e: int) -> int:
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def evaluate_lenient(gold: pd.DataFrame, pred: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluación lenient (mejorada):
    - ID igual
    - Entidad canónica igual
    - Cualquier solape cuenta, pero emparejamos por MAYOR solape (greedy)
    """
    g = add_canonical_to_gold(gold)
    p = pred.copy()

    g_rows = list(g.to_dict("records"))
    p_rows = list(p.to_dict("records"))

    used_pred = set()
    rows = []

    for gr in g_rows:
        candidates = []
        for pj, pr in enumerate(p_rows):
            if pj in used_pred:
                continue
            if gr["ID"] != pr["ID"]:
                continue
            if gr["Entidad_canon"] != pr["Entidad_pred_canon"]:
                continue
            if not spans_overlap(gr["Start"], gr["End"], pr["Start_pred"], pr["End_pred"]):
                continue

            ov = _overlap_len(gr["Start"], gr["End"], pr["Start_pred"], pr["End_pred"])
            candidates.append((ov, pj, pr))

        if candidates:
            # Elegimos el de mayor solape; empate → el primero
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_pj, best_pr = candidates[0]
            used_pred.add(best_pj)

            rows.append(
                {
                    "ID": gr["ID"],
                    "Entidad_Gold": gr["Entidad"],
                    "Entidad_Gold_canon": gr["Entidad_canon"],
                    "Texto_Gold": gr["Texto"],
                    "Start_Gold": gr["Start"],
                    "End_Gold": gr["End"],
                    "Entidad_Pred": best_pr["Entidad_pred"],
                    "Entidad_Pred_canon": best_pr["Entidad_pred_canon"],
                    "Texto_Pred": best_pr["Texto_pred"],
                    "Start_Pred": best_pr["Start_pred"],
                    "End_Pred": best_pr["End_pred"],
                    "Match": "✓",
                    "Error_Type": "TP",
                    "Eval": "lenient",
                }
            )
        else:
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

    # Predicciones no usadas = FP
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

    metrics = {"TP": TP, "FP": FP, "FN": FN, "Precision": precision, "Recall": recall, "F1": f1}
    return matching_df, metrics



def analyze_lenient_vs_strict(gold: pd.DataFrame, pred: pd.DataFrame):
    g = add_canonical_to_gold(gold)
    p = pred.copy()

    # index preds por (ID, entidad_canon)
    p_group = {}
    for _, pr in p.iterrows():
        key = (int(pr["ID"]), str(pr["Entidad_pred_canon"]))
        p_group.setdefault(key, []).append(pr)

    lenient_hit = 0
    strict_hit = 0

    for _, gr in g.iterrows():
        key = (int(gr["ID"]), str(gr["Entidad_canon"]))
        candidates = p_group.get(key, [])
        any_overlap = False
        any_exact = False
        for pr in candidates:
            if spans_overlap(int(gr["Start"]), int(gr["End"]), int(pr["Start_pred"]), int(pr["End_pred"])):
                any_overlap = True
            if int(gr["Start"]) == int(pr["Start_pred"]) and int(gr["End"]) == int(pr["End_pred"]):
                any_exact = True
        if any_overlap:
            lenient_hit += 1
        if any_exact:
            strict_hit += 1

    print(f"\n[LENIENT-vs-STRICT] Gold con algún solape (lenient-hit): {lenient_hit}")
    print(f"[LENIENT-vs-STRICT] Gold con match exacto de span (strict-hit): {strict_hit}")
    if lenient_hit > 0:
        print(f"[LENIENT-vs-STRICT] % exactos entre los que solapan: {strict_hit/lenient_hit:.3f}")



def report_by_entity(df_gold_fixed: pd.DataFrame, df_preds: pd.DataFrame):
    g = add_canonical_to_gold(df_gold_fixed)
    p = df_preds.copy()

    # STRICT por entidad
    g["key"] = g["ID"].astype(str)+"|"+g["Start"].astype(str)+"|"+g["End"].astype(str)+"|"+g["Entidad_canon"].astype(str)
    p["key"] = p["ID"].astype(str)+"|"+p["Start_pred"].astype(str)+"|"+p["End_pred"].astype(str)+"|"+p["Entidad_pred_canon"].astype(str)

    gold_keys = set(g["key"])
    pred_keys = set(p["key"])

    tp = gold_keys & pred_keys
    fn = gold_keys - pred_keys
    fp = pred_keys - gold_keys

    def ent_from_key(k):  # último campo
        return k.split("|")[-1]

    ents = sorted(set(g["Entidad_canon"]) | set(p["Entidad_pred_canon"]))
    rows = []
    for e in ents:
        TP = sum(1 for k in tp if ent_from_key(k) == e)
        FN = sum(1 for k in fn if ent_from_key(k) == e)
        FP = sum(1 for k in fp if ent_from_key(k) == e)
        prec = TP/(TP+FP) if TP+FP else 0.0
        rec = TP/(TP+FN) if TP+FN else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        rows.append({"Entidad": e, "TP": TP, "FP": FP, "FN": FN, "Precision": prec, "Recall": rec, "F1": f1})

    out = pd.DataFrame(rows).sort_values(["F1","FN","FP"], ascending=[True, False, False])
    print("\n[STRICT] Métricas por entidad (peores arriba):")
    print(out.to_string(index=False))


# ========= MAIN =========

if __name__ == "__main__":
    # 1) Cargar datos
    df_guidelines = load_guidelines()
    df_textos = load_textos()

    df_gold = load_gold()
    # 0) Check offsets originales
    sanity_check_gold_offsets(df_textos, df_gold, max_print=30)
    # 1) Reparar offsets del GOLD contra el texto real actual
    df_gold_fixed, df_gold_repair_report = repair_gold_offsets(df_textos, df_gold, max_print=30)
    # 2) Check offsets tras reparación (debería bajar muchísimo "bad")
    sanity_check_gold_offsets(df_textos, df_gold_fixed, max_print=10)



    print("Guidelines cargados:")
    print(df_guidelines.head())


    # 2) Construir reglas regex (Guidelines + EXTRA_RULES)
    rules = build_all_rules(df_guidelines)
    print(f"\nTotal reglas regex (Guidelines + Extra): {len(rules)}")



    # 3) Cargar modelo NER
    ner_pipe = load_ner_pipeline()

    print("\nLanzando detección híbrida (NER + regex)...")

    # A) Para EVALUACIÓN: NO fusionamos DIRECCION (para que case con spans del GOLD)
    all_preds_eval = detect_all_texts(df_textos, rules, ner_pipe, merge_address=False)
    print(f"Total entidades (eval, sin merge direcciones): {len(all_preds_eval)}")

    df_preds = preds_to_dataframe(all_preds_eval)
    report_by_entity(df_gold_fixed, df_preds)
    analyze_lenient_vs_strict(df_gold_fixed, df_preds)

    print("\nPrimeras predicciones (eval):")
    print(df_preds.head(20))

    # B) Para ANONIMIZACIÓN: SÍ fusionamos DIRECCION (mejor para producción)
    all_preds_anon = detect_all_texts(df_textos, rules, ner_pipe, merge_address=True)
    print(f"Total entidades (anon, con merge direcciones): {len(all_preds_anon)}")

    # 5) Anonimizar textos
    df_anon = anonymize_all_texts(df_textos, all_preds_anon, rules)

    # 6) Evaluar (estricto + lenient) SOBRE GOLD REPARADO
    matching_strict, metrics_strict = evaluate_strict(df_gold_fixed, df_preds)
    matching_lenient, metrics_lenient = evaluate_lenient(df_gold_fixed, df_preds)


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
        df_gold_fixed.to_excel(writer, sheet_name="Gold_Standard_FIXED", index=False)
        df_gold_repair_report.to_excel(writer, sheet_name="Gold_Repair_Report", index=False)
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
