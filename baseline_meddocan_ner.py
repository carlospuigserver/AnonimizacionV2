import pandas as pd
from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# ========= CONFIGURACIÓN =========

# Ruta a tu Excel con los textos originales
EXCEL_PATH = "Gold_Standard_Anonimizacion.xlsx"
SHEET_TEXTOS = "Datos_Originales"   # pestaña con columnas: ID, Texto

# Nombre del modelo MEDDOCAN (BSC, clínico español)
MODEL_NAME = "BSC-NLP4BIA/bsc-bio-ehr-es-meddocan"

# Archivo de salida con las predicciones
OUTPUT_PRED_EXCEL = "Predicciones_Meddocan.xlsx"


# ========= ESTRUCTURA DE PREDICCIÓN =========

@dataclass
class Prediction:
    id_texto: int
    start: int
    end: int
    entity: str
    text: str
    score: float


# ========= CARGA DE TEXTOS =========

def load_textos(path: str = EXCEL_PATH, sheet_name: str = SHEET_TEXTOS) -> pd.DataFrame:
    """
    Espera columnas: ID, Texto
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df["ID"] = df["ID"].astype(int)
    df["Texto"] = df["Texto"].astype(str)
    return df


# ========= CARGA DEL MODELO NER =========

def load_ner_pipeline(model_name: str = MODEL_NAME):
    print(f"Cargando modelo NER MEDDOCAN: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # agrupa tokens B-/I- en una sola entidad
    )
    return ner


# ========= DETECCIÓN NER SOBRE UN TEXTO =========

def detect_entities_ner(text_id: int, text: str, ner_pipeline) -> List[Prediction]:
    preds: List[Prediction] = []
    outputs = ner_pipeline(text)

    for ent in outputs:
        start = int(ent["start"])
        end = int(ent["end"])
        # entity_group suele ser algo tipo "CALLE", "HAB", "PROFESIONAL", etc.
        label = str(ent.get("entity_group", ent.get("entity", "")))
        span_text = text[start:end]
        score = float(ent.get("score", 1.0))

        preds.append(
            Prediction(
                id_texto=text_id,
                start=start,
                end=end,
                entity=label,
                text=span_text,
                score=score,
            )
        )

    return preds


def detect_all_texts(df_textos: pd.DataFrame, ner_pipeline) -> List[Prediction]:
    all_preds: List[Prediction] = []

    for _, row in df_textos.iterrows():
        tid = int(row["ID"])
        ttext = str(row["Texto"])
        preds = detect_entities_ner(tid, ttext, ner_pipeline)
        all_preds.extend(preds)

    return all_preds


# ========= CONVERSIÓN A DATAFRAME Y EXPORT =========

def preds_to_dataframe(preds: List[Prediction]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": p.id_texto,
                "Start_pred": p.start,
                "End_pred": p.end,
                "Entidad_pred": p.entity,
                "Texto_pred": p.text,
                "Score": p.score,
            }
            for p in preds
        ]
    )


# ========= MAIN =========

if __name__ == "__main__":
    # 1) Cargar textos
    print(f"Leyendo textos desde: {EXCEL_PATH} / hoja '{SHEET_TEXTOS}'")
    df_textos = load_textos()
    print(f"Se han cargado {len(df_textos)} textos.")

    # 2) Cargar modelo NER
    ner_pipe = load_ner_pipeline()

    # 3) Detectar entidades en todos los textos
    print("Lanzando NER sobre los textos...")
    all_preds = detect_all_texts(df_textos, ner_pipe)
    print(f"Total de entidades detectadas: {len(all_preds)}")

    # 4) Pasar a DataFrame
    df_preds = preds_to_dataframe(all_preds)

    # 5) Mostrar un pequeño resumen en consola
    print("\nPrimeras predicciones:")
    print(df_preds.head(20))

    print("\nEntidades únicas detectadas (Entidad_pred):")
    print(df_preds["Entidad_pred"].value_counts())

    # 6) Guardar a Excel
    df_preds.to_excel(OUTPUT_PRED_EXCEL, sheet_name="Predicciones_Meddocan", index=False)
    print(f"\nPredicciones guardadas en: {OUTPUT_PRED_EXCEL}")
