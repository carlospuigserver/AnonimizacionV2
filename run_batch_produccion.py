import argparse
import time
from pathlib import Path
import pandas as pd

from anonimizador import (
    load_guidelines,
    build_all_rules,
    load_ner_pipeline,
    detect_all_texts,
    preds_to_dataframe,
    anonymize_all_texts,
    canonicalize_entity_name,
)

def load_notes(input_path: str, sheet: str | None = None) -> pd.DataFrame:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el fichero: {p.resolve()}")

    if p.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        if not sheet:
            sheet = "Datos_Originales"
        df = pd.read_excel(p, sheet_name=sheet)
    else:
        # CSV / TSV
        df = pd.read_csv(p)

    # Normaliza columnas
    cols = {c.lower().strip(): c for c in df.columns}
    if "id" not in cols or ("texto" not in cols and "text" not in cols):
        raise ValueError("El input debe tener columnas 'ID' y 'Texto' (o 'text').")

    id_col = cols["id"]
    text_col = cols.get("texto", cols.get("text"))

    out = df[[id_col, text_col]].copy()
    out.columns = ["ID", "Texto"]
    out["ID"] = out["ID"].astype(int)
    out["Texto"] = out["Texto"].astype(str)
    return out

def compute_stats(df_textos: pd.DataFrame, df_preds: pd.DataFrame, df_anon: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - Stats_By_Note: por ID (longitudes, cambios, nº entidades, ratio redacción, etc.)
      - Entity_Counts_By_Note: tabla ancha por entidad canónica (conteos por ID)
      - Entity_Counts_Global: conteo global por entidad canónica
    """
    # Conteos por entidad por nota
    by_note_ent = (
        df_preds.groupby(["ID", "Entidad_pred_canon"])
        .size()
        .reset_index(name="n")
    )

    entity_wide = (
        by_note_ent.pivot(index="ID", columns="Entidad_pred_canon", values="n")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    # Stats por nota
    stats = df_anon[["ID", "Texto_original", "Texto_anon"]].copy()
    stats["len_original"] = stats["Texto_original"].astype(str).str.len()
    stats["len_anon"] = stats["Texto_anon"].astype(str).str.len()
    stats["ratio_len_anon"] = stats["len_anon"] / stats["len_original"].replace(0, 1)
    stats["cambio"] = (stats["Texto_original"] != stats["Texto_anon"]).astype(int)

    total_entities = df_preds.groupby("ID").size().rename("n_entidades_total").reset_index()
    stats = stats.merge(total_entities, on="ID", how="left")
    stats["n_entidades_total"] = stats["n_entidades_total"].fillna(0).astype(int)

    # añade columnas por entidad (wide)
    stats = stats.merge(entity_wide, on="ID", how="left").fillna(0)

    # global
    global_counts = (
        df_preds.groupby("Entidad_pred_canon")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    return stats, entity_wide, global_counts

def main():
    ap = argparse.ArgumentParser(description="Batch production runner: anonimiza y genera estadísticas para 1000+ notas.")
    ap.add_argument(
    "--input",
    default="notas_clinicas_sinteticas_1000.csv",
    help="Ruta a Excel/CSV con columnas ID y Texto."
    )
    ap.add_argument("--sheet", default=None, help="Hoja si el input es Excel (default: Datos_Originales).")
    ap.add_argument("--guidelines", default="Gold_Standard_Anonimizacion.xlsx", help="Excel de Guidelines (default: Gold_Standard_Anonimizacion.xlsx).")
    ap.add_argument("--out", default="Resultados_Anonimizador_Produccion.xlsx", help="Excel de salida.")
    ap.add_argument("--model", default=None, help="Nombre del modelo NER (si quieres sobrescribir).")
    ap.add_argument("--no_merge_address", action="store_true", help="Desactiva merge de DIRECCION por proximidad.")
    args = ap.parse_args()

    t0 = time.time()

    print("[1/5] Cargando notas...")
    df_textos = load_notes(args.input, args.sheet)
    print(f"      Notas: {len(df_textos)}")

    print("[2/5] Cargando Guidelines y construyendo reglas...")
    df_guidelines = load_guidelines(path=args.guidelines)
    rules = build_all_rules(df_guidelines)

    print("[3/5] Cargando modelo NER...")
    if args.model:
        ner_pipe = load_ner_pipeline(model_name=args.model)
    else:
        ner_pipe = load_ner_pipeline()

    print("[4/5] Detectando entidades (NER + regex)...")
    all_preds = detect_all_texts(df_textos, rules, ner_pipe, merge_address=(not args.no_merge_address))
    df_preds = preds_to_dataframe(all_preds)

    print("[5/5] Anonimizando textos y calculando stats...")
    df_anon = anonymize_all_texts(df_textos, all_preds, rules)
    stats_by_note, entity_counts_by_note, entity_counts_global = compute_stats(df_textos, df_preds, df_anon)

    # Resumen global rápido
    pct_changed = 100.0 * stats_by_note["cambio"].mean() if len(stats_by_note) else 0.0
    print(f"      % notas con cambios: {pct_changed:.1f}%")
    print("      Top entidades (global):")
    print(entity_counts_global.head(10).to_string(index=False))

    out_path = Path(args.out)
    print(f"[OUT] Guardando: {out_path.resolve()}")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_textos.to_excel(writer, sheet_name="Datos_Originales", index=False)
        df_preds.to_excel(writer, sheet_name="Predicciones_Modelo", index=False)
        df_anon.to_excel(writer, sheet_name="Textos_Anonimizados", index=False)

        stats_by_note.to_excel(writer, sheet_name="Stats_By_Note", index=False)
        entity_counts_global.to_excel(writer, sheet_name="Entity_Counts_Global", index=False)

        # tabla ancha puede ser grande; pero para 1000 notas va bien
        entity_counts_by_note.to_excel(writer, sheet_name="Entity_Counts_By_Note", index=False)

    dt = time.time() - t0
    print(f"[OK] Terminado en {dt:.1f}s")
    print("     Para revisar: streamlit run visor_notas.py -- (y apunta PATH al Excel de salida)")

if __name__ == "__main__":
    main()
