import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
PATH = "Resultados_Anonimizador_Hibrido_v3.xlsx"
OUT_DIR = "graficas_out"
SHOW = True          # True: plt.show() / False: no mostrar (solo guardar)
SAVE = True          # True: guarda PNGs

os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name: str):
    if SAVE:
        out = os.path.join(OUT_DIR, name)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[OK] Guardado: {out}")

def safe_div(num, den):
    den = den.replace(0, 1)
    return num / den

def compute_by_entity_from_validation(df_validation: pd.DataFrame, canon_col: str) -> pd.DataFrame:
    """
    df_validation: hoja Validacion_Strict o Validacion_Lenient (tu output)
    canon_col: columna de entidad canónica en esa hoja:
        - strict: "Entidad_Gold_canon" existe
        - lenient: "Entidad_Gold_canon" también existe (y además pred)
    Devuelve DataFrame por entidad con TP/FP/FN + Precision/Recall/F1.
    """
    df = df_validation.copy()

    # Normaliza strings por si vienen con espacios raros
    df["Error_Type"] = df["Error_Type"].astype(str).str.strip()
    df[canon_col] = df[canon_col].astype(str).str.strip()

    by_ent = (
        df.groupby([canon_col, "Error_Type"])
          .size()
          .unstack(fill_value=0)
    )

    # Asegura columnas siempre
    for c in ["TP", "FP", "FN"]:
        if c not in by_ent.columns:
            by_ent[c] = 0

    by_ent["Precision"] = safe_div(by_ent["TP"], (by_ent["TP"] + by_ent["FP"]))
    by_ent["Recall"]    = safe_div(by_ent["TP"], (by_ent["TP"] + by_ent["FN"]))
    by_ent["F1"]        = safe_div(2 * by_ent["Precision"] * by_ent["Recall"],
                                   (by_ent["Precision"] + by_ent["Recall"]))

    # Ordena por peor F1 arriba
    by_ent = by_ent.sort_values("F1", ascending=True)
    return by_ent


# =========================
# CARGA
# =========================
print(f"[INFO] Leyendo Excel: {PATH}")

metrics = pd.read_excel(PATH, sheet_name="Metricas")
strict_val = pd.read_excel(PATH, sheet_name="Validacion_Strict")
lenient_val = pd.read_excel(PATH, sheet_name="Validacion_Lenient")

# =========================
# 1) MÉTRICAS GLOBALES (Precision/Recall/F1)
# =========================
pivot = metrics.pivot(index="Métrica", columns="Tipo", values="Valor")

# Asegura orden si existen
want = [m for m in ["Precision", "Recall", "F1"] if m in pivot.index]
pivot_main = pivot.loc[want]

plt.figure()
pivot_main.plot(kind="bar")
plt.title("Métricas globales: Strict vs Lenient")
plt.ylabel("Valor")
plt.xlabel("Métrica")
plt.ylim(0, 1.0)
plt.tight_layout()
savefig("01_metricas_globales.png")
if SHOW:
    plt.show()
plt.close()

# =========================
# 2) TP / FP / FN globales
# =========================
errs = metrics[metrics["Métrica"].isin(["TP", "FP", "FN"])].copy()
pivot_err = errs.pivot(index="Métrica", columns="Tipo", values="Valor")

plt.figure()
pivot_err.plot(kind="bar")
plt.title("Distribución de errores (global)")
plt.ylabel("Conteo")
plt.xlabel("Tipo de error")
plt.tight_layout()
savefig("02_errores_globales.png")
if SHOW:
    plt.show()
plt.close()

# =========================
# 3) F1 por entidad (STRICT)
# =========================
strict_only = strict_val[strict_val["Eval"].astype(str).str.strip().str.lower() == "strict"]
by_ent_strict = compute_by_entity_from_validation(strict_only, canon_col="Entidad_Gold_canon")

plt.figure()
by_ent_strict["F1"].plot(kind="barh")
plt.title("F1 por entidad (Strict)")
plt.xlabel("F1")
plt.ylabel("Entidad")
plt.xlim(0, 1.0)
plt.tight_layout()
savefig("03_f1_por_entidad_strict.png")
if SHOW:
    plt.show()
plt.close()

plt.figure()
by_ent_strict["FN"].plot(kind="barh")
plt.title("Falsos Negativos por entidad (Strict)")
plt.xlabel("FN")
plt.ylabel("Entidad")
plt.tight_layout()
savefig("04_fn_por_entidad_strict.png")
if SHOW:
    plt.show()
plt.close()

plt.figure()
by_ent_strict["FP"].plot(kind="barh")
plt.title("Falsos Positivos por entidad (Strict)")
plt.xlabel("FP")
plt.ylabel("Entidad")
plt.tight_layout()
savefig("05_fp_por_entidad_strict.png")
if SHOW:
    plt.show()
plt.close()

# =========================
# 4) F1 por entidad (LENIENT)
# =========================
lenient_only = lenient_val[lenient_val["Eval"].astype(str).str.strip().str.lower() == "lenient"]
by_ent_lenient = compute_by_entity_from_validation(lenient_only, canon_col="Entidad_Gold_canon")

plt.figure()
by_ent_lenient["F1"].plot(kind="barh")
plt.title("F1 por entidad (Lenient)")
plt.xlabel("F1")
plt.ylabel("Entidad")
plt.xlim(0, 1.0)
plt.tight_layout()
savefig("06_f1_por_entidad_lenient.png")
if SHOW:
    plt.show()
plt.close()

plt.figure()
by_ent_lenient["FN"].plot(kind="barh")
plt.title("Falsos Negativos por entidad (Lenient)")
plt.xlabel("FN")
plt.ylabel("Entidad")
plt.tight_layout()
savefig("07_fn_por_entidad_lenient.png")
if SHOW:
    plt.show()
plt.close()

plt.figure()
by_ent_lenient["FP"].plot(kind="barh")
plt.title("Falsos Positivos por entidad (Lenient)")
plt.xlabel("FP")
plt.ylabel("Entidad")
plt.tight_layout()
savefig("08_fp_por_entidad_lenient.png")
if SHOW:
    plt.show()
plt.close()

# =========================
# 5) TOP errores (opcional): lista rápida en consola
# =========================
print("\n[INFO] Top entidades con peor F1 (STRICT):")
print(by_ent_strict[["TP", "FP", "FN", "Precision", "Recall", "F1"]].head(10).to_string())

print("\n[INFO] Top entidades con peor F1 (LENIENT):")
print(by_ent_lenient[["TP", "FP", "FN", "Precision", "Recall", "F1"]].head(10).to_string())

print(f"\n[FIN] Gráficas en: {OUT_DIR}")
