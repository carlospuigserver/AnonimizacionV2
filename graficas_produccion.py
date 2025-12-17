import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
PATH = "Resultados_Anonimizador_Produccion.xlsx"  # cambia si quieres
OUT_DIR = "graficas_out_prod"
SHOW = True
SAVE = True

os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name: str):
    if SAVE:
        out = os.path.join(OUT_DIR, name)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[OK] Guardado: {out}")

def main():
    print(f"[INFO] Leyendo Excel: {PATH}")
    stats = pd.read_excel(PATH, sheet_name="Stats_By_Note")
    ent_global = pd.read_excel(PATH, sheet_name="Entity_Counts_Global")

    # 1) % notas con cambios
    pct_changed = 100.0 * stats["cambio"].mean() if len(stats) else 0.0
    plt.figure()
    plt.bar(["% con cambios", "% sin cambios"], [pct_changed, 100 - pct_changed])
    plt.title("Cobertura de anonimización (notas con cambios)")
    plt.ylabel("% notas")
    plt.ylim(0, 100)
    plt.tight_layout()
    savefig("01_cobertura_cambios.png")
    if SHOW: plt.show()
    plt.close()

    # 2) Distribución nº entidades por nota (histograma)
    plt.figure()
    stats["n_entidades_total"].plot(kind="hist", bins=30)
    plt.title("Distribución: nº de entidades detectadas por nota")
    plt.xlabel("n_entidades_total")
    plt.ylabel("nº notas")
    plt.tight_layout()
    savefig("02_hist_entidades_por_nota.png")
    if SHOW: plt.show()
    plt.close()

    # 3) Top entidades globales
    top = ent_global.sort_values("n", ascending=False).head(15)
    plt.figure()
    top.set_index("Entidad_pred_canon")["n"].plot(kind="barh")
    plt.title("Top entidades detectadas (global)")
    plt.xlabel("conteo")
    plt.ylabel("entidad")
    plt.tight_layout()
    savefig("03_top_entidades_global.png")
    if SHOW: plt.show()
    plt.close()

    # 4) Ratio longitud anon/original (para detectar casos raros)
    plt.figure()
    stats["ratio_len_anon"].plot(kind="hist", bins=30)
    plt.title("Distribución: ratio len(anon) / len(original)")
    plt.xlabel("ratio_len_anon")
    plt.ylabel("nº notas")
    plt.tight_layout()
    savefig("04_hist_ratio_longitud.png")
    if SHOW: plt.show()
    plt.close()

    print(f"[FIN] Gráficas en: {OUT_DIR}")

if __name__ == "__main__":
    main()
