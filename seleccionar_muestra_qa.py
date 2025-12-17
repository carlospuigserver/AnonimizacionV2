import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Selecciona una muestra QA estratificada a partir del output de producción.")
    ap.add_argument("--in", dest="inp", required=True, help="Excel output (Resultados_Anonimizador_Produccion.xlsx).")
    ap.add_argument("--n", type=int, default=150, help="Tamaño muestra (default 150).")
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    ap.add_argument("--out", default="qa_sample_ids.csv", help="CSV de salida con IDs.")
    args = ap.parse_args()

    xls = Path(args.inp)
    if not xls.exists():
        raise FileNotFoundError(f"No existe: {xls.resolve()}")

    stats = pd.read_excel(xls, sheet_name="Stats_By_Note")
    # columnas esperadas: ID, len_original, n_entidades_total, cambio, ratio_len_anon
    if "ID" not in stats.columns or "n_entidades_total" not in stats.columns:
        raise ValueError("No encuentro columnas esperadas en Stats_By_Note.")

    rng = np.random.default_rng(args.seed)

    n = int(args.n)
    n = min(n, len(stats))

    # 1) 25% notas “más sensibles”: top por n_entidades_total
    k_top = max(1, int(round(0.25 * n)))
    top = stats.sort_values(["n_entidades_total", "len_original"], ascending=False).head(k_top)

    # 2) 25% notas largas (para direcciones/fechas con contexto)
    k_long = max(1, int(round(0.25 * n)))
    remaining = stats[~stats["ID"].isin(top["ID"])]
    long = remaining.sort_values("len_original", ascending=False).head(k_long)

    # 3) 25% notas con cambios “raros” por ratio (muy comprimidas o casi iguales)
    k_ratio = max(1, int(round(0.25 * n)))
    remaining = remaining[~remaining["ID"].isin(long["ID"])]
    # extremos del ratio: muy bajo o muy alto
    remaining["ratio_rank"] = (remaining["ratio_len_anon"] - remaining["ratio_len_anon"].median()).abs()
    ratio_extreme = remaining.sort_values("ratio_rank", ascending=False).head(k_ratio)

    # 4) Resto: muestreo aleatorio estratificado por “bins” de entidades
    remaining = remaining[~remaining["ID"].isin(ratio_extreme["ID"])]
    k_rest = n - (len(top) + len(long) + len(ratio_extreme))
    k_rest = max(0, k_rest)

    if k_rest > 0 and len(remaining) > 0:
        # bins por nº entidades: 0, 1-2, 3-5, 6-10, >10
        bins = [-0.1, 0.5, 2.5, 5.5, 10.5, float("inf")]
        labels = ["0", "1-2", "3-5", "6-10", ">10"]
        remaining["ent_bin"] = pd.cut(remaining["n_entidades_total"], bins=bins, labels=labels)

        picks = []
        # reparte proporcionalmente
        counts = remaining["ent_bin"].value_counts(dropna=False)
        total = counts.sum()
        for b, c in counts.items():
            if c <= 0:
                continue
            quota = int(round(k_rest * (c / total)))
            subset = remaining[remaining["ent_bin"] == b]
            if len(subset) <= quota:
                picks.append(subset)
            else:
                idx = rng.choice(subset.index.to_numpy(), size=quota, replace=False)
                picks.append(subset.loc[idx])

        rest = pd.concat(picks, ignore_index=True) if picks else remaining.sample(n=min(k_rest, len(remaining)), random_state=args.seed)
        # si por redondeos nos quedamos cortos/largos
        if len(rest) > k_rest:
            rest = rest.sample(n=k_rest, random_state=args.seed)
        elif len(rest) < k_rest:
            # completa con aleatorio simple
            leftover = remaining[~remaining["ID"].isin(rest["ID"])]
            need = k_rest - len(rest)
            if need > 0 and len(leftover) > 0:
                rest = pd.concat([rest, leftover.sample(n=min(need, len(leftover)), random_state=args.seed)], ignore_index=True)
    else:
        rest = stats.iloc[0:0].copy()

    sample = pd.concat([top, long, ratio_extreme, rest], ignore_index=True).drop_duplicates(subset=["ID"])
    # ajusta exactamente n si sobran
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=args.seed)

    out = Path(args.out)
    sample[["ID", "n_entidades_total", "len_original", "ratio_len_anon", "cambio"]].sort_values("ID").to_csv(out, index=False, encoding="utf-8")
    print(f"[OK] Muestra QA guardada: {out.resolve()} (n={len(sample)})")

if __name__ == "__main__":
    main()
