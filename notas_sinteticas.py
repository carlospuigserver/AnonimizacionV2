import pandas as pd
from anonimizador import (
    # carga de datos base
    load_guidelines,
    load_textos,
    load_gold,
    repair_gold_offsets,
    sanity_check_gold_offsets,

    # reglas y modelo
    build_all_rules,
    load_ner_pipeline,

    # detección
    detect_all_texts,
    preds_to_dataframe,

    # anonimización
    anonymize_all_texts,

    # evaluación
    evaluate_strict,
    evaluate_lenient,

    # constantes
    OUTPUT_EXCEL,
)
OUTPUT_EXCEL_SYNTH = "Resultados_Anonimizador_Hibrido_v3_con_sinteticas.xlsx"


# =====================================================
# NOTAS CLÍNICAS SINTÉTICAS
# =====================================================

def load_synthetic_notes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": 1001,
                "Texto": (
                    "Roberto Sánchez, de 52 años, ingresó el 14/03/2023 en el Hospital Clínico San Carlos "
                    "por dolor torácico. Vive en C/ Alcalá 123, 28009 Madrid. "
                    "Teléfono de contacto: +34 612 345 678. DNI: 12345678Z. "
                    "Fue valorado por el Dr. Luis Martín. Fecha de nacimiento: 06/08/1970."
                )
            },
            {
                "ID": 1002,
                "Texto": (
                    "María López García, mujer de 34 años, acude a urgencias del Hospital Universitario La Paz "
                    "el día 02-11-2022. Dirección Av. de América 45, 28002 Madrid. "
                    "Email: maria.lopez@gmail.com. Teléfono 699 234 111. NHC HC-0098123."
                )
            },
            {
                "ID": 1003,
                "Texto": (
                    "Juan Pérez Martín, paciente de 67 años, ingresado el 01/01/2023 en el Hospital Vall d'Hebron. "
                    "Dirección Paseo de la Castellana 89, Madrid. Teléfono 91 555 22 11. DNI 87654321X."
                )
            },
            {
                "ID": 1004,
                "Texto": (
                    "Ana García Fernández, de 45 años, nacida el 12/12/1977, atendida en el Hospital Gregorio Marañón. "
                    "Email ana.garcia@outlook.es. Teléfono +34 622 998 441. HC-445566."
                )
            },
            {
                "ID": 1005,
                "Texto": (
                    "Luis Martín Gómez, varón de 29 años, atendido el 18/05/2021 en el Hospital de la Santa Creu i Sant Pau. "
                    "Av. Diagonal 456, Barcelona. DNI 33445566P."
                )
            },
            {
                "ID": 1006,
                "Texto": (
                    "Carmen Rodríguez López, de 73 años, domicilio Plaza Mayor 3, 47001 Valladolid. "
                    "Teléfono 600 987 654. DNI 99887766L."
                )
            },
            {
                "ID": 1007,
                "Texto": (
                    "José Manuel Ruiz Ortega, 58 años, ingresó el 23/06/2022 en el Hospital Virgen del Rocío. "
                    "Dirección C/ San Fernando 12, Sevilla. Email jose.ruiz@yahoo.es."
                )
            },
            {
                "ID": 1008,
                "Texto": (
                    "Laura Méndez, de 41 años, atendida en el Hospital Universitario de Canarias el 30/08/2021. "
                    "Av. Tres de Mayo 77, Tenerife. DNI 44556677R."
                )
            },
            {
                "ID": 1009,
                "Texto": (
                    "Francisco Gómez Sánchez, 64 años, historia clínica HC-778899. Hospital de Cruces. "
                    "Email fgomez@empresa.es. Teléfono 944 555 666."
                )
            },
            {
                "ID": 1010,
                "Texto": """
Roberto Sánchez, varón de 52 años, acude por su propio pie a Urgencias del Hospital Universitario La Paz
(Pº de la Castellana, 261, 28046 Madrid) el día 14/03/2023 a las 08:43. El paciente, con NHC: HC-0099123
y DNI 12345678Z, refiere dolor torácico de inicio brusco desde hace 2 horas, irradiado a brazo izquierdo.
Vive en C/ Alcalá 123, 3ºB, 28009 Madrid, aunque actualmente se encuentra empadronado en Valencia.
Teléfono de contacto: +34 612 345 678. Email: roberto.sanchez@gmail.com.
IP registrada en el sistema: 192.168.1.34.

Antecedentes personales: HTA, DM2 y dislipemia. Seguro médico privado ID_ASEGURAMIENTO: INS-2023-445-99.
Intervenido quirúrgicamente en 2018 en Clínica Universidad de Navarra (Pamplona).
Médico responsable: Dr. Juan Pérez Martín, colegiado en Madrid,
quien firma el informe junto con la Dra. María López García, cardióloga del Servicio de Cardiología.

Durante la anamnesis, el paciente indica que su padre, José Sánchez López, falleció en 2010 por IAM.
La madre, Ana Gómez Ruiz, vive actualmente en Av. de América 45, Madrid.
Contacto alternativo: ana.gomez@correo.es o teléfono 91 234 56 78.
Fecha de nacimiento del paciente: 22/07/1971.
Fecha de ingreso hospitalario: 14-03-2023.
Fecha prevista de alta: 18/03/2023.

Exploración física sin hallazgos relevantes salvo TA 150/95 mmHg y FC 98 lpm.
ECG compatible con SCASEST. Se decide ingreso en Unidad Coronaria.
El informe se genera desde el dispositivo DEV-XK-9933 conectado a la red interna (IP 10.0.0.12).
El paciente firma consentimiento informado con pasaporte ES1234567.

Observaciones adicionales: el Sr. Sánchez acudió previamente a consulta externa el 01/02/23
en el Hospital Clínico San Carlos. Se recomienda seguimiento ambulatorio.
Texto redactado y validado por el Dr. Pérez el 14/03/2023 a las 10:15.
"""
            },
        ]
    )


# =====================================================
# MAIN
# =====================================================

def main():
    print("Cargando datos base...")
    df_guidelines = load_guidelines()
    df_textos = load_textos()
    df_gold = load_gold()

    print("Añadiendo notas sintéticas...")
    df_synth = load_synthetic_notes()
    df_textos = pd.concat([df_textos, df_synth], ignore_index=True)

    # Reparar offsets GOLD (solo reales)
    sanity_check_gold_offsets(df_textos, df_gold)
    df_gold_fixed, _ = repair_gold_offsets(df_textos, df_gold)

    print("Construyendo reglas regex (Guidelines + EXTRA_RULES)...")
    rules = build_all_rules(df_guidelines)

    print("Cargando modelo NER...")
    ner_pipe = load_ner_pipeline()

    print("Detectando entidades (NER + regex)...")
    all_preds = detect_all_texts(df_textos, rules, ner_pipe, merge_address=True)
    df_preds = preds_to_dataframe(all_preds)
    # =====================================================
    # FILTRAR PREDICCIONES SOLO A TEXTOS REALES (PARA MÉTRICAS)
    # =====================================================
    real_ids = set(df_gold_fixed["ID"].unique())
    df_preds_real = df_preds[df_preds["ID"].isin(real_ids)].copy()


    print("Anonimizando textos...")
    df_anon = anonymize_all_texts(df_textos, all_preds, rules)

    print("Evaluando (solo textos reales)...")
    matching_strict, metrics_strict = evaluate_strict(df_gold_fixed, df_preds_real)
    matching_lenient, metrics_lenient = evaluate_lenient(df_gold_fixed, df_preds_real)


    print(f"Guardando resultados en {OUTPUT_EXCEL_SYNTH}")
    with pd.ExcelWriter(OUTPUT_EXCEL_SYNTH, engine="openpyxl") as writer:
        df_textos.to_excel(writer, sheet_name="Datos_Originales", index=False)
        df_preds.to_excel(writer, sheet_name="Predicciones_TODAS", index=False)
        df_preds_real.to_excel(writer, sheet_name="Predicciones_REALES", index=False)
        df_anon.to_excel(writer, sheet_name="Textos_Anonimizados", index=False)
        matching_strict.to_excel(writer, sheet_name="Validacion_Strict", index=False)
        matching_lenient.to_excel(writer, sheet_name="Validacion_Lenient", index=False)

    print("✔️ Listo. Abre Streamlit con:")
    print("   streamlit run visor_notas.py")


if __name__ == "__main__":
    main()
