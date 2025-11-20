import argparse
from pathlib import Path


from zxcvbn import zxcvbn


def score_to_percent(score: int) -> float:
    """
    Convierte el score de zxcvbn (0-4) a un porcentaje 0-100%.
    Aquí usamos una conversión lineal: (score / 4) * 100.
    """
    return round((score / 4) * 100, 2)


def analizar_archivo(input_path: Path, output_csv: Path | None = None):
    resultados = []

    with input_path.open("r", encoding="utf-8") as f:
        for linea in f:
            pw = linea.rstrip("\n\r")
            if not pw:
                continue  # saltar líneas vacías

            res = zxcvbn(pw)
            score = res["score"]          # 0 a 4
            porcentaje = score_to_percent(score)

            resultados.append({
                "password": pw,
                "score_zxcvbn": score,
                "porcentaje_seguridad": porcentaje,
                "crack_time_offline_fast_hash": res["crack_times_display"]["offline_fast_hashing_1e10_per_second"],
                "crack_time_online_slow": res["crack_times_display"]["online_throttling_100_per_hour"],
            })

    # Imprimir en pantalla
    print(f"{'Password':30s} | {'Score':5s} | {'% seguro':9s} | Crack time (offline 1e10/s)")
    print("-" * 80)
    for r in resultados:
        print(f"{r['password'][:30]:30s} | "
              f"{r['score_zxcvbn']:^5d} | "
              f"{r['porcentaje_seguridad']:>8.2f}% | "
              f"{r['crack_time_offline_fast_hash']}")

    # Guardar en CSV si se pidió
    if output_csv is not None:
        import csv
        with output_csv.open("w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=resultados[0].keys())
            writer.writeheader()
            writer.writerows(resultados)
        print(f"\nResultados guardados en: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Analiza un TXT de contraseñas con zxcvbn y muestra su % de seguridad."
    )
    parser.add_argument(
        "input_txt",
        type=str,
        help="Ruta al archivo .txt con contraseñas (una por línea)."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Ruta opcional para guardar los resultados en CSV."
    )

    args = parser.parse_args()
    input_path = Path(args.input_txt)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {input_path}")

    output_csv = Path(args.output_csv) if args.output_csv else None
    analizar_archivo(input_path, output_csv)


if __name__ == "__main__":
    main()
