import argparse
import csv
import secrets
import string
from pathlib import Path

def generate_password(min_len: int, max_len: int, alphabet: str) -> str:
    """Genera una contraseña aleatoria dentro del rango de longitud dado."""
    length = secrets.randbelow(max_len - min_len + 1) + min_len
    return "".join(secrets.choice(alphabet) for _ in range(length))

def main():
    parser = argparse.ArgumentParser(description="Generador de contraseñas sintéticas de alta entropía.")
    parser.add_argument("--output", type=str, required=True,
                        help="Ruta al archivo CSV de salida.")
    parser.add_argument("--n", type=int, default=1_000_000,
                        help="Número de contraseñas a generar. (default=1M)")
    parser.add_argument("--min-len", type=int, default=12,
                        help="Longitud mínima (default=12).")
    parser.add_argument("--max-len", type=int, default=20,
                        help="Longitud máxima (default=20).")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Construcción del alfabeto: minúsculas, mayúsculas, dígitos, símbolos ASCII imprimibles
    alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation

    # Generar y escribir en CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["password"])
        for i in range(args.n):
            pw = generate_password(args.min_len, args.max_len, alphabet)
            writer.writerow([pw])
            if (i+1) % 100000 == 0:
                print(f"Generadas {i+1:,} contraseñas...")

    print(f"\n Generadas {args.n:,} contraseñas en {out_path}")

if __name__ == "__main__":
    main()
