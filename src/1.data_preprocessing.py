from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

# Rutas típicas si no se pasa --input
DEFAULT_INPUTS = [
    Path("data/raw/rockyou.txt.gz"),
    Path("data/raw/rockyou.txt"),
    Path("/usr/share/wordlists/rockyou.txt.gz"),
    Path("/usr/share/wordlists/rockyou.txt"),
]

CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]")  # ASCII control chars + DEL

def find_input_path(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"[ERROR] No existe el archivo de entrada: {p}")
        return p
    for cand in DEFAULT_INPUTS:
        if cand.exists():
            return cand
    sys.exit("[ERROR] No se encontró archivo de entrada. "
             "Pasa --input o coloca rockyou en data/raw/")

def iter_lines_text(path: Path, encodings=("latin-1", "utf-8")) -> Iterable[str]:
    """
    Itera líneas de un archivo .txt o .gz decodificando de forma segura.
    Intenta 'latin-1' (robusto para RockYou) y luego 'utf-8' si falla.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    last_err = None
    for enc in encodings:
        try:
            with opener(path, "rb") as fb:
                with io.TextIOWrapper(fb, encoding=enc, errors="strict", newline="") as f:
                    for line in f:
                        yield line.rstrip("\n\r")
            return
        except Exception as e:
            last_err = e
            continue
    sys.exit(f"[ERROR] No se pudo leer {path} con {encodings}. Último error: {last_err}")

def clean_password(pw_raw: str) -> str:
    """
    - Normaliza Unicode a NFC
    - Elimina caracteres de control/no imprimibles
    - Trim espacios extremos
    """
    # Normaliza
    pw = unicodedata.normalize("NFC", pw_raw)
    # Quita ASCII control + DEL
    pw = CONTROL_RE.sub("", pw)
    # Filtra cualquier categoría Unicode de control/format
    cleaned = []
    for ch in pw:
        # categorías Unicode: Cc (control), Cf (format), Cs, Co, Cn -> descartar
        if unicodedata.category(ch).startswith("C"):
            continue
        cleaned.append(ch)
    pw = "".join(cleaned).strip()
    return pw

def preprocess(
    input_path: Path,
    output_path: Path,
    min_len: int,
    max_len: int,
    sample: Optional[int] = None,
    no_dedup: bool = False,
) -> None:
    """
    Procesa por streaming y escribe CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    read_count = 0
    drop_control = 0
    drop_length = 0
    drop_dupe = 0
    written = 0

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["password"])

        for raw in iter_lines_text(input_path):
            read_count += 1

            pw = clean_password(raw)
            if not pw:
                drop_control += 1
                continue

            n = len(pw)
            if not (min_len <= n <= max_len):
                drop_length += 1
                continue

            if not no_dedup:
                h = hashlib.sha1(pw.encode("utf-8", errors="ignore")).hexdigest()
                if h in seen_hashes:
                    drop_dupe += 1
                    continue
                seen_hashes.add(h)

            writer.writerow([pw])
            written += 1

            if sample is not None and written >= sample:
                break

    # Resumen
    print("=== Resumen de preprocesamiento ===")
    print(f"Entrada                : {input_path}")
    print(f"Salida CSV             : {output_path}")
    print(f"Líneas leídas          : {read_count:,}")
    print(f"Descartadas (control)  : {drop_control:,}")
    print(f"Descartadas (longitud) : {drop_length:,}")
    if not no_dedup:
        print(f"Descartadas (duplicado): {drop_dupe:,}")
    print(f"Escritas               : {written:,}")

def main():
    ap = argparse.ArgumentParser(description="Preprocesamiento mínimo de listas de contraseñas.")
    ap.add_argument("--input", type=str, default=None,
                    help="Ruta del archivo de entrada (.txt o .gz). Si no se indica, se intenta data/raw/rockyou.*")
    ap.add_argument("--output", type=str, default="data/processed/rockyou_clean.csv",
                    help="Ruta del CSV de salida (por defecto: data/processed/rockyou_clean.csv)")
    ap.add_argument("--min-len", type=int, default=4, help="Longitud mínima (incluida).")
    ap.add_argument("--max-len", type=int, default=30, help="Longitud máxima (incluida).")
    ap.add_argument("--sample", type=int, default=None,
                    help="Si se indica, detiene tras escribir N contraseñas válidas (pruebas rápidas).")
    ap.add_argument("--no-dedup", action="store_true",
                    help="Desactiva deduplicación (más veloz, pero puede repetir).")
    args = ap.parse_args()

    input_path = find_input_path(args.input)
    output_path = Path(args.output)

    print("=== Preprocesamiento de contraseñas ===")
    print(f"Archivo de entrada  : {input_path}")
    print(f"Archivo de salida   : {output_path}")
    print(f"Longitud admitida   : [{args.min_len}, {args.max_len}]")
    print(f"Deduplicación       : {'NO' if args.no_dedup else 'SÍ'}")
    if args.sample:
        print(f"Muestreo            : primeras {args.sample} válidas")
    print("Procesando...\n")

    preprocess(
        input_path=input_path,
        output_path=output_path,
        min_len=args.min_len,
        max_len=args.max_len,
        sample=args.sample,
        no_dedup=args.no_dedup,
    )

if __name__ == "__main__":
    main()