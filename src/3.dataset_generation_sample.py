# --- Rutas robustas y carga unificada ---
from pathlib import Path
import pandas as pd
import sys

# Carpeta del script (independiente del cwd con el que lo ejecutes)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # Proyecto_PasswordStrength/
DATA = ROOT / "data"

# Define rutas esperadas (ajústalas si tus nombres difieren)
rockyou_path   = DATA / "processed" / "FullDataset" / "rockyou_patterns_1M.csv"
synthetic_pattern_path = DATA / "processed" / "FullDataset" / "synthetic_patterns_all.csv"
synthetic_path = DATA / "processed" / "FullDataset" / "synthetic_patterns_1M.csv"

# Debug: imprime contexto y existencia de archivos
print(f"[INFO] cwd        : {Path.cwd()}")
print(f"[INFO] script dir : {HERE}")
print(f"[INFO] ROOT       : {ROOT}")
for p in [rockyou_path, synthetic_pattern_path, synthetic_path]:
    print(f"[CHECK] {p} -> {'OK' if p.exists() else 'NO ENCONTRADO'}")

# Si falta alguno, muestra contenido de la carpeta para ubicar errores de nombre
missing = [p for p in [rockyou_path, synthetic_pattern_path, synthetic_path] if not p.exists()]
if missing:
    print("\n[ERROR] No se encontraron estos archivos:")
    for m in missing:
        print(" -", m)
    # Lista alternativas en los directorios clave para ayudarte a corregir el nombre
    for folder in {*(m.parent for m in missing)}:
        if folder.exists():
            print(f"\n[LISTADO] Archivos en {folder}:")
            for f in sorted(folder.glob("*")):
                print("  •", f.name)
        else:
            print(f"\n[AVISO] La carpeta no existe: {folder}")
    sys.exit(1)  # detenemos aquí para que ajustes los nombres/rutas

# Carga segura
df_rockyou   = pd.read_csv(rockyou_path)
df_weak      = pd.read_csv(synthetic_pattern_path)
df_synthetic = pd.read_csv(synthetic_path)

# Estandariza columnas y une (con columna de origen)
common_cols = [
    "password","length","has_seq_alpha","has_seq_num",
    "has_repeat","has_keyboard","has_year","has_common"
]

def ensure_cols(df):
    for c in common_cols:
        if c not in df.columns:
            # elige None o False según tu pipeline
            df[c] = None
    return df[common_cols]

df_rockyou["source"]   = "rockyou"
df_weak["source"]      = "weak"
df_synthetic["source"] = "synthetic"

df_final = pd.concat(
    [ensure_cols(df_rockyou), ensure_cols(df_weak), ensure_cols(df_synthetic)],
    ignore_index=True
)

# Dedupe por contraseña (opcional)
df_final.drop_duplicates(subset=["password"], inplace=True)

# Guarda
out_path = DATA / "processed" / "passwords_final.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(out_path, index=False)
print(f"[OK] Dataset final creado: {df_final.shape} -> {out_path}")
