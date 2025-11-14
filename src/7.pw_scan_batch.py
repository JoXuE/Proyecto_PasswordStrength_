import argparse
import json, math, re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency
import string

# ====== Extractor de características ======
def count_digits(pw): return sum(c.isdigit() for c in str(pw))
def count_upper(pw): return sum(c.isupper() for c in str(pw))
def count_lower(pw): return sum(c.islower() for c in str(pw))
def count_symbols(pw): return sum(c in string.punctuation for c in str(pw))
def charset_size(pw):
    s = str(pw)
    has_lower   = any(c.islower() for c in s)
    has_upper   = any(c.isupper() for c in s)
    has_digits  = any(c.isdigit() for c in s)
    has_symbols = any(c in string.punctuation for c in s)
    size = 0
    if has_lower:   size += 26
    if has_upper:   size += 26
    if has_digits:  size += 10
    if has_symbols: size += len(string.punctuation)
    return size

def max_same_char_run(s):
    s = str(s)
    if not s: return 0
    mx = cur = 1
    prev = s[0]
    for c in s[1:]:
        if c == prev:
            cur += 1
            if cur > mx: mx = cur
        else:
            cur = 1
            prev = c
    return mx

def has_long_repeated_substring(s, minlen=3):
    s = str(s)
    for L in range(minlen, min(6, len(s)//2 + 1)):
        for i in range(len(s) - 2*L + 1):
            chunk = s[i:i+L]
            if chunk and s.count(chunk*2) > 0:
                return True
    return False

def is_strict_email(s: str) -> bool:
    s = str(s)
    if s.count("@") != 1: return False
    local, domain = s.split("@", 1)
    if not (1 <= len(local) <= 64): return False
    if not (4 <= len(domain) <= 253): return False
    if len(s) > 254: return False
    if "." not in domain: return False
    if domain != domain.lower(): return False
    if not re.fullmatch(r"[a-z0-9.-]+", domain): return False
    labels = domain.split(".")
    if len(labels) < 2: return False
    tld = labels[-1]
    if not re.fullmatch(r"[a-z]{2,24}", tld): return False
    label_re = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)")
    for lab in labels:
        if not label_re.fullmatch(lab): return False
    if local[0] == "." or local[-1] == "." or ".." in local: return False
    if not re.fullmatch(r"[A-Za-z0-9._%+-]+", local): return False
    sld = labels[-2]
    if not re.search(r"[a-z]", sld): return False
    return True

YEAR4_RE = re.compile(r"(?<!\d)(19[3-9]\d|20[0-2]\d|2025)(?!\d)")
def contains_year4(s): return bool(YEAR4_RE.search(str(s)))
def contains_word_plus_year4(s): return bool(re.search(r"[A-Za-z]{3,}(19[3-9]\d|20[0-2]\d|2025)", str(s)))
def contains_word_plus_year2(s): return bool(re.search(r"[A-Za-z]{3,}(\d{2})(?!\d)", str(s)))

def _yy_to_year(yy):
    yy = int(str(yy))
    if 0 <= yy <= 25: return 2000 + yy
    return 1900 + yy

def _has_date_tokens(s):
    s = str(s)
    has8 = has6 = date_at_end = False
    runs = re.findall(r"\d{6,8}", s)
    for run in runs:
        n = len(run)
        if n >= 8:
            for i in range(n - 8 + 1):
                chunk = run[i:i+8]
                mm, dd, yyyy = int(chunk[0:2]), int(chunk[2:4]), int(chunk[4:8])
                if 1 <= mm <= 12 and 1900 <= yyyy <= 2025:
                    has8 = True
                    if s.endswith(chunk): date_at_end = True
        if n >= 6:
            for i in range(n - 6 + 1):
                chunk = run[i:i+6]
                if 1 <= int(chunk[0:2]) <= 12:
                    year = _yy_to_year(chunk[4:6])
                    if 1900 <= year <= 2025:
                        has6 = True
                        if s.endswith(chunk): date_at_end = True
    return has8, has6, date_at_end

INC_SEQ4 = set([
    '0123','1234','2345','3456','4567','5678','6789',
    'abcd','bcde','cdef','defg','efgh','fghi','ghij','hijk',
    'ijkl','jklm','klmn','lmno','mnop','nopq','opqr','pqrs',
    'qrst','rstu','stuv','tuvw','uvwx','vwxy','wxyz'
])

def has_common_pattern(s: str, threshold=3.5) -> int:
    # solo wordfreq como pediste
    s = re.sub(r"[^a-záéíóúüñ]", " ", s.lower())
    tokens = [t for t in s.split() if len(t) >= 3]
    for token in tokens:
        if zipf_frequency(token, "en") >= threshold or zipf_frequency(token, "es") >= threshold:
            return 1
    return 0

def extract_features_for_model(pw: str):
    s = str(pw)
    num_lower   = count_lower(s)
    num_upper   = count_upper(s)
    num_digits  = count_digits(s)
    num_symbols = count_symbols(s)
    cs = charset_size(s)
    has8, has6, date_at_end = _has_date_tokens(s)
    feat = {
        'charset_size'       : cs,
        'has_year'           : int(contains_year4(s) or contains_word_plus_year4(s) or contains_word_plus_year2(s)),
        'length'             : len(s),
        'num_symbols'        : num_symbols,
        'has_common'         : int(has_common_pattern(s)),
        'num_upper'          : num_upper,
        '_ends_digit_run6'   : int(bool(re.search(r'\d{6,}$', s))),
        '_starts_digit_run6' : int(bool(re.match(r'^\d{6,}', s))),
        '_has_year4'         : int(contains_year4(s)),
        'has_repeat'         : int(max_same_char_run(s) >= 2),
        'num_digits'         : num_digits,
        '_max_same_char_run' : max_same_char_run(s),
        '_date_at_end'       : int(date_at_end),
        '_has_date6'         : int(has6),
        'has_seq_num'        : int(bool(re.search(r'012|123|234|345|456|567|678|789|987|876|765|654|543|432|321|210', s))),
        'num_lower'          : num_lower,
        '_has_date8'         : int(has8),
        '_has_word_year2'    : int(bool(re.search(r'[A-Za-z]{3,}(\d{2})(?!\d)', s))),
        '_ends_inc_seq4'     : int(str(s[-4:]).lower() in INC_SEQ4),
        '_has_long_repeats'  : int(has_long_repeated_substring(s, minlen=3)),
        '_is_email_strict'   : int(is_strict_email(s)),
    }
    return feat

# ====== Entropía y tiempos ======
def shannon_entropy(s: str) -> float:
    s = str(s)
    if not s: return 0.0
    cnt = Counter(s)
    probs = np.array([v/len(s) for v in cnt.values()])
    return float(-np.sum(probs * np.log2(probs)))

def combinatorial_entropy_estimate(s: str) -> float:
    cs = charset_size(s)
    if cs <= 0: return 0.0
    # upper bound si todo es equiprobable
    return float(math.log2(cs ** len(s)))

def guesses_from_entropy(entropy_bits: float) -> float:
    return 2 ** entropy_bits

def estimate_crack_time_seconds(entropy_bits: float, guesses_per_second: float) -> float:
    return guesses_from_entropy(entropy_bits) / guesses_per_second

def human_time(seconds: float) -> str:
    if seconds < 1: return f"{seconds:.3f} s"
    mins = seconds/60
    if mins < 1: return f"{seconds:.1f} s"
    hours = mins/60
    if hours < 1: return f"{mins:.1f} min"
    days = hours/24
    if days < 1: return f"{hours:.2f} h"
    years = days/365.25
    if years < 1: return f"{days:.2f} d"
    if years < 1000: return f"{years:.2f} years"
    return f"{years:.2e} years"

# ====== Utilidades de nombres de features ======
def load_feature_order_json(model_dir: Path, pattern="*.featurenames.json"):
    j = list(model_dir.glob(pattern))
    if j:
        return json.loads(j[0].read_text(encoding="utf-8"))
    return None

# ====== Carga de modelos ======
def load_lightgbm(model_dir: Path):
    """
    Admite:
      - .pkl/.joblib sklearn wrapper con predict_proba
      - .txt Booster nativo (LightGBM)
    Devuelve (obj, tipo, feature_names_or_None)
    """
    # sklearn joblib/pkl
    try:
        import joblib
        files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))
        if files:
            mdl = joblib.load(str(files[0]))
            feat = getattr(mdl, "feature_names_in_", None)
            return mdl, "sklearn", list(feat) if feat is not None else None
    except Exception:
        pass
    # Booster .txt
    try:
        import lightgbm as lgb
        txts = list(model_dir.glob("*.txt"))
        if txts:
            bst = lgb.Booster(model_file=str(txts[0]))
            return bst, "booster", bst.feature_name()
    except Exception:
        pass
    return None, None, None

def load_xgboost(model_dir: Path):
    """
    Requiere archivos guardados con save_model: .ubj o .json (no dump_model).
    Devuelve (booster_or_None, feature_names_or_None)
    """
    try:
        import xgboost as xgb
        for fname in ["xgboost_gpu_all.ubj", "xgboost_gpu_all.json"]:
            p = model_dir / fname
            if p.exists():
                bst = xgb.Booster()
                bst.load_model(str(p))
                # nombres vendrán del DMatrix de entrada
                return bst, None
        # fallback: cualquier .ubj/.json/.xgb válido
        for f in list(model_dir.glob("*.ubj")) + list(model_dir.glob("*.json")) + list(model_dir.glob("*.xgb")):
            try:
                bst = xgboost.Booster()
                bst.load_model(str(f))
                return bst, None
            except Exception:
                continue
    except Exception:
        pass
    return None, None

def load_logistic(model_dir: Path):
    """
    sklearn LogisticRegression (o Pipeline) en .pkl/.joblib
    Devuelve (modelo, feature_names_or_None)
    """
    try:
        import joblib
        files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))
        if not files:
            return None, None
        mdl = joblib.load(str(files[0]))
        feat = getattr(mdl, "feature_names_in_", None)
        return mdl, list(feat) if feat is not None else None
    except Exception:
        return None, None

# ====== Predicción por modelo ======
def vec_from_order(feat_dict: dict, order: list[str] | None):
    if order:
        return np.array([[float(feat_dict.get(k, 0.0)) for k in order]], dtype=float)
    # fallback: orden alfabético para consistencia
    keys = sorted(feat_dict.keys())
    return np.array([[float(feat_dict[k]) for k in keys]], dtype=float), keys

def predict_lightgbm(obj, kind, x_row_vec, feat_order):
    """
    kind: 'sklearn' o 'booster'
    Retorna probabilidad (float)
    """
    if kind == "booster":
        import lightgbm as lgb
        # Booster espera lgb.Dataset? No, .predict acepta numpy 2D
        return float(obj.predict(x_row_vec)[0])
    else:
        # sklearn wrapper
        if hasattr(obj, "predict_proba"):
            return float(obj.predict_proba(x_row_vec)[0, 1])
        return float(obj.predict(x_row_vec)[0])

def predict_xgboost(booster, feat_order, feat_df_row):
    import xgboost as xgb
    # Creamos DataFrame con columnas=feat_order para alinear
    if feat_order is None:
        # usar todas las claves del dict ordenadas alfabéticamente
        cols = sorted(list(feat_df_row.keys()))
    else:
        cols = list(feat_order)
    X = pd.DataFrame([[feat_df_row.get(c, 0.0) for c in cols]], columns=cols)
    dmat = xgb.DMatrix(X, feature_names=cols)
    return float(booster.predict(dmat)[0])

def predict_logistic(mdl, x_row_vec):
    # sklearn LogisticRegression / Pipeline
    if hasattr(mdl, "predict_proba"):
        return float(mdl.predict_proba(x_row_vec)[0, 1])
    if hasattr(mdl, "decision_function"):
        z = float(mdl.decision_function(x_row_vec)[0])
        return 1.0 / (1.0 + math.exp(-z))
    return float(mdl.predict(x_row_vec)[0])

# ====== Proceso de evaluación una contraseña ======
def evaluate_one(pw: str,
                 lgbm_pack, xgb_pack, logi_pack,
                 lgbm_featnames_json, xgb_featnames_json, logi_featnames_json,
                 gps_custom: float):
    feat = extract_features_for_model(pw)

    # Entropías y crack time
    sh_bits_char = shannon_entropy(pw)
    total_bits = sh_bits_char * len(pw)
    comb_bits = combinatorial_entropy_estimate(pw)
    scenarios = {
        "online_throttled": 100.0,
        "offline_cpu"     : 1e4,
        "offline_gpu"     : 1e8,
        "custom"          : gps_custom,
    }
    times = {}
    for name, gps in scenarios.items():
        times[name] = {
            "time_comb":   human_time(estimate_crack_time_seconds(comb_bits, gps)),
            "time_shann":  human_time(estimate_crack_time_seconds(total_bits, gps)),
        }

    row = {
        "password": pw,
        "masked": (pw[:2] + "*" * max(0, len(pw) - 2)) if len(pw) > 2 else pw,
        "length": len(pw),
        "charset_size": feat["charset_size"],
        "shannon_bits_per_char": sh_bits_char,
        "shannon_total_bits": total_bits,
        "combinatorial_bits": comb_bits,
        "crack_online_throttled_comb": times["online_throttled"]["time_comb"],
        "crack_online_throttled_shan": times["online_throttled"]["time_shann"],
        "crack_offline_cpu_comb": times["offline_cpu"]["time_comb"],
        "crack_offline_cpu_shan": times["offline_cpu"]["time_shann"],
        "crack_offline_gpu_comb": times["offline_gpu"]["time_comb"],
        "crack_offline_gpu_shan": times["offline_gpu"]["time_shann"],
        "crack_custom_comb": times["custom"]["time_comb"],
        "crack_custom_shan": times["custom"]["time_shann"],
    }

    # LightGBM
    lgbm_obj, lgbm_kind, lgbm_feat_in = lgbm_pack
    lgbm_order = lgbm_featnames_json or lgbm_feat_in
    if lgbm_obj is not None:
        vec = np.array([[float(feat.get(k, 0.0)) for k in lgbm_order]]) if lgbm_order else np.array([[float(v) for v in feat.values()]])
        try:
            row["prob_secure_lightgbm"] = predict_lightgbm(lgbm_obj, lgbm_kind, vec, lgbm_order)
        except Exception as e:
            row["prob_secure_lightgbm"] = None
            row["error_lightgbm"] = str(e)
    else:
        row["prob_secure_lightgbm"] = None

    # XGBoost
    xgb_obj, _ = xgb_pack
    xgb_order = xgb_featnames_json  # XGB Booster no guarda nombres; DMatrix los define
    if xgb_obj is not None:
        try:
            row["prob_secure_xgboost"] = predict_xgboost(xgb_obj, xgb_order, feat)
        except Exception as e:
            row["prob_secure_xgboost"] = None
            row["error_xgboost"] = str(e)
    else:
        row["prob_secure_xgboost"] = None

    # Logistic
    logi_obj, logi_feat_in = logi_pack
    logi_order = logi_featnames_json or logi_feat_in
    if logi_obj is not None:
        vec = np.array([[float(feat.get(k, 0.0)) for k in logi_order]]) if logi_order else np.array([[float(v) for v in feat.values()]])
        try:
            row["prob_secure_logistic_l2"] = predict_logistic(logi_obj, vec)
        except Exception as e:
            row["prob_secure_logistic_l2"] = None
            row["error_logistic_l2"] = str(e)
    else:
        row["prob_secure_logistic_l2"] = None

    # Agrega las features crudas (útil para interpretabilidad tabular)
    for k, v in feat.items():
        row[f"feat__{k}"] = v

    return row

# ====== Main batch ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-txt", required=True, help="Ruta del TXT con una contraseña por línea.")
    ap.add_argument("--out-csv", default="resultados.csv", help="Archivo CSV de salida.")
    ap.add_argument("--out-jsonl", default="resultados.jsonl", help="Archivo JSONL de salida.")
    ap.add_argument("--guessec", type=float, default=1e6, help="Guesses/seg para escenario 'custom'.")
    # Rutas modelos
    ap.add_argument("--lgbm-dir", default=r"D:\Universidad\ULTIMO_SEMESTRE\Tesis\Proyecto_PasswordStrength_\models\lightgbm\all")
    ap.add_argument("--xgb-dir",  default=r"D:\Universidad\ULTIMO_SEMESTRE\Tesis\Proyecto_PasswordStrength_\models\xgboost\all")
    ap.add_argument("--logi-dir", default=r"D:\Universidad\ULTIMO_SEMESTRE\Tesis\Proyecto_PasswordStrength_\models\sklearn\logistic_regression\L2")
    args = ap.parse_args()

    input_path = Path(args.input_txt)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    # Cargar modelos
    lgbm_dir = Path(args.lgbm_dir)
    xgb_dir  = Path(args.xgb_dir)
    logi_dir = Path(args.logi_dir)

    lgbm_obj, lgbm_kind, lgbm_feat_in = load_lightgbm(lgbm_dir)
    xgb_obj,  _                      = load_xgboost(xgb_dir)
    logi_obj, logi_feat_in           = load_logistic(logi_dir)

    # Feature names JSON (si los tienes guardados)
    lgbm_featnames_json = load_feature_order_json(lgbm_dir)
    xgb_featnames_json  = load_feature_order_json(xgb_dir)
    logi_featnames_json = load_feature_order_json(logi_dir)

    lgbm_pack = (lgbm_obj, lgbm_kind, lgbm_feat_in)
    xgb_pack  = (xgb_obj, None)
    logi_pack = (logi_obj, logi_feat_in)

    # Leer contraseñas
    pw_list = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            pw = line.rstrip("\n\r")
            if pw.strip() == "":
                continue
            pw_list.append(pw)

    rows = []
    for pw in pw_list:
        row = evaluate_one(
            pw=pw,
            lgbm_pack=lgbm_pack,
            xgb_pack=xgb_pack,
            logi_pack=logi_pack,
            lgbm_featnames_json=lgbm_featnames_json,
            xgb_featnames_json=xgb_featnames_json,
            logi_featnames_json=logi_featnames_json,
            gps_custom=args.guessec
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    with open(args.out_jsonl, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Procesadas {len(rows)} contraseñas.")
    print(f"[OK] CSV  -> {args.out_csv}")
    print(f"[OK] JSONL-> {args.out_jsonl}")

if __name__ == "__main__":
    main()
