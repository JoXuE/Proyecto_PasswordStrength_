import argparse
import getpass
import json, math, re, sys
from pathlib import Path
from collections import Counter
import numpy as np
from wordfreq import zipf_frequency



import string
def count_digits(pw): return sum(c.isdigit() for c in str(pw))
def count_upper(pw): return sum(c.isupper() for c in str(pw))
def count_lower(pw): return sum(c.islower() for c in str(pw))
def count_symbols(pw): return sum(c in string.punctuation for c in str(pw))
def charset_size(pw):
    has_lower   = any(c.islower() for c in str(pw))
    has_upper   = any(c.isupper() for c in str(pw))
    has_digits  = any(c.isdigit() for c in str(pw))
    has_symbols = any(c in string.punctuation for c in str(pw))
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
    if s.count("@") != 1:
        return False
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
        if not label_re.fullmatch(lab):
            return False
    if local[0] == "." or local[-1] == "." or ".." in local:
        return False
    if not re.fullmatch(r"[A-Za-z0-9._%+-]+", local):
        return False
    sld = labels[-2]
    if not re.search(r"[a-z]", sld):
        return False
    return True

# year/date heuristics
YEAR4_RE = re.compile(r"(?<!\d)(19[3-9]\d|20[0-2]\d|2025)(?!\d)")
def contains_year4(s): return bool(YEAR4_RE.search(str(s)))
def contains_word_plus_year4(s): return bool(re.search(r"[A-Za-z]{3,}(19[3-9]\d|20[0-2]\d|2025)", str(s)))
def contains_word_plus_year2(s): return bool(re.search(r"[A-Za-z]{3,}(\d{2})(?!\d)", str(s)))

def _yy_to_year(yy):
    yy = int(str(yy))
    if yy >= 0 and yy <= 25:
        return 2000 + yy
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

INC_SEQ4 = set(['0123','1234','2345','3456','4567','5678','6789',
                'abcd','bcde','cdef','defg','efgh','fghi','ghij','hijk',
                'ijkl','jklm','klmn','lmno','mnop','nopq','opqr','pqrs',
                'qrst','rstu','stuv','tuvw','uvwx','vwxy','wxyz'])


def has_common_pattern(s: str, threshold=3.5) -> int:
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

# Entropía y tiempos
def shannon_entropy(s: str) -> float:
    s = str(s)
    if not s: return 0.0
    cnt = Counter(s)
    probs = np.array([v/len(s) for v in cnt.values()])
    return float(-np.sum(probs * np.log2(probs)))

def combinatorial_entropy_estimate(s: str) -> float:
    cs = charset_size(s)
    if cs <= 0: return 0.0
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

# carga de modelo / features
def load_feature_order(model_dir: Path):
    j = model_dir.glob("*.featurenames.json")
    try:
        jf = next(j)
        return json.loads(jf.read_text(encoding="utf-8"))
    except StopIteration:
        try:
            import lightgbm as lgb
            txt = next(model_dir.glob("*.txt"))
            bst = lgb.Booster(model_file=str(txt))
            return bst.feature_name()
        except Exception:
            return None

def load_model(model_dir: Path):
    try:
        import joblib
        pkl = next(model_dir.glob("*.pkl"))
        mdl = joblib.load(str(pkl))
        return mdl, "joblib"
    except Exception:
        try:
            import lightgbm as lgb
            txt = next(model_dir.glob("*.txt"))
            bst = lgb.Booster(model_file=str(txt))
            return bst, "lightgbm"
        except Exception as e:
            return None, None

def evaluate_password(pw: str, model=None, model_type=None, feature_order=None, guesses_per_second=1e6):
    feat = extract_features_for_model(pw)
    if feature_order:
        vec = np.array([float(feat.get(k, 0.0)) for k in feature_order]).reshape(1, -1)
    else:
        vec = np.array([float(v) for v in feat.values()]).reshape(1, -1)

    result = {"password_masked": (pw[:2] + "*"*(max(0,len(pw)-2))) if len(pw)>2 else pw, "features": feat}
    base_comb = combinatorial_entropy_estimate(pw)
    base_shan = shannon_entropy(pw) * len(pw)
    scenarios = {"online_throttled":100.0,"offline_cpu":1e4,"offline_gpu":1e8,"custom":guesses_per_second}
    ct = {}
    for name,gps in scenarios.items():
        secs_comb = estimate_crack_time_seconds(base_comb, gps)
        secs_shan = estimate_crack_time_seconds(base_shan, gps)
        ct[name] = {"gps":gps, "time_comb":human_time(secs_comb), "time_shan":human_time(secs_shan)}
    result["crack_time"] = ct

    if model is not None:
        try:
            if model_type == "lightgbm":
                preds = model.predict(vec)
                result["model_prob_secure"] = float(preds[0])
            else:
                if hasattr(model, "predict_proba"):
                    result["model_prob_secure"] = float(model.predict_proba(vec)[0,1])
                else:
                    result["model_pred"] = float(model.predict(vec)[0])
        except Exception as e:
            result["model_error"] = str(e)

    recs = []
    L = len(pw)
    if L < 8:
        recs.append("Aumenta la longitud a 12+ caracteres.")
    if feat.get("charset_size",0) < 40:
        recs.append("Mezcla mayúsculas, minúsculas, números y símbolos.")
    if feat.get("_max_same_char_run",0) >= 3:
        recs.append("Evita repeticiones largas de un mismo carácter.")
    if feat.get("has_seq_num",0):
        recs.append("Evita secuencias numéricas consecutivas.")
    if feat.get("_is_email_strict",0):
        recs.append("No uses una dirección de correo como contraseña.")
    result["recommendations"] = recs
    return result

def pretty_print(res, pw, show_pw=False):
    print("\n--- Evaluación ---\n")
    if show_pw:
        print("Password:", pw)
    else:
        print("Password (masked):", res["password_masked"])
    feat = res["features"]
    print(f"Length: {feat.get('length')}  Charset_size: {feat.get('charset_size')}")
    print(f"Shannon bits per char: {round((shannon_entropy(pw) if pw else 0),4)}")
    print(f"Combinatorial bits (upper bound): {round(combinatorial_entropy_estimate(pw),2)}\n")
    print("Crack time estimates:")
    for k,v in res["crack_time"].items():
        print(f" - {k}: combinatorial -> {v['time_comb']}, shannon -> {v['time_shan']}")
    if "model_prob_secure" in res:
        print("\nModel probability secure:", res["model_prob_secure"])
    if "model_error" in res:
        print("\n[Model error]", res["model_error"])
    print("\nRecommendations:")
    for r in res["recommendations"]:
        print(" -", r)
    print("\nFeatures (raw):")
    for k,v in feat.items():
        print(f"  - {k}: {v}")
    print("\n--- End ---\n")

def main():
    # Detecta automáticamente la ruta absoluta del proyecto
    BASE_DIR = Path(__file__).resolve().parent.parent
    DEFAULT_MODEL_DIR = BASE_DIR / "models" / "lightgbm" / "all"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directorio del modelo exportado y featurenames.json"
    )
    parser.add_argument("--guessec", type=float, default=1e6, help="Guesses por segundo para escenario 'custom'")
    parser.add_argument("--shap", action="store_true", help="Intentar SHAP (si está instalado y es compatible)")
    parser.add_argument("--show-pw", action="store_true", help="Mostrar contraseña en claro en salida")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    feature_order = load_feature_order(model_dir)
    model, model_type = load_model(model_dir)


    if feature_order is None:
        print("[WARN] No se encontraron nombres de feature; usando extractor por defecto.")
    else:
        print("[INFO] Feature order cargado:", feature_order)

    if model is None:
        print("[WARN] No se cargó modelo. Continuando con heurísticas.")
    else:
        print(f"[INFO] Modelo cargado (tipo={model_type})")

    # Interactively ask for password (hidden)
    try:
        pw = getpass.getpass("Ingrese la contraseña a evaluar (entrada oculta): ")
    except Exception:
        # fallback to simple input if getpass fails
        pw = input("Ingrese la contraseña a evaluar: ")

    res = evaluate_password(pw, model=model, model_type=model_type, feature_order=feature_order, guesses_per_second=args.guessec)
    pretty_print(res, pw, show_pw=args.show_pw)

if __name__ == "__main__":
    main()
