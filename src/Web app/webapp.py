# webapp.py
import os
import base64
import io
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from pw_check_interpretable_interactive import (
    evaluate_password,
    load_feature_order,
    load_model,
    shannon_entropy,
    combinatorial_entropy_estimate,
)

# ============================
# CONFIG STREAMLIT
# ============================

st.set_page_config(
    page_title="Evaluador Inteligente de Contraseñas",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Evaluador Inteligente de Contraseñas")
st.markdown("### **Modelo ML — Proyecto Integrador**")
st.write("Analiza tu contraseña con un modelo LightGBM entrenado y heurísticas avanzadas.")



# CONFIG GITHUB 

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_FILE_PATH = os.getenv("GITHUB_FILE_PATH", "data/password_evals.csv")


def append_row_to_github_csv(row: dict) -> None:
    """
    Añade una fila a un CSV almacenado en un repositorio de GitHub.
    Si el archivo no existe, lo crea.
    """

    if not (GITHUB_TOKEN and GITHUB_OWNER and GITHUB_REPO and GITHUB_BRANCH and GITHUB_FILE_PATH):
        raise RuntimeError(
            "Variables de entorno de GitHub incompletas. "
            "Se requieren: GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_FILE_PATH."
        )

    api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # 1. Intentar obtener el archivo actual en la rama indicada
    resp = requests.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH})

    if resp.status_code == 200:
        data = resp.json()
        sha = data["sha"]
        content_b64 = data["content"]
        # El contenido viene en base64 (a veces con saltos de línea)
        content_bytes = base64.b64decode(content_b64)
        csv_text = content_bytes.decode("utf-8")

        # Cargar CSV existente a DataFrame
        existing_df = pd.read_csv(io.StringIO(csv_text))

        # Agregar nueva fila
        new_df = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)

    elif resp.status_code == 404:
        # No existe el archivo: creamos un DataFrame nuevo
        sha = None
        new_df = pd.DataFrame([row])
    else:
        raise RuntimeError(
            f"Error al leer archivo en GitHub: {resp.status_code} - {resp.text}"
        )

    # 2. Convertir DataFrame a CSV
    out_csv = new_df.to_csv(index=False)
    out_b64 = base64.b64encode(out_csv.encode("utf-8")).decode("utf-8")

    payload = {
        "message": f"Add password eval at {datetime.utcnow().isoformat()}",
        "content": out_b64,
        "branch": GITHUB_BRANCH,
    }
    if sha is not None:
        payload["sha"] = sha

    # 3. PUT para crear/actualizar el archivo
    put_resp = requests.put(api_url, headers=headers, json=payload)
    if put_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Error al guardar archivo en GitHub: {put_resp.status_code} - {put_resp.text}"
        )
    


# 
# CARGA DEL MODELO
# 

# BASE_DIR = carpeta donde está este archivo (webapp.py)
BASE_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = raíz del proyecto (un nivel por encima de "src" o "Web app")
PROJECT_ROOT = BASE_DIR.parent.parent

# Ruta del modelo: <project_root>/models/lightgbm/all
MODEL_DIR = PROJECT_ROOT / "models" / "lightgbm" / "all"

feature_order = load_feature_order(MODEL_DIR)
model, model_type = load_model(MODEL_DIR)

if model is None:
    st.error("⚠ No se pudo cargar el modelo. Verifique la ruta y los archivos.")
else:
    st.success(f"Modelo cargado correctamente ✔ (tipo: {model_type})")
    if feature_order is not None:
        st.caption(f"Features del modelo: {len(feature_order)}")
    else:
        st.caption("No se encontró featurenames.json, usando orden por defecto del extractor.")


# 
# INPUT DE CONTRASEÑA

st.markdown("## Ingrese la contraseña que desea evaluar")

password = st.text_input(
    "Contraseña",
    type="password",
    max_chars=64,
    help=(
        "La contraseña se procesa solo para este análisis. "
        "También se guarda junto con sus features en un CSV en GitHub para el análisis del proyecto."
    ),
)


# BOTÓN: EVALUAR 

if st.button("Evaluar contraseña"):
    if not password:
        st.warning("⚠ Ingrese una contraseña primero.")
        st.stop()

    #  EVALUACIÓN 
    res = evaluate_password(
        password,
        model=model,
        model_type=model_type,
        feature_order=feature_order,
        guesses_per_second=1e6,
    )

    # RESULTADO DEL MODELO 
    st.markdown("## Resultado del Modelo")
    if "model_prob_secure" in res:
        prob = res["model_prob_secure"]
        st.metric("Probabilidad de ser segura", f"{prob:.4f}")
        st.progress(min(max(prob, 0), 1))
    else:
        st.error("No se pudo calcular probabilidad con el modelo.")

    # INFO GENERAL 
    st.markdown("## Información General")

    feat = res["features"]
    col1, col2, col3 = st.columns(3)

    col1.metric("Longitud", feat.get("length", 0))
    col2.metric("Charset Size", feat.get("charset_size", 0))
    col3.metric("Entropía Shannon (por char)", f"{shannon_entropy(password):.2f}")

    st.write("### Entropía Combinatoria")
    st.code(f"{combinatorial_entropy_estimate(password):.2f} bits")

    # CRACK TIME 
    st.markdown("## Estimaciones de Crack Time")

    crack = res["crack_time"]
    df_crack = pd.DataFrame({
        "Escenario": list(crack.keys()),
        "Tiempo (Shannon)": [crack[k]["time_shan"] for k in crack],
        "Tiempo (Combinatorial)": [crack[k]["time_comb"] for k in crack],
    })
    st.table(df_crack)

    # RADAR 
    st.markdown("## Radar — Complejidad de la Contraseña")

    radar_features = {
        "num_lower": "Minúsculas",
        "num_upper": "Mayúsculas",
        "num_digits": "Dígitos",
        "num_symbols": "Símbolos",
        "_max_same_char_run": "Repeticiones",
        "has_seq_num": "Secuencias numéricas",
        "has_year": "Años detectados",
    }

    labels = list(radar_features.values())
    values = [feat.get(k, 0) for k in radar_features.keys()]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        name="Características",
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=500,
    )
    st.plotly_chart(fig)

    # FEATURES COMPLETOS 
    st.markdown("## Features detectados")
    st.json(res["features"])

    # RECOMENDACIONES 
    st.markdown("## 🛠 Recomendaciones")
    if res["recommendations"]:
        for r in res["recommendations"]:
            st.markdown(f"- {r}")
    else:
        st.markdown("_No se generaron recomendaciones específicas._")

    # GUARDADO AUTOMÁTICO EN GITHUB 
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "password_plain": password,  # texto sin enmascarar (solo para análisis del proyecto)
        "model_prob_secure": res.get("model_prob_secure", None),
    }
    # Añadir todas las características al row
    row.update(res["features"])

    try:
        append_row_to_github_csv(row)
        st.success("Evaluación guardada en GitHub correctamente.")
    except Exception as e:
        st.error(f"Error al guardar en GitHub: {e}")
