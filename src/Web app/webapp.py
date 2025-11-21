# webapp.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


from pw_check_interpretable_interactive import (
    evaluate_password,
    load_feature_order,
    load_model,
    shannon_entropy,
    combinatorial_entropy_estimate
)

# CONFIGURACIÓN GENERAL DE LA APP
st.set_page_config(
    page_title="Evaluador Inteligente de Contraseñas",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Evaluador Inteligente de Contraseñas")
st.markdown("### **Modelo ML — Proyecto Integrador**")
st.write("Analiza tu contraseña con un modelo LightGBM entrenado y heurísticas avanzadas.")

# CARGA DEL MODELO Y FEATURE ORDER
# Detectar ruta absoluta del proyecto independientemente del SO o entorno
# src/ directory
BASE_DIR = Path(__file__).resolve().parent

# project root (one level above)
PROJECT_ROOT = BASE_DIR.parent.parent

# models directory at project root
MODEL_DIR = PROJECT_ROOT / "models" / "lightgbm" / "all"

feature_order = load_feature_order(MODEL_DIR)
model, model_type = load_model(MODEL_DIR)


if model is None:
    st.error("No se pudo cargar el modelo. Verifique la ruta y los archivos.")
else:
    st.success(f"Modelo '{model_type}' cargado correctamente.")

# INPUT DE CONTRASEÑA
st.markdown("## Ingrese la contraseña que desea evaluar")

show_pw = st.checkbox("Mostrar contraseña")

password = st.text_input(
    "Contraseña",
    type="default" if show_pw else "password",
    max_chars=64,
    help="Se analiza solamente localmente. No se almacena.",
)

if st.button("Evaluar contraseña"):
    if not password:
        st.warning(" Ingrese una contraseña primero.")
        st.stop()

    # EVALUACIÓN
    res = evaluate_password(
        password,
        model=model,
        model_type=model_type,
        feature_order=feature_order,
        guesses_per_second=1e6
    )

    st.markdown("## Resultado del Modelo")
    if "model_prob_secure" in res:
        prob = res["model_prob_secure"]
        st.metric("Probabilidad de ser segura", f"{prob:.4f}")

        # Barra visual
        st.progress(min(max(prob, 0), 1))
    else:
        st.error("No se pudo calcular probabilidad.")

    # INFORMACIÓN GENERAL
    st.markdown("## Información General")

    col1, col2, col3 = st.columns(3)
    feat = res["features"]

    col1.metric("Longitud", feat["length"])
    col2.metric("Charset Size", feat["charset_size"])
    col3.metric("Entropía Shannon (por char)", f"{shannon_entropy(password):.2f}")

    st.write("### Entropía Combinatoria")
    st.code(f"{combinatorial_entropy_estimate(password):.2f} bits")

    # TIEMPOS DE CRACKEO
    st.markdown("## Estimaciones de Crack Time")

    crack = res["crack_time"]
    df = pd.DataFrame({
        "Escenario": crack.keys(),
        "Tiempo (Shannon)": [crack[k]["time_shan"] for k in crack],
        "Tiempo (Combinatorial)": [crack[k]["time_comb"] for k in crack],
    })

    st.table(df)

    # GRÁFICO RADAR (FEATURES CLAVE)
    st.markdown("## Radar — Complejidad de la Contraseña")

    radar_features = {
        "num_lower": "Minúsculas",
        "num_upper": "Mayúsculas",
        "num_digits": "Dígitos",
        "num_symbols": "Símbolos",
        "_max_same_char_run": "Repeticiones",
        "has_seq_num": "Secuencias numéricas",
        "has_year": "Años detectados"
    }

    labels = list(radar_features.values())
    values = [feat[k] for k in radar_features.keys()]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        name="Características"
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
    for r in res["recommendations"]:
        st.markdown(f"- {r}")

