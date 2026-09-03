# À lancer depuis la RACINE du projet avec : streamlit run app/app.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd

st.set_page_config(page_title="Détection de cellules sanguines — YOLO", page_icon="🩸", layout="wide")

MODEL_PATH = "runs/detect/outputs/yolo_runs/bccd_full/weights/best.pt"

# ─────────────────────────────────────────────
# Chargement du modèle (mis en cache pour ne pas le recharger à chaque interaction)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ─────────────────────────────────────────────
# En-tête
# ─────────────────────────────────────────────
st.title("🩸 Détection de cellules sanguines (YOLO)")
st.markdown(
    "Détection automatique de **globules rouges (RBC)**, **globules blancs (WBC)** et "
    "**plaquettes (Platelets)** sur des images de frottis sanguin, via un modèle **YOLOv8** "
    "fine-tuné sur le dataset [BCCD](https://github.com/Shenggan/BCCD_Dataset)."
)

with st.expander("ℹ️ Performance du modèle (jeu de validation)"):
    st.markdown(
        "| Classe | Precision | Recall | mAP50 |\n"
        "|---|---|---|---|\n"
        "| RBC | 0.787 | 0.879 | 0.896 |\n"
        "| WBC | 0.974 | 1.000 | 0.995 |\n"
        "| Platelets | 0.853 | 0.879 | 0.924 |\n"
        "| **Global** | **0.871** | **0.919** | **0.938** |\n\n"
        "⚠️ Modèle entraîné sur un petit dataset (364 images) à des fins de démonstration "
        "— non validé pour un usage médical réel."
    )

st.divider()

# ─────────────────────────────────────────────
# Barre latérale — réglages
# ─────────────────────────────────────────────
st.sidebar.header("Réglages")
confidence_threshold = st.sidebar.slider(
    "Seuil de confiance minimum", min_value=0.0, max_value=1.0, value=0.25, step=0.05,
    help="Les détections en dessous de ce seuil de confiance ne sont pas affichées."
)

st.sidebar.divider()
st.sidebar.subheader("Images d'exemple")
st.sidebar.caption(
    "Pas d'image sous la main ? Utilise une image du dossier "
    "`data/bccd_yolo/val/images/` de ton repo pour tester rapidement."
)

# ─────────────────────────────────────────────
# Upload d'image
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Uploadez une image de frottis sanguin (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with st.spinner("Détection en cours..."):
        results = model.predict(image, conf=confidence_threshold, verbose=False)
        result = results[0]

    with col2:
        st.subheader("Détections")
        # result.plot() retourne l'image annotée (boîtes + labels) au format numpy (BGR)
        annotated_image = result.plot()[:, :, ::-1]  # conversion BGR -> RGB pour un affichage correct
        st.image(annotated_image, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────
    # Tableau récapitulatif des détections
    # ─────────────────────────────────────────────
    st.subheader("📋 Détail des détections")

    if len(result.boxes) == 0:
        st.info("Aucune cellule détectée avec ce seuil de confiance. Essayez de le réduire dans la barre latérale.")
    else:
        detections = []
        for box in result.boxes:
            classe_id = int(box.cls[0])
            classe_nom = model.names[classe_id]
            confiance = float(box.conf[0])
            detections.append({"Classe": classe_nom, "Confiance": f"{confiance:.2%}"})

        df_detections = pd.DataFrame(detections)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Total de cellules détectées", len(detections))
            comptage = df_detections["Classe"].value_counts()
            for classe, count in comptage.items():
                st.metric(classe, count)

        with col_b:
            st.dataframe(df_detections, use_container_width=True, hide_index=True)

else:
    st.info("👆 Uploadez une image de frottis sanguin pour lancer la détection.")

# ─────────────────────────────────────────────
# Pied de page
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Projet réalisé par [sejrolabs](https://github.com/sejrolabs) — "
    "[Code source](https://github.com/sejrolabs/object-detection-yolo)"
)