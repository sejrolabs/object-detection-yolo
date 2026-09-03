# 🩸 Détection de cellules sanguines — YOLOv8

Détection automatique de globules rouges, globules blancs et plaquettes sur des images de frottis sanguin, via **fine-tuning de YOLOv8** sur le dataset **BCCD**, avec une démo interactive **Streamlit**.

## 📌 Contexte

Ce projet illustre la **détection d'objets** — une tâche différente de la classification d'images (projet `image-classifier-cnn`) : il ne s'agit plus seulement de dire "quel objet est sur l'image", mais de prédire **où** se trouve chaque objet (boîte englobante) et **combien** d'objets sont présents. Application concrète ici : le comptage automatique de cellules sanguines, une tâche réelle en analyse biomédicale (numération formule sanguine).

## 📊 Dataset

- **Source** : [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset) (Blood Cell Count and Detection)
- **Volume** : 364 images, 4 888 objets annotés
- **3 classes** : RBC (globules rouges), WBC (globules blancs), Platelets (plaquettes)
- **Format d'origine** : Pascal VOC (XML) — converti au format YOLO via un script dédié (`scripts/convert_annotations.py`)
- **Split** : 292 images train / 72 images validation (80/20, reproductible)

## 🛠️ Méthodologie

### Pourquoi YOLO
YOLO ("You Only Look Once") prédit en une seule passe du réseau à la fois la localisation (boîtes englobantes) et la classification des objets présents sur une image — contrairement à des approches en plusieurs étapes plus anciennes. Cette architecture en fait un standard pour la détection en temps réel.

### Conversion des annotations (VOC → YOLO)
Pascal VOC décrit chaque boîte par ses coins en pixels absolus (`xmin`, `ymin`, `xmax`, `ymax`). YOLO attend un format différent : centre de la boîte + largeur/hauteur, normalisés entre 0 et 1 (indépendant de la résolution de l'image). Le script `scripts/convert_annotations.py` effectue cette conversion et organise automatiquement la structure de dossiers attendue par Ultralytics (train/val, `data.yaml`).

### Transfer learning
Le modèle part de **YOLOv8n** ("nano", le plus petit de la famille), pré-entraîné sur COCO (80 classes génériques), et est fine-tuné sur les 3 classes spécifiques du dataset BCCD — même principe que pour ResNet18 dans le projet de classification d'images, appliqué ici à la détection.

## 📈 Résultats

### Entraînement en deux temps
Un premier entraînement rapide (3 époques) a permis de valider que le pipeline complet fonctionnait (conversion des données, chargement, entraînement, évaluation) avant de lancer un entraînement plus long.

| | 3 époques | 50 époques |
|---|---|---|
| **mAP50 global** | 0.874 | **0.938** |
| Durée | ~7 min | 1h22 (CPU) |

### Résultats détaillés par classe (50 époques)

| Classe | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| RBC | 0.787 | 0.879 | 0.896 | 0.642 |
| WBC | 0.974 | 1.000 | 0.995 | 0.824 |
| Platelets | 0.853 | 0.879 | 0.924 | 0.500 |
| **Global** | **0.871** | **0.919** | **0.938** | **0.655** |

### Analyse par classe

- **WBC (globules blancs)** : quasi-parfait (mAP50 = 0.995) dès le départ — cellules grandes et visuellement distinctes
- **Platelets (plaquettes)** : la plus forte progression entre 3 et 50 époques (mAP50 : 0.764 → 0.924, recall : 0.415 → 0.879). Leur petite taille les rend intrinsèquement plus difficiles à détecter (moins de pixels = moins d'information exploitable par le modèle), un défi classique en détection d'objets
- **RBC (globules rouges)** : bon recall mais precision plus limitée (0.787) — cohérent avec leur nombre très élevé (832 instances sur seulement 71 images de validation) et leur chevauchement fréquent sur les images, rendant les frontières entre cellules ambiguës

![Courbes d'entraînement](outputs/results.png)
![Matrice de confusion](outputs/confusion_matrix.png)

## 🖥️ Application interactive

L'application Streamlit permet d'uploader une image de frottis sanguin et de visualiser en direct :
- L'image annotée avec les boîtes englobantes détectées
- Un seuil de confiance ajustable
- Le décompte de cellules détectées par classe

### Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## 🚀 Comment reproduire ce projet

```bash
git clone https://github.com/sejrolabs/object-detection-yolo.git
cd object-detection-yolo

pip install -r requirements.txt

# 1. Télécharger le dataset
git clone https://github.com/Shenggan/BCCD_Dataset.git data/BCCD_raw

# 2. Convertir les annotations VOC -> YOLO
python -m scripts.convert_annotations

# 3. Entraîner (mode test rapide, ou --full pour l'entraînement complet ~1h20 sur CPU)
python -m scripts.train_yolo
python -m scripts.train_yolo --full

# 4. Lancer la démo
streamlit run app/app.py
```

## 📦 Stack technique

- **Langage** : Python 3.12
- **Détection d'objets** : Ultralytics YOLOv8
- **Traitement des annotations** : xml.etree (standard library)
- **Interface** : Streamlit
- **Environnement d'entraînement** : CPU local (Intel Core i5-8365U)

## 🔍 Limites et pistes d'amélioration

- Entraînement réalisé sur CPU faute de GPU disponible — un GPU permettrait des résolutions d'image plus élevées (640px au lieu de 416px ici) et davantage d'époques dans un temps raisonnable
- Petit dataset (364 images) : les performances, bien que solides, ne reflètent pas un usage médical de production — ce modèle n'a pas vocation à être utilisé cliniquement
- La classe RBC (globules rouges), la plus dense et la plus chevauchante, reste la plus difficile à détecter précisément — une architecture plus grande (YOLOv8s/m) ou davantage de données pourrait améliorer ce point
- Pas de validation croisée : un seul split train/val a été utilisé