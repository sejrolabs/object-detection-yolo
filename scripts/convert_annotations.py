"""
Conversion des annotations BCCD du format Pascal VOC (XML) vers le format YOLO.

Pascal VOC décrit chaque boîte par ses coins en pixels absolus (xmin, ymin,
xmax, ymax). YOLO attend, pour chaque image, un fichier .txt avec une ligne
par objet au format :

    classe x_centre y_centre largeur hauteur

où toutes les valeurs (sauf la classe) sont normalisées entre 0 et 1,
c'est-à-dire divisées par la largeur/hauteur de l'image. Cette normalisation
rend le format indépendant de la résolution de l'image.

Ce script :
1. Parse chaque fichier XML de data/BCCD_raw/BCCD/Annotations/
2. Convertit les boîtes au format YOLO
3. Répartit les images en train/val (80/20)
4. Organise le tout dans la structure attendue par Ultralytics YOLO :

    data/bccd_yolo/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml

Usage : python -m scripts.convert_annotations
"""

import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path

RAW_DIR = Path("data/BCCD_raw/BCCD")
OUTPUT_DIR = Path("data/bccd_yolo")
CLASSES = ["RBC", "WBC", "Platelets"]  # l'ordre définit l'identifiant numérique de chaque classe (0, 1, 2)
VAL_RATIO = 0.2
RANDOM_SEED = 42


def convert_bbox_to_yolo(size, box):
    """
    Convertit une boîte VOC (xmin, ymin, xmax, ymax) en boîte YOLO
    (x_centre, y_centre, largeur, hauteur), toutes les valeurs normalisées
    entre 0 et 1 par rapport à la taille de l'image.
    """
    img_width, img_height = size
    xmin, ymin, xmax, ymax = box

    x_centre = (xmin + xmax) / 2.0 / img_width
    y_centre = (ymin + ymax) / 2.0 / img_height
    largeur = (xmax - xmin) / img_width
    hauteur = (ymax - ymin) / img_height

    return x_centre, y_centre, largeur, hauteur


def parse_xml_annotation(xml_path):
    """
    Parse un fichier XML Pascal VOC et retourne la liste des annotations
    au format YOLO : [(classe_id, x_centre, y_centre, largeur, hauteur), ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    annotations = []
    for obj in root.iter("object"):
        classe_nom = obj.find("name").text.strip()
        if classe_nom not in CLASSES:
            # Sécurité : si le XML contient une classe inattendue (faute de frappe,
            # variante d'écriture), on la signale plutôt que de planter silencieusement
            print(f"  ⚠️  Classe inconnue ignorée : '{classe_nom}' dans {xml_path.name}")
            continue
        classe_id = CLASSES.index(classe_nom)

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x_c, y_c, w, h = convert_bbox_to_yolo((img_width, img_height), (xmin, ymin, xmax, ymax))
        annotations.append((classe_id, x_c, y_c, w, h))

    return annotations


def main():
    annotations_dir = RAW_DIR / "Annotations"
    images_dir = RAW_DIR / "JPEGImages"

    if not annotations_dir.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {annotations_dir}\n"
            f"As-tu bien exécuté : git clone https://github.com/Shenggan/BCCD_Dataset.git data/BCCD_raw ?"
        )

    xml_files = sorted(annotations_dir.glob("*.xml"))
    print(f"Fichiers d'annotation trouvés : {len(xml_files)}")

    # Split train/val reproductible
    random.seed(RANDOM_SEED)
    xml_files_shuffled = xml_files.copy()
    random.shuffle(xml_files_shuffled)
    n_val = int(len(xml_files_shuffled) * VAL_RATIO)
    val_files = set(xml_files_shuffled[:n_val])

    # Création de la structure de dossiers de sortie
    for split in ["train", "val"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    n_converted = 0
    n_objets_total = 0

    for xml_path in xml_files:
        split = "val" if xml_path in val_files else "train"

        annotations = parse_xml_annotation(xml_path)
        n_objets_total += len(annotations)

        # Nom de l'image correspondante (même nom de base, extension .jpg)
        image_name = xml_path.stem + ".jpg"
        image_path = images_dir / image_name

        if not image_path.exists():
            print(f"  ⚠️  Image manquante pour {xml_path.name}, ignorée")
            continue

        # Copie de l'image vers le bon split
        shutil.copy(image_path, OUTPUT_DIR / split / "images" / image_name)

        # Écriture du fichier de labels YOLO (même nom de base, extension .txt)
        label_path = OUTPUT_DIR / split / "labels" / (xml_path.stem + ".txt")
        with open(label_path, "w") as f:
            for classe_id, x_c, y_c, w, h in annotations:
                f.write(f"{classe_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        n_converted += 1

    # Fichier de configuration attendu par Ultralytics YOLO
    yaml_content = f"""train: {(OUTPUT_DIR / 'train' / 'images').resolve()}
val: {(OUTPUT_DIR / 'val' / 'images').resolve()}

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nConversion terminée : {n_converted} images converties, {n_objets_total} objets annotés")
    print(f"Train : {n_converted - len(val_files & set(xml_files))} images")
    print(f"Val   : {len(val_files)} images")
    print(f"Configuration écrite dans {OUTPUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()