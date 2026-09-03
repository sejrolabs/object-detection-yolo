"""
Fine-tuning de YOLOv8 (modèle nano) sur le dataset BCCD converti.

On part du modèle yolov8n.pt, pré-entraîné sur COCO (80 classes génériques),
et on l'adapte à nos 3 classes spécifiques (RBC, WBC, Platelets) via
transfer learning — même principe que ResNet18 pour la classification
d'images (projet image-classifier-cnn), mais appliqué à la détection.

MODE TEST (par défaut ici) : très peu d'époques (3), pour valider que tout
le pipeline fonctionne bout en bout (chargement des données, entraînement,
sauvegarde) avant de lancer un entraînement long et sérieux.

Usage :
    python -m scripts.train_yolo              # mode test rapide (3 époques)
    python -m scripts.train_yolo --full        # entraînement complet (voir EPOCHS_FULL)
"""

import argparse
from ultralytics import YOLO

DATA_YAML = "data/bccd_yolo/data.yaml"

EPOCHS_TEST = 3       # mode test : juste pour vérifier que le pipeline fonctionne
EPOCHS_FULL = 50       # entraînement complet (~2h05 estimées sur CPU, ~2.5 min/époque observées en test)

IMG_SIZE = 416          # résolution réduite par rapport aux 640 par défaut,
                         # pour accélérer l'entraînement sur CPU (compromis vitesse/précision
                         # raisonnable vu la petite taille du dataset et des objets)
BATCH_SIZE = 8           # petit batch, adapté au CPU (pas de contrainte de mémoire GPU ici,
                         # mais un batch trop grand ralentit chaque itération sur CPU)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Lance l'entraînement complet plutôt que le mode test")
    args = parser.parse_args()

    epochs = EPOCHS_FULL if args.full else EPOCHS_TEST
    run_name = "bccd_full" if args.full else "bccd_test"

    print(f"{'='*50}")
    print(f"Mode : {'ENTRAÎNEMENT COMPLET' if args.full else 'TEST RAPIDE'}")
    print(f"Époques : {epochs}")
    print(f"{'='*50}\n")

    # Chargement du modèle YOLOv8 nano pré-entraîné sur COCO
    # (le fichier .pt est téléchargé automatiquement au premier lancement, ~6 Mo)
    model = YOLO("yolov8n.pt")

    # Lancement du fine-tuning sur notre dataset
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        name=run_name,
        project="outputs/yolo_runs",
        patience=20,       # arrêt anticipé si pas d'amélioration pendant 20 époques (ignoré en mode test)
        verbose=True,
    )

    print(f"\nEntraînement terminé.")
    print(f"Résultats et poids sauvegardés dans : outputs/yolo_runs/{run_name}/")
    print(f"Le meilleur modèle est dans : outputs/yolo_runs/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()