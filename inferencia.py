"""
inferencia.py
-------------
Script de inferência de UMA imagem usando um modelo YOLO treinado (best.pt).

Uso:
    python inferencia.py --model runs/detect/transfer_yolo26n_ep100/yolo26n_transfer/weights/best.pt 
                         --image caminho/para/imagem.jpg 
                         --conf 0.25

Saída:
    Imagem anotada com bounding boxes salva em output/<nome>_pred.jpg
    Lista das detecções (classe, confiança, bbox) impressa no console.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência YOLO em uma imagem.")
    parser.add_argument("--model", required=True, help="Caminho do best.pt")
    parser.add_argument("--image", required=True, help="Caminho da imagem de entrada")
    parser.add_argument("--conf", type=float, default=0.25, help="Confiança mínima (0-1)")
    parser.add_argument("--out", default="output", help="Pasta de saída")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), conf=args.conf, verbose=False)

    res = results[0]
    annotated = res.plot()
    out_path = out_dir / f"{image_path.stem}_pred.jpg"
    cv2.imwrite(str(out_path), annotated)

    print(f"\nDetecções em {image_path.name}:")
    if res.boxes is None or len(res.boxes) == 0:
        print("  (nenhuma detecção acima do threshold)")
    else:
        names = res.names
        for b in res.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            print(
                f"  - {names[cls_id]:8s}  conf={conf:.3f}  "
                f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
            )

    print(f"\nImagem anotada salva em: {out_path}")


if __name__ == "__main__":
    main()
