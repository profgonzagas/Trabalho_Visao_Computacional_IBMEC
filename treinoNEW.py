"""
Pipeline aprimorado de Transfer Learning para deteccao de armas de fogo.

Melhorias em relacao a treino.py:
  * Treina YOLOv8n e YOLO26n no mesmo dataset (comparacao justa).
  * Valida em val + test e roda inferencia no video.
  * Hiperparametros alinhados as recomendacoes do relatorio.
  * Resumo consolidado em CSV.
  * Reprodutibilidade (seed), uso correto de DEVICE, validacoes de path.
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import torch
from ultralytics import YOLO

# Filtra apenas categorias ruidosas, nao tudo
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Configuracao
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PATH_CONFIG_YAML = ROOT / "config.yaml"
PATH_VIDEO = ROOT / "video" / "video_teste.mov"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Modelos a comparar (rotulo -> arquivo de pesos pre-treinados)
MODELOS = {
    "yolov8n": "yolov8n.pt",
    "yolo26n": "yolo26n.pt",
}

# Hiperparametros 
TRAIN_ARGS = dict(
    epochs=100,
    imgsz=640,
    batch=4,
    patience=30,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    freeze=10,                 # congela backbone (transfer learning forte)
    # augmentations agressivas para dataset pequeno
    mosaic=1.0,
    mixup=0.3,
    copy_paste=0.3,
    degrees=10.0,
    scale=0.7,
    fliplr=0.5,
    flipud=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    # operacionais
    amp=True,
    plots=True,
    seed=SEED,
    workers=4,                 # reduz overhead de spawn no Windows
    exist_ok=True,
    verbose=True,
)



def coletar_metricas(model: YOLO, particao: str, data_yaml: Path) -> dict:
    """Roda model.val em uma particao e retorna metricas como dicionario."""
    if particao not in ("val", "test"):
        raise ValueError(f"particao invalida: {particao}")

    print(f"\n{'-' * 60}\nValidacao - particao: {particao}\n{'-' * 60}")
    m = model.val(data=str(data_yaml), split=particao, verbose=False, device=DEVICE)
    metricas = {
        "particao": particao,
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }
    print(f"  mAP@0.50      : {metricas['mAP50']:.4f}")
    print(f"  mAP@0.50:0.95 : {metricas['mAP50_95']:.4f}")
    print(f"  Precisao      : {metricas['precision']:.4f}")
    print(f"  Recall        : {metricas['recall']:.4f}")
    for i, cls_idx in enumerate(m.box.ap_class_index):
        nome = model.names[int(cls_idx)]
        print(f"  classe {int(cls_idx)} {nome:<10} AP@0.5 = {m.box.ap50[i]:.4f}")
    return metricas


def inferencia_video(model: YOLO, video_path: Path, save_dir: Path) -> None:
    """Roda predict no video e salva resultado anotado."""
    if not video_path.exists():
        print(f"[skip] video nao encontrado: {video_path}")
        return
    print(f"\nInferencia em video: {video_path.name}")
    model.predict(
        source=str(video_path),
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        device=DEVICE,
        verbose=False,
    )
    print(f"  resultado salvo em: {save_dir}")


def validar_ambiente() -> None:
    if not PATH_CONFIG_YAML.exists():
        sys.exit(f"ERRO: config.yaml nao encontrado em {PATH_CONFIG_YAML}")
    print(f"Device         : {DEVICE}")
    print(f"Config YAML    : {PATH_CONFIG_YAML}")
    print(f"Video          : {PATH_VIDEO} (existe={PATH_VIDEO.exists()})")
    import ultralytics
    print(f"Ultralytics    : {ultralytics.__version__}")
    print(f"PyTorch        : {torch.__version__}")
    if DEVICE == "cuda":
        print(f"GPU            : {torch.cuda.get_device_name(0)}")



def main() -> None:
    validar_ambiente()
    torch.manual_seed(SEED)

    resumo: list[dict] = []

    for label, pesos in MODELOS.items():
        projeto = f"transfer_{label}_ep{TRAIN_ARGS['epochs']}"
        nome_run = f"{label}_transfer"
        print(f"\n{'=' * 70}\nTreinando {label} ({pesos})\n{'=' * 70}")

        modelo = YOLO(pesos)
        try:
            results = modelo.train(
                data=str(PATH_CONFIG_YAML),
                device=DEVICE,
                project=projeto,
                name=nome_run,
                **TRAIN_ARGS,
            )
        except Exception as exc:
            print(f"[erro] treino de {label} falhou: {exc}")
            continue

        save_dir = Path(results.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        if not best_pt.exists():
            print(f"[erro] best.pt nao gerado em {best_pt}")
            continue

        best_model = YOLO(best_pt)

        for particao in ("val", "test"):
            try:
                m = coletar_metricas(best_model, particao, PATH_CONFIG_YAML)
                m["modelo"] = label
                m["best_pt"] = str(best_pt)
                resumo.append(m)
            except Exception as exc:
                print(f"[erro] val em {particao} falhou ({label}): {exc}")

        inferencia_video(best_model, PATH_VIDEO, save_dir / "video_pred")

    # ------------------------------------------------------------------
    # Resumo consolidado
    # ------------------------------------------------------------------
    if not resumo:
        print("\nNenhum resultado consolidado.")
        return

    out_csv = ROOT / "runs" / "comparacao_modelos.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["modelo", "particao", "mAP50", "mAP50_95", "precision", "recall", "best_pt"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(resumo)
    print(f"\nResumo consolidado salvo em: {out_csv}")
    print(f"\n{'modelo':<10} {'particao':<8} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}")
    for r in resumo:
        print(f"{r['modelo']:<10} {r['particao']:<8} "
              f"{r['mAP50']:>8.4f} {r['mAP50_95']:>10.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f}")


if __name__ == "__main__":
    main()
