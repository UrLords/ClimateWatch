"""Train an isolated candidate; never changes the running detector configuration."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlretrieve

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".local/training/taco-small-v1"
OUTPUT = ROOT / "models/candidate-waste-taco-v1"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((DATA / "manifest.json").read_text())
    assert dataset["counts"] == {"train": 180, "val": 30, "test": 40}
    base = OUTPUT / "yolo26n-seg-pretrained.pt"
    url = (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt"
    )
    if not base.exists():
        urlretrieve(url, base)
    torch.set_num_threads(8)
    started = time.monotonic()
    model = YOLO(base)

    def bound_time(trainer):
        if time.monotonic() - started > 1200:
            trainer.stop = True

    model.add_callback("on_train_epoch_end", bound_time)
    config = {
        "data": str(DATA / "dataset.yaml"),
        "epochs": 12,
        "imgsz": 416,
        "batch": 8,
        "workers": 0,
        "device": "cpu",
        "seed": 31031,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "patience": 12,
        "amp": False,
        "cache": False,
        "project": str(OUTPUT),
        "name": "run",
        "exist_ok": False,
        "plots": False,
        "save": True,
        "mosaic": 0.5,
        "close_mosaic": 3,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 5.0,
        "fliplr": 0.5,
        "freeze": 0,
    }
    (OUTPUT / "experiment.json").write_text(
        json.dumps(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "base_model_url": url,
                "base_model_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "base_model_license": "AGPL-3.0",
                "dataset_manifest": str(DATA / "manifest.json"),
                "dataset_manifest_sha256": hashlib.sha256(
                    (DATA / "manifest.json").read_bytes()
                ).hexdigest(),
                "hardware": {
                    "torch": torch.__version__,
                    "cuda": False,
                    "cpu_threads": 8,
                },
                "config": config,
                "max_training_seconds": 1200,
                "deployment_decision": "candidate only; held-out comparison required",
            },
            indent=2,
        )
    )
    model.train(**config)
    best = Path(model.trainer.best)
    metrics = YOLO(best).val(
        data=str(DATA / "dataset.yaml"),
        split="test",
        imgsz=416,
        batch=8,
        device="cpu",
        workers=0,
        plots=False,
        project=str(OUTPUT),
        name="heldout",
        verbose=False,
    )
    result = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "best_weights": str(best),
        "weights_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
        "test_metrics": metrics.results_dict,
        "test_images": 40,
        "deployed": False,
    }
    (OUTPUT / "training-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
