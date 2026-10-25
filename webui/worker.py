"""
Runs one training job in its own process: ``python -m webui.worker <job_dir>``.

SIGINT stops training early; the best checkpoint so far is still exported.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def _write_status(job_dir: Path, **fields):
    path = job_dir / "status.json"
    try:
        status = json.loads(path.read_text())
    except (OSError, ValueError):
        status = {}
    status.update(fields)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status))
    tmp.replace(path)


def main(job_dir: Path):
    params = json.loads((job_dir / "params.json").read_text())
    _write_status(job_dir, state="running", pid=os.getpid(), started=time.time())
    try:
        from nam.models.metadata import UserMetadata
        from nam.train import core
        from nam.train.metadata import TRAINING_KEY

        user_metadata = UserMetadata(
            name=params.get("name") or None,
            modeled_by=params.get("modeled_by") or None,
        )
        output = core.train(
            input_path=str(job_dir / "input.wav"),
            output_path=str(job_dir / "output.wav"),
            train_path=str(job_dir / "train"),
            epochs=params["epochs"],
            latency=params["latency"],
            batch_size=params["batch_size"],
            seed=params["seed"],
            silent=True,
            ignore_checks=params["ignore_checks"],
            local=False,
            user_metadata=user_metadata,
        )
        if output.model is None:
            checks = output.metadata.data.checks
            print("Data checks failed:", flush=True)
            print(
                json.dumps(checks.model_dump() if checks else None, indent=2),
                flush=True,
            )
            _write_status(
                job_dir,
                state="failed",
                message="Data quality checks failed. Fix the recording or "
                "tick 'Ignore data checks'.",
                ended=time.time(),
            )
            return 1
        outdir = job_dir / "model"
        outdir.mkdir(exist_ok=True)
        basename = params["basename"]
        output.model.net.export(
            outdir,
            basename=basename,
            user_metadata=user_metadata,
            other_metadata={TRAINING_KEY: output.metadata.model_dump()},
        )
        _write_status(
            job_dir,
            state="done",
            model=f"model/{basename}.nam",
            validation_esr=output.metadata.validation_esr,
            ended=time.time(),
        )
        return 0
    except KeyboardInterrupt:
        _write_status(job_dir, state="cancelled", ended=time.time())
        return 130
    except BaseException as e:
        traceback.print_exc()
        _write_status(job_dir, state="failed", message=str(e), ended=time.time())
        return 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
