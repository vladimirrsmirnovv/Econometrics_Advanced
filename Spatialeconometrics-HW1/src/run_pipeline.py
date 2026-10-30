"""Запуск конвейера по шагам из `Runs/<название>.jsonl`.

Каждый шаг выполняется отдельным процессом. После шага проверяется, что
его выходные файлы появились, и в `Runs/<название>.history.jsonl`
дописывается запись с длительностью и контрольными суммами выходов.
Совпадение сумм между запусками подтверждает воспроизводимость. Первый
упавший шаг останавливает конвейер.

Запуск: `python -m src.run_pipeline preprocessing`, продолжение с шага:
`python -m src.run_pipeline preprocessing --from 8`.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime

from src import paths


def read_steps(pipeline: str) -> list[dict]:
    """Читает описание шагов, по одному JSON-объекту на строку."""
    pipeline_path = paths.RUNS / f"{pipeline}.jsonl"
    assert pipeline_path.exists(), f"нет описания конвейера {pipeline_path}"
    lines = pipeline_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def file_checksum(relative_path: str) -> str:
    """Возвращает SHA-256 файла, путь задан от корня проекта."""
    return hashlib.sha256((paths.PROJECT_ROOT / relative_path).read_bytes()).hexdigest()


def run_step(step: dict) -> dict:
    """Выполняет шаг и возвращает запись для журнала."""
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", step["module"]],
        cwd=paths.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    missing = [
        path for path in step["outputs"] if not (paths.PROJECT_ROOT / path).exists()
    ]
    is_ok = result.returncode == 0 and not missing
    return {
        "step": step["step"],
        "name": step["name"],
        "status": "ok" if is_ok else "failed",
        "seconds": round(time.perf_counter() - started, 1),
        "message": output.splitlines()[-1] if output else "",
        "outputs": {path: file_checksum(path) for path in step["outputs"]}
        if is_ok
        else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pipeline", help="название конвейера, файл Runs/<название>.jsonl"
    )
    parser.add_argument("--from", dest="first_step", type=int, default=1)
    arguments = parser.parse_args()

    steps = read_steps(arguments.pipeline)
    history_path = paths.RUNS / f"{arguments.pipeline}.history.jsonl"
    run_id = datetime.now().isoformat(timespec="seconds")
    with history_path.open("a", encoding="utf-8") as history:
        for step in steps:
            if step["step"] < arguments.first_step:
                continue
            record = {
                "run_id": run_id,
                "pipeline": arguments.pipeline,
                **run_step(step),
            }
            history.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{record['status']}] {step['step']:>2}. {step['name']}: {record['message']}"
            )
            if record["status"] != "ok":
                sys.exit(1)


if __name__ == "__main__":
    main()
