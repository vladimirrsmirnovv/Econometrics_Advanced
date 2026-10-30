from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline


def main() -> None:
    """
    Запускает пайплайн построения таблицы координат населённых пунктов.

    Returns
    -------
    None
        Итоговый CSV и служебные артефакты формируются на диске.
    """

    run_pipeline()


if __name__ == "__main__":
    main()
