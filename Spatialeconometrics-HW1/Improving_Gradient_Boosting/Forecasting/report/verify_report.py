"""Verify the sole final report without changing its source or PDF."""
from pathlib import Path
import hashlib
import json
import re
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'Прогноз_РТО_финальный_отчёт.tex'
PDF = HERE / 'output/pdf/Прогноз_РТО_финальный_отчёт.pdf'
QA = ROOT / 'tmp/pdfs/final_forecast_report'
CHECKS = HERE / 'quality_checks.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = json.loads(CHECKS.read_text(encoding='utf-8-sig'))
    assert sha(SOURCE) == previous['source_sha256'], 'Final LaTeX source changed'
    reader = PdfReader(PDF)
    text = '\n'.join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) == 11
    assert sum(len(page.images) for page in reader.pages) == 3
    assert '??' not in text, 'Unresolved reference'
    for missing in ('Данные в исходном проекте', 'Диаграмма сравнений находится',
                    'Диаграмма способов усреднения находится', 'Схема алгоритма (файл'):
        assert missing not in text, 'Missing table or figure'
    for fragment in ('Гончаров Алексей', '90,83', '90,78', '4,333', '4,276',
                     '5,122', '0,0710', 'Воронцов'):
        assert fragment in text, fragment
    build_log = QA / 'build/final_report.log'
    log_path = build_log if build_log.exists() else QA / 'compilation.log'
    if log_path.exists():
        log = log_path.read_text(encoding='utf-8', errors='replace')
        assert not re.search(r'Overfull|undefined references|LaTeX Error|Label\(s\) may have changed', log)
    else:
        # A clean clone has no local build logs. Its unchanged PDF inherits the
        # layout checks recorded after the original visual review.
        assert sha(PDF) == previous['pdf_sha256'], 'Changed PDF requires fresh build logs'
        assert previous['no_overfull_boxes'] and previous['all_references_resolved']
    result = dict(previous, status='passed', source_file=SOURCE.name,
                  pdf_file='output/pdf/' + PDF.name, source_sha256=sha(SOURCE),
                  pdf_sha256=sha(PDF), pages=len(reader.pages), figures=3,
                  no_missing_tables_or_figures=True, all_references_resolved=True,
                  no_overfull_boxes=True)
    CHECKS.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'pages', 'figures',
                     'source_sha256', 'pdf_sha256')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
