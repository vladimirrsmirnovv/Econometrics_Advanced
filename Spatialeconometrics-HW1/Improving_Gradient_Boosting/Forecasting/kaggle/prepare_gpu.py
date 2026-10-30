"""Prepare the private Kaggle input package and GPU-only notebook; no credentials."""
from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path


import hashlib
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebook_api import ROOT, NOTEBOOK, NOTEBOOK_SHA256, load_recipe, matrices
from run_validation import get_fold
from gpu_runner import fingerprint

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xgboost-version', default='3.1.3', choices=['3.1.3', '3.2.0', '3.4.1'])
    args = parser.parse_args()
    owner = json.loads((HERE / 'account.json').read_text())['owner']
    upload = HERE / 'upload'
    kernel = HERE / 'kernel'
    upload.mkdir(exist_ok=True)
    kernel.mkdir(exist_ok=True)
    shutil.copyfile(resource_path('Data/Shops/train_2.csv'), upload / 'train_2.csv')
    shutil.copyfile(NOTEBOOK, upload / 'original_solution.ipynb')
    assert hashlib.sha256((upload / 'original_solution.ipynb').read_bytes()).hexdigest() == NOTEBOOK_SHA256
    recipe = load_recipe()
    compact = get_fold(recipe, 2025, 2)
    train, valid, ytrain, yvalid, _ = matrices(recipe, compact, 'xgboost_log_diff')
    import pandas as pd
    import numpy as np
    expected = {name: fingerprint(frame) for name, frame in {
        'train': train, 'holdout': valid, 'ytrain': pd.DataFrame({'target': ytrain}),
        'yholdout': pd.DataFrame({'target': yvalid})}.items()}
    (upload / 'expected_validation_features.json').write_text(json.dumps(expected, indent=2))
    inputs = {name: hashlib.sha256((upload / name).read_bytes()).hexdigest()
              for name in ['train_2.csv', 'original_solution.ipynb', 'expected_validation_features.json']}
    manifest = dict(input_hashes=inputs, validation='2025-02', forecast='2025-03',
                    source_docker_image_version_id=31401, scope='XGBoost only; original features and parameters')
    (upload / 'input_manifest.json').write_text(json.dumps(manifest, indent=2))
    dataset_id = owner + '/x5-xgboost-reproduction-20261009'
    dataset_metadata = dict(id=dataset_id, title='X5 XGBoost Reproduction 20261009',
                            licenses=[{'name': 'other'}],
                            description='Private inputs for authorized reproduction of the provided X5 notebook. No redistribution license is granted.')
    (upload / 'dataset-metadata.json').write_text(json.dumps(dataset_metadata, indent=2))
    installation = """import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'xgboost==3.4.1', 'pandas==2.2.3', 'numpy==2.3.5', 'pyarrow==26.0.0', 'holidays==0.106'])
""".replace('xgboost==3.4.1', 'xgboost==' + args.xgboost_version)
    def cell(kind, source):
        value = dict(cell_type=kind, metadata={}, source=source)
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        return value
    # Kaggle's notebook kernel may already have imported NumPy before pip replaces
    # it. A fresh child avoids mixing the old loaded extension with the new files.
    effective = {name: fingerprint(frame.astype('float32').where(np.isfinite(frame), np.nan))
                 for name, frame in {'ytrain': pd.DataFrame({'target': ytrain}),
                                     'yholdout': pd.DataFrame({'target': yvalid})}.items()}
    runner_source = (HERE / 'gpu_runner.py').read_text(encoding='utf-8').replace(
        'EXPECTED_EFFECTIVE_TARGETS = None', f'EXPECTED_EFFECTIVE_TARGETS = {effective!r}')
    execution = "from pathlib import Path\n" + \
        "runner = Path('/kaggle/working/gpu_run.py')\n" + \
        f"runner.write_text({runner_source!r}, encoding='utf-8')\n" + \
        "subprocess.check_call([sys.executable, '-u', str(runner)])\n"
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, cells=[
        cell('markdown', '# X5: original XGBoost on CUDA\nPrivate reproduction; validation and final forecast only. No competition submission.'),
        cell('code', installation), cell('code', execution)])
    (kernel / 'xgboost_gpu.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    metadata = dict(id=owner + '/x5-xgboost-gpu-reproduction-20261009',
                    title='X5 XGBoost GPU Reproduction 20261009', code_file='xgboost_gpu.ipynb',
                    language='python', kernel_type='notebook', is_private=True,
                    enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4',
                    dataset_sources=[dataset_id], competition_sources=[], kernel_sources=[], model_sources=[])
    (kernel / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    (HERE / 'dispatch_manifest.json').write_text(json.dumps(dict(dataset=dataset_id, kernel=metadata['id'],
        private=True, inputs=manifest, xgboost_version=args.xgboost_version,
        dependencies=['kaggle==2.2.4', 'kagglesdk==0.1.37']), indent=2))
    print(json.dumps(dict(dataset=dataset_id, kernel=metadata['id'],
                          upload_bytes=sum(p.stat().st_size for p in upload.iterdir())), indent=2))


if __name__ == '__main__':
    main()
