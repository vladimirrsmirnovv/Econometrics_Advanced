"""Change private kernel code; reuse only datasets already authorised and uploaded."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
directory=HERE/'kaggle/kernel'
metadata=json.loads((directory/'kernel-metadata.json').read_text())
owner=metadata['id'].split('/')[0]
metadata['dataset_sources']=[owner+'/x5-spatial-research-20261009',owner+'/x5-xgboost-reproduction-20261009']
source=(HERE/'kaggle_followup.py').read_text()
install="import subprocess,sys\nsubprocess.check_call([sys.executable,'-m','pip','install','-q','xgboost==3.1.3','pandas==2.2.3','numpy==2.3.5','pyarrow==26.0.0','holidays==0.106'])\n"
run="from pathlib import Path\np=Path('/kaggle/working/followup_run.py')\np.write_text("+repr(source)+",encoding='utf-8')\nsubprocess.check_call([sys.executable,'-u',str(p)])\n"
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    cells=[dict(cell_type='code',id='step'+str(i),metadata={},execution_count=None,outputs=[],source=s) for i,s in enumerate([install,run])])
(directory/'spatial_xgboost.ipynb').write_text(json.dumps(notebook,indent=1))
(directory/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Follow-up: 17 fits; existing private datasets only')
