"""Keep executable source versions attached to every computation phase."""
from pathlib import Path
import ast,base64,hashlib,io,json,zipfile
HERE=Path(__file__).resolve().parent

def main():
    source=json.loads((HERE/'kaggle/kernel/stage2.ipynb').read_text())['cells'][1]['source']
    launcher=[n.value.args[0] for n in ast.parse(source).body if isinstance(n,ast.Expr) and
        isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='write_text'][0]
    launcher=ast.literal_eval(launcher)
    payload=next(ast.literal_eval(n.value) for n in ast.parse(launcher).body if isinstance(n,ast.Assign) and
        isinstance(n.targets[0],ast.Name) and n.targets[0].id=='PAYLOAD')
    manifest=json.loads((HERE/'kaggle/dispatch_manifest.json').read_text())
    destination=HERE/'sources/phase1';destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        for name in ['experiment.py','features.py','notebook_api.py','run_validation.py','plan.md']:
            content=archive.read(name);assert hashlib.sha256(content).hexdigest()==manifest['files'][name]
            path=destination/name
            if path.exists():assert path.read_bytes()==content
            else:path.write_bytes(content)
    print('PHASE1_SOURCE_ARCHIVED',hashlib.sha256((destination/'experiment.py').read_bytes()).hexdigest())

if __name__=='__main__':main()
