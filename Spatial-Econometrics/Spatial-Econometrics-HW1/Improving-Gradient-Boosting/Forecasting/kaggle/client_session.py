"""Authenticated Kaggle session; credentials stay only in this process's memory.

The access token is read without echo if KAGGLE_API_TOKEN is absent. Subsequent
stdin lines are JSON commands. Neither token nor authorization headers are printed.
"""
from __future__ import annotations


def _directory_path(value):
    """Resolve directory names from preserved metadata after the folder rename."""
    from pathlib import Path as _Path
    path = _Path(value)
    if path.is_absolute():
        return path
    directory_count = len(path.parts) - bool(path.suffix)
    return _Path(*(part.replace("_", "-") if index < directory_count else part
                   for index, part in enumerate(path.parts)))


import json
import getpass
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def local_path(relative):
    path = (HERE / _directory_path(relative)).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError('Path must stay within Forecasting/kaggle')
    return path


def main():
    print('READY_FOR_CREDENTIAL', flush=True)
    token = os.environ.get('KAGGLE_API_TOKEN') or getpass.getpass('Kaggle token: ').strip()
    if not token:
        raise ValueError('Missing credential')
    os.environ['KAGGLE_API_TOKEN'] = token
    try:
        from kaggle import api
        owner = api.get_config_value('username')
        print(json.dumps({'authenticated': bool(owner), 'owner': owner}), flush=True)
        (HERE / 'account.json').write_text(json.dumps({'owner': owner}, indent=2), encoding='utf-8')
        streams={}
        for line in sys.stdin:
            try:
                command = json.loads(line)
                action = command['action']
                profile = command.get('profile', 'baseline')
                if profile == 'baseline':
                    active = HERE
                elif profile == 'research':
                    active = HERE.parent / 'research/kaggle'
                elif profile == 'forecast':
                    active = HERE.parent / 'final-forecast/kaggle'
                elif profile == 'stage2':
                    active = HERE.parent / 'research-stage2/kaggle'
                elif profile == 'stage2cpu':
                    active = HERE.parent / 'research-stage2/kaggle-cpu'
                elif profile == 'stage2final':
                    active = HERE.parent / 'research-stage2/kaggle-final'
                elif profile in {'course_cat','course_ld','course_ll','course_xgb'}:
                    active = HERE.parent / ('course-only/kaggle_'+profile.removeprefix('course_'))
                else:
                    raise ValueError('Unknown fixed workspace profile')
                if action == 'datasets':
                    result = [{'ref': d.ref} for d in api.dataset_list(mine=True, page=1)]
                elif action == 'create_dataset':
                    folder = active / 'upload'
                    metadata = json.loads((folder / 'dataset-metadata.json').read_text())
                    assert metadata['id'].startswith(owner + '/')
                    result = api.dataset_create_new(str(folder), public=False, quiet=True,
                                                    convert_to_csv=False, dir_mode='skip')
                elif action == 'dataset_version':
                    folder = active / 'upload'
                    metadata = json.loads((folder / 'dataset-metadata.json').read_text())
                    assert metadata['id'].startswith(owner + '/')
                    result = api.dataset_create_version(str(folder), version_notes='Additional authorised validation feature matrices',
                        quiet=True, convert_to_csv=False, dir_mode='skip')
                elif action == 'dataset_status':
                    result = api.dataset_status(command['ref'])
                elif action in {'push','push_cpu'}:
                    folder = active / 'kernel'
                    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
                    assert metadata['is_private'] is True
                    assert metadata['enable_gpu'] is (action=='push')
                    assert metadata['id'].startswith(owner + '/')
                    result = api.kernels_push(str(folder), timeout='14400', acc='NvidiaTeslaT4' if action=='push' else None)
                elif action == 'status':
                    result = api.kernels_status(command['ref'])
                elif action == 'logs':
                    log=api.kernels_logs(command['ref'])
                    (active/'live_log.txt').write_text(log,encoding='utf-8')
                    result={'characters':len(log),'tail':log[-5000:]}
                elif action=='stream_begin':
                    ref=command['ref']
                    if ref not in streams or streams[ref]['finished']:
                        state={'tail':deque(maxlen=150),'finished':False};streams[ref]=state
                        def follow(ref=ref,state=state,active=active):
                            try:
                                with (active/'stream_log.txt').open('w',encoding='utf-8') as output:
                                    seen=0
                                    for attempt in range(6):
                                        try:
                                            for index,item in enumerate(api.kernels_logs_stream(ref)):
                                                if index<seen:continue
                                                data=str(item.get('data','')).replace(token,'[REDACTED]')
                                                output.write(data);output.flush();state['tail'].append(data);seen=index+1
                                            break
                                        except Exception:
                                            if attempt==5:raise
                                            time.sleep(1)
                            except Exception as error:state['tail'].append(str(error).replace(token,'[REDACTED]'))
                            finally:state['finished']=True
                        threading.Thread(target=follow,daemon=True).start()
                    result={'stream_started':True}
                elif action=='stream_snapshot':
                    state=streams[command['ref']];result={'finished':state['finished'],'tail':''.join(state['tail'])[-1500:]}
                elif action == 'output':
                    destination = active / 'output'
                    destination.mkdir(exist_ok=True)
                    result = api.kernels_output(command['ref'], path=str(destination),
                                                force=True, quiet=True)
                elif action == 'quit':
                    break
                else:
                    raise ValueError('Unsupported command')
                if hasattr(result, 'to_dict'):
                    result = result.to_dict()
                if action in {'push','status','dataset_status'}:
                    active.mkdir(exist_ok=True, parents=True)
                    (active / (action + '_response.json')).write_text(json.dumps(
                        {'at_utc':datetime.now(timezone.utc).isoformat(),'result':result},
                        default=str,indent=2),encoding='utf-8')
                print(json.dumps({'action': action, 'result': result}, default=str), flush=True)
            except Exception as error:
                detail = getattr(error, 'response', None)
                detail = detail.text[:2000] if detail is not None else None
                print(json.dumps({'action': command.get('action'), 'error_type': type(error).__name__,
                                  'error': str(error).replace(token, '[REDACTED]'),
                                  'response': detail.replace(token, '[REDACTED]') if detail else None}), flush=True)
    except Exception as error:
        print(json.dumps({'error_type': type(error).__name__,
                          'error': str(error).replace(token, '[REDACTED]')}), flush=True)
        raise SystemExit(1)
    finally:
        os.environ.pop('KAGGLE_API_TOKEN', None)


if __name__ == '__main__':
    main()
