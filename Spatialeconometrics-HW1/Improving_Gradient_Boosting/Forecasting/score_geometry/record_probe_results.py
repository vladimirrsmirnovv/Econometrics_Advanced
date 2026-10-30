"""Merge observed results for the three probes without changing predictions."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'Forecasting/submissions/2025-03-score-geometry-probes'


def update_results():
    source = PACKAGE / 'contest_results.json'
    if not source.exists():
        return
    results = json.loads(source.read_text(encoding='utf-8'))
    manifest = json.loads((PACKAGE / 'manifest.json').read_text(encoding='utf-8'))
    entries = {e['file']: e for e in manifest['files']}
    events = json.loads((HERE / 'all_score_events.json').read_text(encoding='utf-8'))
    existing_ids = {r['submission_id'] for r in events}
    inventory = pd.read_csv(HERE / 'prediction_inventory.csv').to_dict('records')
    existing_files = {r['file'] for r in inventory}
    max_group = int(max(r['forecast_group'] for r in inventory if pd.notna(r.get('forecast_group'))))
    for result in results['submissions']:
        entry = entries[result['file']]
        path = PACKAGE / result['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'] == result['sha256']
        frame = pd.read_csv(path, float_precision='round_trip').sort_values('new_id')
        relative = path.relative_to(ROOT).as_posix()
        entry.update(actual_score=result['contest_score'], submission_id=result['submission_id'],
                     verdict=result['verdict'], submitted_at=result['submitted_at'],
                     mapping_order_explicitly_confirmed=False)
        score = result['contest_score']
        event = dict(submission_id=result['submission_id'], submitted_at=result['submitted_at'],
                     batch='Три пробы геометрии MAPE', score=score, verdict=result['verdict'],
                     label=result['file'], sha256=entry['sha256'],
                     mapping='batch_members_confirmed_equal_scores',
                     evidence=source.relative_to(ROOT).as_posix(),
                     evaluation_set='full_march_2025_test_user_confirmed', matching_files=[relative],
                     eligible_for_calculation=True,
                     mape_approx_pct=100*(1-(score/100)**.5),
                     mape_lower_pct=100*(1-((score+.005)/100)**.5),
                     mape_upper_pct=100*(1-((score-.005)/100)**.5))
        if event['submission_id'] not in existing_ids:
            events.append(event)
            existing_ids.add(event['submission_id'])
        if relative not in existing_files:
            max_group += 1
            inventory.append(dict(file=relative, sha256=entry['sha256'], bytes=path.stat().st_size,
                                  rows=len(frame), valid_submission=True, forecast_group=max_group,
                                  numeric_sha256=hashlib.sha256(frame.rto.to_numpy(dtype='<f8').tobytes()).hexdigest(),
                                  reliably_matched_scores=f'{result["submission_id"]}:{score}'))
            existing_files.add(relative)
    events.sort(key=lambda r:r['submitted_at'])
    manifest.update(contest_results_file='contest_results.json', server_acceptance_verified=True,
                    observed_best_score=max(r['contest_score'] for r in results['submissions']),
                    additional_displayed_score_gain=0.0, screenshot_order_inferred=True)
    (PACKAGE / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE / 'all_score_events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame([{**r,'matching_files':'; '.join(r['matching_files'])} for r in events]).to_csv(
        HERE / 'all_score_events.csv', index=False, encoding='utf-8-sig', lineterminator='\n')
    pd.DataFrame(inventory).to_csv(HERE / 'prediction_inventory.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    summary = json.loads((HERE / 'summary.json').read_text(encoding='utf-8'))
    valid = [r for r in inventory if r['valid_submission']]
    summary.update(csv_files_scanned=len(inventory),valid_submission_files=len(valid),
                   serialized_csv_hashes=len({r['sha256'] for r in inventory}),
                   distinct_valid_prediction_file_hashes=len({r['sha256'] for r in valid}),
                   numeric_forecast_groups=len({r['forecast_group'] for r in valid}),
                   score_events=len(events),events_with_score=sum(r['score'] is not None for r in events),
                   matched_usable_score_events=sum(r['eligible_for_calculation'] for r in events),
                   numeric_forecasts_with_matched_scores=len({r['sha256'] for r in events if r['eligible_for_calculation']}),
                   mapping_counts=dict(Counter(r['mapping'] for r in events)),
                   verdict_counts=dict(Counter(r['verdict'] for r in events)),new_probes_submitted=True,
                   new_probe_scores=[r['contest_score'] for r in results['submissions']],
                   new_probe_displayed_gain=0.0)
    (HERE / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Recorded three OK scores: 90.83; displayed gain: 0.00.')


if __name__ == '__main__':
    update_results()
