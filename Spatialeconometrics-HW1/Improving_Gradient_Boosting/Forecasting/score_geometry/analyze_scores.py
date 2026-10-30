"""Audit saved March submissions and assess metric-based blend reconstruction.

Uses saved predictions only. No training, external upload or target recovery.
Run with the bundled Python (NumPy and pandas).
"""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORECASTING = ROOT / 'Forecasting'
SUBMISSIONS = FORECASTING / 'submissions'
MODELS = ['catboost_log_level', 'lightgbm_log_diff', 'lightgbm_log_level', 'xgboost_log_diff']
FOLDS = ['2024-08', '2024-11', '2024-12', '2025-01', '2025-02']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def table(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def loss_from_score(score):
    if not 0 < score <= 100:
        raise ValueError('Inverse MAPE requires a positive score; score zero means MAPE >= 100%.')
    return 100 * (1 - math.sqrt(score / 100))


def score_from_loss(loss):
    return 100 * (1 - min(max(loss, 0), 100) / 100) ** 2


def score_interval(score):
    return loss_from_score(min(100, score + .005)), loss_from_score(max(1e-15, score - .005))


def mape(truth, prediction):
    assert np.all(truth > 0)
    return float(100 * np.mean(np.abs((prediction - truth) / truth)))


def optimal_mape_weight(truth, base, probe):
    """Exact minimizer on [0,1] when the actual targets ARE available."""
    delta = probe - base
    keep = np.abs(delta) > 1e-12
    crossings = (truth[keep] - base[keep]) / delta[keep]
    weights = np.abs(delta[keep]) / truth[keep]
    if not weights.size:
        return 0.
    order = np.argsort(crossings)
    index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)
    return float(np.clip(crossings[order[index]], 0, 1))


def rmse_weight_from_two_scores(base_rmse, probe_rmse, correction):
    curvature = float(np.mean(correction ** 2))
    if curvature <= 0:
        return None
    return (base_rmse ** 2 - probe_rmse ** 2 + curvature) / (2 * curvature)


def reconstruct_squared_error_optimum(base_rmse, probe_rmse, weights, corrections):
    """The screenshot's multi-correction method, ONLY for RMSE in the same scale.

    For RMSLE all vectors must already be log1p-transformed. MAPE scores cannot
    be passed here as a mathematically equivalent replacement for RMSLE.
    """
    gram = corrections.T @ corrections / len(corrections)
    rhs = (np.asarray(probe_rmse)**2-base_rmse**2-np.einsum('ij,jk,ik->i',weights,gram,weights))/2
    linear, _, rank, _ = np.linalg.lstsq(weights,rhs,rcond=None)
    if rank < corrections.shape[1]:
        raise ValueError('The probe coefficient matrix does not identify all linear terms.')
    if np.linalg.matrix_rank(gram) < len(linear):
        raise ValueError('The correction Gram matrix does not give a unique optimum.')
    optimum = -np.linalg.solve(gram,linear)
    return optimum, linear, gram


def blends(components):
    """Same operations as Forecasting/run_validation.py:48; checked against CSVs."""
    matrix = np.column_stack([components[m] for m in MODELS])
    arithmetic = matrix.mean(axis=1)
    logarithmic = np.expm1(np.log1p(np.maximum(matrix, 0)).mean(axis=1))
    gate = lambda a: np.where(a.max(axis=1) / np.maximum(a.min(axis=1), 1e-9) > 1.08,
                              np.median(a, axis=1), a.mean(axis=1))
    gated = gate(matrix)
    no_ld = gate(matrix[:, [0, 2, 3]])
    meta = np.expm1(np.log1p(np.maximum(np.column_stack([arithmetic, logarithmic, gated, no_ld]), 0)).mean(axis=1))
    return dict(equal=arithmetic, log_average=logarithmic, gated_1_08_all=gated,
                gated_1_08_no_weak=no_ld, final_meta_logmean=meta)


def inventory():
    expected = pd.read_csv(SUBMISSIONS / '2025-03/baseline/test_catboost_log_level.csv').sort_values('new_id').new_id.to_numpy()
    entries, by_sha, arrays, groups = [], defaultdict(list), {}, []
    for path in sorted(SUBMISSIONS.rglob('*.csv')):
        if '2025-03-score-geometry-probes' in path.parts:
            continue  # Audit the historical files, not this builder's own outputs.
        record = dict(file=rel(path), sha256=sha(path), bytes=path.stat().st_size)
        try:
            frame = pd.read_csv(path, float_precision='round_trip')
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
            record.update(rows=None, valid_submission=False, parse_error=str(exc))
            entries.append(record)
            by_sha[record['sha256']].append(rel(path))
            continue
        record['rows'] = len(frame)
        valid = list(frame.columns) == ['new_id', 'rto'] and len(frame) == len(expected) and frame.new_id.is_unique
        if valid:
            frame = frame.sort_values('new_id')
            valid = np.array_equal(expected, frame.new_id.to_numpy()) and np.isfinite(frame.rto).all() and (frame.rto > 0).all()
        record['valid_submission'] = bool(valid)
        if valid:
            array = frame.rto.to_numpy()
            record['numeric_sha256'] = hashlib.sha256(np.asarray(array, dtype='<f8').tobytes()).hexdigest()
            group = next((i for i, prior in enumerate(groups) if np.allclose(prior, array, rtol=1e-12, atol=1e-6)), None)
            if group is None:
                group = len(groups)
                groups.append(array)
            record['forecast_group'] = group + 1
            arrays[rel(path)] = array
        entries.append(record)
        by_sha[record['sha256']].append(rel(path))
    return entries, by_sha, arrays, expected


def events(by_sha):
    rows = []
    source = FORECASTING / 'report/tables/submission_registry.csv'
    for row in csv.DictReader(source.open(encoding='utf-8', newline='')):
        batch = row['batch']
        mapping = 'confirmed_order' if batch in ['Модели и первые ансамбли', 'Семь смесей'] else 'inferred_order'
        if batch == 'Пять финальных смесей':
            mapping = 'batch_members_confirmed_equal_scores'
        rows.append(dict(submission_id=int(row['submission_id']), submitted_at=row['submitted_at'],
                         batch=batch, score=float(row['report_score']), verdict='OK',
                         label=row.get('label') or ' / '.join(filter(None, [row.get('composition_label'), row.get('rule_label'), row.get('file')])),
                         sha256=row['sha256'], mapping=mapping, evidence=rel(source)))
    # First before/after batch: ordering explicitly provided by the user.
    first = SUBMISSIONS / '2025-03'
    first_expected = {e['file']: e['sha256'] for e in json.loads((first/'submission_manifest.json').read_text(encoding='utf-8'))['files']}
    before = [(167812607, '22:15:04', 90.55), (167812856, '22:19:12', 90.32),
              (167812879, '22:19:28', 90.57), (167812903, '22:19:46', 90.45)]
    after = [(167812841, '22:18:57', 90.52), (167812869, '22:19:19', 90.31),
             (167812891, '22:19:37', 90.46), (167812913, '22:19:53', 90.46)]
    for variant, outcomes in [('baseline', before), ('spatial', after)]:
        for model, (identity, time, score) in zip(MODELS, outcomes):
            path = first / variant / f'test_{model}.csv'
            rows.append(dict(submission_id=identity, submitted_at=f'2026-10-09 {time}', batch='Первая партия', score=score,
                             verdict='OK', label=f'{model}: {variant}', sha256=first_expected[path.relative_to(first).as_posix()], mapping='confirmed_order',
                             evidence='Forecasting/research_stage2/leaderboard_provenance.json'))
    for identity, time, score, file in [(167811520, '21:57:43', 90.78, 'ensemble/reference_rules/spatial_equal.csv'),
                                       (167811533, '21:57:54', 90.79, 'ensemble/spatial_gated_1_08_all.csv')]:
        rows.append(dict(submission_id=identity, submitted_at=f'2026-10-09 {time}', batch='Первая партия', score=score,
                         verdict='OK', label=file, sha256=sha(first / file), mapping='historical_provenance',
                         evidence='Forecasting/research_stage2/leaderboard_provenance.json'))
    # Four failed early-stopping models: the user said the mapping was only approximate.
    stage = SUBMISSIONS / '2025-03-stage2'
    for identity, time, score, model in [(167894371, '21:19:03', 76.99, 'xgboost_log_diff'),
                                       (167894581, '21:22:14', 89.68, 'catboost_log_level'),
                                       (167894592, '21:22:23', 83.24, 'lightgbm_log_diff'),
                                       (167894602, '21:22:32', 89.92, 'lightgbm_log_level')]:
        path = stage / 'individual' / f'test_{model}.csv'
        rows.append(dict(submission_id=identity, submitted_at=f'2026-10-10 {time}', batch='Ранняя остановка', score=score,
                         verdict='OK', label=f'{model}: предварительная привязка', sha256=sha(path), mapping='tentative_excluded',
                         evidence='User screenshot 2c948468; approximate-order reply'))
    for identity, time, score in [(167894615, '21:22:42', 90.79), (167894625, '21:22:51', 90.78),
                                 (167894632, '21:22:58', 90.78), (167894646, '21:23:09', None),
                                 (167894661, '21:23:22', 90.77), (167894674, '21:23:30', 90.79),
                                 (167894688, '21:23:39', 90.78), (167894702, '21:23:47', 90.79),
                                 (167894713, '21:23:53', 90.77), (167894718, '21:23:59', None)]:
        rows.append(dict(submission_id=identity, submitted_at=f'2026-10-10 {time}', batch='Ранняя остановка: ансамбли',
                         score=score, verdict='OK' if score is not None else 'CE', label='Файл не установлен',
                         sha256='', mapping='unresolved_excluded', evidence='User screenshot 7026cb02'))
    format_rows = [(167903411, '00:28:50', 'CE', None), (167903415, '00:29:00', 'CE', None),
                   (167903416, '00:29:06', 'CE', None), (167903418, '00:29:12', 'CE', None),
                   (167903457, '00:31:12', 'CE', None), (167903459, '00:31:20', 'CE', None),
                   (167903463, '00:31:25', 'CE', None), (167903464, '00:31:31', 'CE', None),
                   (167903488, '00:32:19', 'OK', 90.79), (167903492, '00:32:25', 'OK', 90.78),
                   (167903495, '00:32:30', 'OK', 90.78), (167903503, '00:33:08', 'CE', None),
                   (167903506, '00:33:16', 'CE', None), (167903511, '00:33:21', 'FINAL_VERDICT_NOT_SHOWN', None),
                   (167903518, '00:33:27', 'CE', None)]
    for identity, time, verdict, score in format_rows:
        rows.append(dict(submission_id=identity, submitted_at=f'2026-10-11 {time}', batch='До исправления имени файла',
                         score=score, verdict=verdict, label='Файл не установлен', sha256='',
                         mapping='unresolved_excluded', evidence='User screenshots a69572e5 / 8fbdae61 / aea7e50f'))
    assert len({r['submission_id'] for r in rows}) == len(rows)
    for row in rows:
        row['evaluation_set'] = 'full_march_2025_test_user_confirmed'
        row['matching_files'] = by_sha.get(row['sha256'], [])
        row['eligible_for_calculation'] = row['score'] is not None and bool(row['matching_files']) and 'excluded' not in row['mapping']
        if row['score'] is not None:
            row['mape_approx_pct'] = loss_from_score(row['score'])
            row['mape_lower_pct'], row['mape_upper_pct'] = score_interval(row['score'])
        if row['sha256']:
            assert row['matching_files'], row
    return sorted(rows, key=lambda row: row['submitted_at'])


def convex_bounds(t, points):
    """Conservative bounds from convexity and displayed-score rounding only."""
    lower, upper = 0., math.inf
    for x, score in points:
        if abs(t - x) < 1e-12:
            lo, hi = score_interval(score)
            lower, upper = max(lower, lo), min(upper, hi)
    for ia, (a, sa) in enumerate(points):
        for b, sb in points[ia + 1:]:
            alo, ahi = score_interval(sa)
            blo, bhi = score_interval(sb)
            if a <= t <= b:
                upper = min(upper, ((b-t)*ahi + (t-a)*bhi)/(b-a))
            elif t < a:
                lower = max(lower, alo+(t-a)*(bhi-alo)/(b-a))
            else:
                lower = max(lower, blo+(t-b)*(blo-ahi)/(b-a))
    return dict(mape_lower_pct=lower, mape_upper_pct=upper,
                score_lower=score_from_loss(upper), score_upper=score_from_loss(lower))


def metric_diagnostics():
    p0, p1 = np.ones(3), np.array([2., 3., 4.])
    truths = [1/np.array([.75, .45, .45]), 1/np.array([.65, .65, .35])]
    counter = [dict(targets=y.tolist(), mape_at_0=mape(y, p0), mape_at_1=mape(y, p1),
                    mape_at_half=mape(y, (p0+p1)/2), optimal_weight=optimal_mape_weight(y, p0, p1)) for y in truths]
    np.testing.assert_allclose([counter[0]['mape_at_0'], counter[0]['mape_at_1']],
                               [counter[1]['mape_at_0'], counter[1]['mape_at_1']], atol=1e-12)
    z0, z1, zy = np.log1p(p0), np.log1p(p1), np.log1p(truths[0])
    correction = z1-z0
    q0, q1 = np.sqrt(np.mean((z0-zy)**2)), np.sqrt(np.mean((z1-zy)**2))
    recovered = rmse_weight_from_two_scores(q0, q1, correction)
    direct = -np.mean(correction*(z0-zy))/np.mean(correction**2)
    assert abs(recovered-direct) < 1e-12
    corrections=np.column_stack([correction,np.array([.08,-.04,.12])])
    weights=np.array([[1.,0.],[0.,1.],[.5,.5],[.2,.7]])
    rmse=np.array([np.sqrt(np.mean((z0+corrections@k-zy)**2)) for k in weights])
    optimum,linear,gram=reconstruct_squared_error_optimum(q0,rmse,weights,corrections)
    true_linear=corrections.T@(z0-zy)/len(z0)
    direct_optimum=np.linalg.lstsq(corrections,zy-z0,rcond=None)[0]
    np.testing.assert_allclose(linear,true_linear,atol=1e-12)
    np.testing.assert_allclose(optimum,direct_optimum,atol=1e-12)
    return dict(counterexample_same_predictions_and_endpoint_MAPE=counter,
                exact_RMSLE_formula_check=dict(recovered_weight=recovered, direct_weight=float(direct), difference=abs(recovered-direct)),
                exact_RMSLE_system_check=dict(correction_count=2,probe_count=4,recovered_weights=optimum.tolist(),
                 direct_weights=direct_optimum.tolist(),max_weight_difference=float(np.max(np.abs(optimum-direct_optimum)))))


def local_diagnostics():
    rows, curves = [], []
    for fold in FOLDS:
        base, wx, truth, ids = {}, {}, None, None
        for model in MODELS:
            configs = ['baseline'] if model == 'catboost_log_level' else ['baseline', 'wx_knn6' if model == 'lightgbm_log_level' else 'wx_knn12']
            for config in configs:
                path = FORECASTING / 'course_only/results' / fold / model / f'{config}.csv'
                frame = pd.read_csv(path, float_precision='round_trip').sort_values('new_id')
                if ids is None:
                    ids, truth = frame.new_id.to_numpy(), frame.pto.to_numpy()
                np.testing.assert_array_equal(ids, frame.new_id)
                np.testing.assert_array_equal(truth, frame.pto)
                (base if config == 'baseline' else wx)[model] = frame.prediction.to_numpy()
        full = dict(base, **wx)
        blend_full = blends(full)
        u = (blend_full['final_meta_logmean']+blend_full['gated_1_08_all'])/2
        c = blends(dict(base, xgboost_log_diff=wx['xgboost_log_diff'], lightgbm_log_diff=wx['lightgbm_log_diff']))['gated_1_08_all']
        d = blends(dict(base, xgboost_log_diff=wx['xgboost_log_diff'], lightgbm_log_level=wx['lightgbm_log_level']))['gated_1_08_no_weak']
        for direction, probe in [('C', c), ('D', d)]:
            points = [0., 1/3, 1.]
            observed = [mape(truth, u+t*(probe-u)) for t in points]
            coef = np.polyfit(points, observed, 2)
            kq = float(np.clip(-coef[1]/(2*coef[0]), 0, 1)) if coef[0] > 0 else float(np.argmin([observed[0],observed[-1]]))
            exact = optimal_mape_weight(truth, u, probe)
            actual = mape(truth, u+kq*(probe-u))
            dense = np.linspace(0, 1, 201)
            actual_curve = [mape(truth, u+t*(probe-u)) for t in dense]
            assert mape(truth, u+exact*(probe-u)) <= min(actual_curve)+1e-10
            rows.append(dict(fold=fold, direction=direction, mape_U=observed[0], mape_at_third=observed[1], mape_endpoint=observed[2],
                             quadratic_weight=kq, exact_MAPE_weight_with_known_targets=exact,
                             quadratic_predicted_MAPE=float(np.polyval(coef, kq)), actual_MAPE_at_quadratic_weight=actual,
                             exact_minimum_MAPE=mape(truth, u+exact*(probe-u)),
                             maximum_curve_fit_error_pp=float(np.max(np.abs(np.polyval(coef,dense)-actual_curve)))))
            curves.append(dict(fold=fold, direction=direction, weights=dense.tolist(), mape=actual_curve,
                               quadratic=np.polyval(coef,dense).tolist()))
        for name, pred in [('U',u), ('probe_C',(5/6)*u+c/6), ('probe_D',(5/6)*u+d/6), ('probe_CD',(5/6)*u+(c+d)/12)]:
            rows.append(dict(fold=fold, direction=name, candidate_mape=mape(truth,pred)))
    return rows, curves


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    protected = [FORECASTING/'course_only/FINAL_DECISION.json', FORECASTING/'run_validation.py', resource_path('Paper/main.tex'),
                 FORECASTING/'report/Прогноз_РТО_финальный_отчёт.tex',
                 FORECASTING/'report/output/pdf/Прогноз_РТО_финальный_отчёт.pdf']
    protected_before = {rel(p):sha(p) for p in protected}
    entries, by_sha, arrays, ids = inventory()
    valid_by_sha = defaultdict(list)
    for e in entries:
        if e['valid_submission']:
            valid_by_sha[e['sha256']].append(e['file'])
    records = events(valid_by_sha)
    dump('all_score_events.json', records)
    table('all_score_events.csv', [{**r, 'matching_files': '; '.join(r['matching_files'])} for r in records])
    scores_by_sha = defaultdict(list)
    for row in records:
        if row['eligible_for_calculation']:
            scores_by_sha[row['sha256']].append((row['submission_id'],row['score']))
    for entry in entries:
        entry['reliably_matched_scores'] = '; '.join(f'{i}:{s}' for i,s in scores_by_sha[entry['sha256']])
    table('prediction_inventory.csv', entries)
    dump('metric_diagnostics.json', metric_diagnostics())
    local, curves = local_diagnostics()
    table('local_metric_diagnostics.csv', local)
    dump('local_curves.json', curves)

    # Exact arithmetic line identities, unlike log blends or gated model changes.
    last = SUBMISSIONS / '2025-03-course-final-five'
    last_manifest = json.loads((last / 'manifest.json').read_text(encoding='utf-8'))
    saved = {e['order']: arrays[rel(last/e['file'])] for e in last_manifest['files']}
    prior = SUBMISSIONS / '2025-03-course-final-ten'
    a = arrays[rel(prior/'04_three_wx_original_mix/test.csv')]
    b = arrays[rel(prior/'05_three_wx_gated_all/test.csv')]
    c = arrays[rel(prior/'01_winner_gated_all/test.csv')]
    d = arrays[rel(prior/'08_xgb_ll_wx_gated_no_ld/test.csv')]
    u = saved[1]
    line_checks = {'U=(A+B)/2':(u,(a+b)/2), 'E3=(2U+C)/3':(saved[3],(2*u+c)/3), 'E4=(2U+D)/3':(saved[4],(2*u+d)/3)}
    line_result = {name:float(np.max(np.abs(x-y)/np.maximum(np.abs(x),1))) for name,(x,y) in line_checks.items()}
    assert max(line_result.values()) < 1e-12
    points = [(0.,90.83),(1/3,90.83),(1.,90.82)]
    coef = np.polyfit([x for x,s in points],[loss_from_score(s) for x,s in points],2)
    vertex = float(-coef[1]/(2*coef[0]))
    sensitivity = []
    for bits in range(8):
        losses=[score_interval(s)[(bits>>i)&1] for i,(t,s) in enumerate(points)]
        co=np.polyfit([t for t,s in points],losses,2)
        w=float(np.clip(-co[1]/(2*co[0]),0,1)) if co[0]>1e-14 else float(np.argmin([losses[0],losses[-1]]))
        sensitivity.append(dict(weight=w,predicted_score=score_from_loss(float(np.polyval(co,w))),convex_quadratic=bool(co[0]>0)))
    dump('weight_analysis.json',dict(exact_optimum_identifiable=False, reason='MAPE is not a quadratic loss; true denominators and residual sign changes are unknown.',
         exact_line_checks=line_result, known_points=points,endpoint_score_mapping='inferred order of the ten-file batch; final-five equal scores are permutation-independent',
         quadratic_approximation_only=dict(vertex=vertex,predicted_score=score_from_loss(float(np.polyval(coef,vertex))),
         gain_in_displayed_score=score_from_loss(float(np.polyval(coef,vertex)))-90.83),rounding_sensitivity_corner_cases=sensitivity,
         bounds_at_sixth=convex_bounds(1/6,points),original_reports_and_training_decisions_unchanged=True))

    # New probes stay within the convex hull of already scored 90.83 predictions.
    probes=[('01_half_U_half_E3',{1:.5,3:.5}),('02_half_U_half_E4',{1:.5,4:.5}),('03_half_U_quarter_E3_E4',{1:.5,3:.25,4:.25})]
    out=SUBMISSIONS/'2025-03-score-geometry-probes'
    out.mkdir(exist_ok=True)
    files=[]
    for name,weights in probes:
        prediction=sum(saved[k]*w for k,w in weights.items())
        assert sum(weights.values())==1 and np.isfinite(prediction).all() and (prediction>0).all()
        duplicates=[p for p,v in arrays.items() if np.allclose(v,prediction,rtol=1e-12,atol=1e-6)]
        assert not duplicates, duplicates
        path=out/name/'test.csv';path.parent.mkdir(exist_ok=True)
        pd.DataFrame(dict(new_id=ids,rto=prediction)).to_csv(path,index=False,encoding='utf-8',lineterminator='\n')
        frame=pd.read_csv(path,float_precision='round_trip')
        np.testing.assert_array_equal(ids,frame.new_id)
        np.testing.assert_array_equal(prediction,frame.rto)
        assert path.stat().st_size<1_000_000
        files.append(dict(file=path.relative_to(out).as_posix(),sha256=sha(path),weights_over_final_five=weights,
                          rows=len(ids),bytes=path.stat().st_size,actual_score=None,
                          mathematical_rounded_score_lower_bound=90.83,exact_score_prediction=None,
                          lower_bound_assumptions=['same full test and MAPE scoring','the five final files are the scored five-file batch','arithmetic averaging with nonnegative weights']))
    sources={rel(last/e['file']):sha(last/e['file']) for e in last_manifest['files']}
    manifest=dict(no_models_retrained=True,no_new_features=True,no_new_spatial_methods=True,automatic_submissions=False,
                  based_on_full_test_feedback=True,required_upload_filename='test.csv',sources=sources,files=files,
                  method='arithmetic convex mixtures of scored final-five predictions; no claimed RMSLE vertex for MAPE')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(out/'score_geometry_probes.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in files:archive.write(out/file['file'],file['file'])
        archive.write(out/'manifest.json','manifest.json')
        if (out/'README.md').exists():archive.write(out/'README.md','README.md')
    with zipfile.ZipFile(out/'score_geometry_probes.zip') as archive:assert archive.testzip() is None
    for entry in entries:assert sha(ROOT/entry['file'])==entry['sha256']
    for path,digest in protected_before.items():assert sha(ROOT/path)==digest
    dump('protected_files.json',protected_before)
    eligible = [r for r in records if r['eligible_for_calculation']]
    groups_by_file = {e['file']:e['forecast_group'] for e in entries if e['valid_submission']}
    group_scores = defaultdict(set)
    for r in eligible:
        group_scores[groups_by_file[r['matching_files'][0]]].add(r['score'])
    assert all(len(values)==1 for values in group_scores.values()),group_scores
    summary=dict(csv_files_scanned=len(entries),valid_submission_files=sum(e['valid_submission'] for e in entries),
                 serialized_csv_hashes=len(by_sha),distinct_valid_prediction_file_hashes=len(valid_by_sha),
                 numeric_forecast_groups=len({e.get('forecast_group') for e in entries if e['valid_submission']}),
                 score_events=len(records),events_with_score=sum(r['score'] is not None for r in records),
                 matched_usable_score_events=sum(r['eligible_for_calculation'] for r in records),
                 mapping_counts=dict(Counter(r['mapping'] for r in records)),verdict_counts=dict(Counter(r['verdict'] for r in records)),
                 inputs_unchanged=True,model_fits_run=0,new_probe_count=3,new_probes_submitted=False,
                 numeric_forecasts_with_matched_scores=len(group_scores),duplicate_forecast_scores_consistent=True,
                 latest_full_test_scope_confirmation='User: оценка была произведена на всей тестовой выборке')
    dump('summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    # Keep later observed scores when the analysis is rebuilt.
    from record_probe_results import update_results
    update_results()


if __name__=='__main__':
    main()
