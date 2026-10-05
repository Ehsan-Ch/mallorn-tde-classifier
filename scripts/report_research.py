"""Render aggregate local experiment evidence without per-object private data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(cycle):
    source = ROOT/f'artifacts/cycle{cycle}/comparison.json'
    if not source.exists():
        return
    record = json.loads(source.read_text())
    folder = ROOT/f'reports/cycle{cycle}_20261004'
    folder.mkdir(exist_ok=True)
    names = {
        'reference': 'Retained cycle-2 reference',
        'blend_lgb_physics_aux': 'Reference + physics three-class LightGBM',
        'blend_lgb_physical_hierarchy': 'Reference + hierarchical LightGBM',
        'blend_lgb_complete_binary': 'Reference + complete binary LightGBM',
        'blend_fine5': 'Reference + five-class LightGBM',
        'blend_hard_negative_binary': 'Reference + weighted confusing negatives',
        'lgb_ensemble': 'New LightGBM ensemble',
        'blend_lgb_ensemble': 'Reference + new LightGBM ensemble',
    }
    ref = record['recipes']['reference']
    best = max(record['recipes'], key=lambda k: record['recipes'][k]['combined']['f1'])
    chosen = record['chosen']
    text = [f'# Cycle {cycle} — local development results', '',
        f'**Selected by the frozen rule: {names[chosen]}.** Highest pooled development F1: '
        f'**{record["recipes"][best]["combined"]["f1"]:.6f}** ({names[best]}).', '',
        'These scores use all 3,043 labeled objects (148 TDEs), two stratified five-fold '
        'repetitions, and the mean of each object\'s held-out probabilities. Thresholds '
        'are tuned using 500 stratified bootstraps. This is repeated development '
        'selection on historically reused labels, not independent test evidence.', '',
        '| Recipe | F1 | Precision | Recall | AP | Repeat 1 F1 | Repeat 2 F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, result in record['recipes'].items():
        m = result['combined']
        reps = result['repeat_scores']
        text.append(f'| {names[name]} | {m["f1"]:.6f} | {m["precision"]:.6f} | '
                    f'{m["recall"]:.6f} | {m["average_precision"]:.6f} | '
                    f'{reps[0]["f1"]:.6f} | {reps[1]["f1"]:.6f} |')
    text += ['', '## Selection and uncertainty', '',
        'An alternative must gain at least 0.005 pooled F1, avoid any AP decrease, '
        'and avoid lowering either repetition\'s fixed-threshold F1 by more than 0.005. '
        'The rule was written before this cycle\'s new fits.']
    for name, result in record['recipes'].items():
        if name == 'reference':
            continue
        m = result['combined']
        failures = []
        if m['f1'] < ref['combined']['f1']+.005:
            failures.append('F1 gain below 0.005')
        if m['average_precision'] < ref['combined']['average_precision']:
            failures.append('AP decreased')
        if any(a['f1'] < b['f1']-.005 for a,b in zip(result['repeat_scores'], ref['repeat_scores'])):
            failures.append('repetition stability failed')
        lo, hi = result['paired_f1_difference_95pct_descriptive']
        text.append(f'\n- {names[name]}: {"; ".join(failures) if failures else "eligible"}. '
                    f'Descriptive paired-bootstrap F1 difference interval: [{lo:.4f}, {hi:.4f}].')
    text += ['', 'The intervals keep fitted models and thresholds fixed. They do not account '
        'for historical selection, threshold tuning uncertainty, or unknown shared '
        'simulation sources. They are not proof of a population-level improvement.', '',
        '## Verification and continuation', '',
        f'All {len(record["model_replay_max_errors"])} saved fold models were reloaded and replayed; '
        f'maximum probability difference: **{max(record["model_replay_max_errors"].values()):.1g}**. '
        'Every checkpoint verifies its manifest and exact fitting/validation IDs. '
        'Each object is held out exactly once in each repetition. SpecType is used '
        'only as auxiliary training supervision or training sample weights, never as a feature.', '',
        'See [restart instructions](../../docs/RESUME_RESEARCH.md). The worker resumes '
        'completed folds without retraining them and rejects duplicate workers. '
        'Eight checkpoint tests cover failed atomic writes, stale configuration, '
        'invalid fold records, concurrency and lock release after a killed worker.', '',
        '**No GitHub or Kaggle upload was made.** Verified private F1 remains **0.6648**, '
        'versus the top private benchmark **0.6824**. New local development scores '
        'cannot establish that the top-ranked entry has been surpassed.']
    (folder/'RESULTS.md').write_text('\n'.join(text)+'\n')
    (folder/'comparison.json').write_bytes(source.read_bytes())
    (folder/'start_receipt.json').write_bytes((source.parent/'start_receipt.json').read_bytes())


if __name__ == '__main__':
    for number in (3, 4):
        render(number)
