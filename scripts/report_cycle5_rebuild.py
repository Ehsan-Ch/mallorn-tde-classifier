"""Compare verified rebuilt development results with historical session records."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/cycle5_recovered'
HISTORY = {
    'reference': [.684058, .677698, .699708, .672464],
    'blend_selected_complete_fine5': [.689024, .704545, .668693, .660494],
    'blend_selected_physical_aux': [.695906, .685389, .688427, .674487],
    'lgb_ensemble': [.672897, .707030, .645963, .672956],
    'blend_lgb_ensemble': [.688235, .699186, .686391, .680352],
}


def report():
    comparison = json.loads((OUT/'comparison.json').read_text())
    folds = sorted(OUT.rglob('*.joblib'))
    assert len(folds) == 20
    errors = comparison['model_replay_max_errors']
    assert len(errors) == 20 and max(errors.values()) <= 1e-12
    rows = []
    table = ['| Recipe | Historical F1 | Rebuilt F1 | Rebuilt AP | Repeat 1 F1 | Repeat 2 F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, historical in HISTORY.items():
        result = comparison['recipes'][name]
        rebuilt = [result['combined']['f1'], result['combined']['average_precision'],
                   *[r['f1'] for r in result['repeat_scores']]]
        rows.append({'recipe': name, 'historical_rounded_metrics': historical,
                     'rebuilt_metrics': rebuilt,
                     'matches_historical_six_decimal_record': all(abs(a-b) <= 5.1e-7 for a,b in zip(historical,rebuilt))})
        table.append(f'| {name} | {historical[0]:.6f} | {rebuilt[0]:.6f} | {rebuilt[1]:.6f} | {rebuilt[2]:.6f} | {rebuilt[3]:.6f} |')
    receipt = {'kind':'newly_rebuilt_models_not_original_file_recovery',
               'completed_folds':len(folds), 'retained_final_models':60,
               'replay_max_error':max(errors.values()), 'chosen':comparison['chosen'],
               'historical_comparison':rows,
               'sha256': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [OUT/'manifest.json',OUT/'comparison.json',*folds]}}
    (OUT/'verification_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    matches = sum(r['matches_historical_six_decimal_record'] for r in rows)
    text = '# Cycle-5 rebuild results\n\n'
    text += 'The reconstructed method completed all **20 fold checkpoints**, retaining **60 final LightGBM models**. All 20 saved checkpoints passed prediction replay verification (maximum absolute error '+str(receipt['replay_max_error'])+').\n\n'
    text += '\n'.join(table)+'\n\n'
    text += f'{matches} of 5 recipes match all four recorded historical metrics to six-decimal rounding. These comparisons use historical session records; the original cycle-5 model files remain unavailable, so byte-for-byte equivalence cannot be established.\n\n'
    text += f'The unchanged acceptance gates selected **{comparison["chosen"]}**. These are repeated development-selection results, not independent validation or new Kaggle scores. The existing Kaggle submission remains private F1 0.6648 / public F1 0.6825.\n\n'
    text += 'The runner resumed the four checkpoints that survived the interruption, verified their experiment and row assignments, and fitted the remaining folds. The backup includes checkpoint hashes, training feature caches, the frozen reference predictions and the source needed to resume or audit. Preserve the verified ZIP separately: local files alone do not guarantee survival across workspace resets.\n'
    destination = ROOT/'reports/CYCLE5_REBUILD_RESULTS.md'
    destination.write_text(text)
    print(json.dumps({k:v for k,v in receipt.items() if k!='sha256'},indent=2))


if __name__ == '__main__':
    report()
