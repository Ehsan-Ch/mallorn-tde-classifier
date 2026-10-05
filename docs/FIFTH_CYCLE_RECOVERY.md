# Recovered cycle-5 method

The new `selected_ensemble.py` reconstructs the recorded modeling method:
two training-fold LightGBM selectors (600 trees; seeds +10000/+20000), normalized
gain averaging, 125 features pruned at absolute Pearson correlation .95, and
three 900-tree final models (seeds +0/+1000/+2000). Median filling is used only
for training-fold correlation. The model receives missing values directly.

Complete-feature five-class and physical-view three-class components use the
same fixed LightGBM parameters and auxiliary class weights as the historical
cycle. `cycle5_recovered.py` reuses cycle 3's two five-fold partitions, threshold
method, five fixed blends, checkpoint audits and unchanged acceptance gates.

This is recovered source, not an assertion that the missing original model
files were recovered or byte-for-byte identical source was restored. A rerun
writes a separate `artifacts/cycle5_recovered` namespace. It needs the local
authorized feature caches and cycle-2 reference OOF predictions, which are not
distributed in the public repository. Do not modify or fabricate old manifests.

```bash
MPLCONFIGDIR=artifacts/mpl OPENBLAS_NUM_THREADS=1 MALLORN_THREADS=4 \
  python scripts/cycle5_recovered.py --max-seconds 1500
```

The historical cycle-5 measurements are preserved with their provenance in
[the recovery note](../reports/RECOVERY_20261005.md). They remain development
selection evidence, not independent validation or Kaggle scores.
