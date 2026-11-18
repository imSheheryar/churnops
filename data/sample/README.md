# Sample batches

The CSVs in this folder are generated, not committed. Create them after the
pipeline has produced a reference sample:

```bash
python scripts/make_sample_batches.py
```

That writes two 800-row batches drawn from the held-out data:

- `baseline_batch.csv` - representative traffic, used to demonstrate a `stable`
  drift verdict.
- `drifted_batch.csv` - the same customers with widened tenure and monthly
  charges plus shifted categorical mixes, used to demonstrate a `significant`
  verdict and the alerting path.

Both files are git-ignored because they are reproducible from the dataset and
the fixed random seed.
