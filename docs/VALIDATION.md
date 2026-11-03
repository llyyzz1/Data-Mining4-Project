# Packaging validation

The upload directory contains 10 unchanged Python source files: three Adult,
six Mushroom and one cross-dataset comparison script. All 10 pass syntax
parsing and match the source SHA-256 values. All copied files and the three raw
dataset hashes were checked. Original project files and source reports remained
unchanged.

Checks were run using the existing Python 3.14.6 Anaconda environment in a
separate relocated copy, outside both the original directories and this upload
directory. Recorded checks include:

- Adult's three CLI help entries and comparison CLI help.
- Full Adult preprocessing: all three train/test feature and label states
  match the packaged snapshot.
- Adult evaluation from the three saved model states: all 12 result rows match.
- Full Mushroom preprocessing: modeling CSV matches the packaged data.
- Mushroom tree/linear-model entry points and import of the tuning module.
- Fitting and evaluating the four original Mushroom baseline classifiers via
  their existing functions. The historical grid search was not repeated.
- Standalone Mushroom Apriori and Adult Apriori function execution.
- Complete matched-parameter cross-dataset ten-fold/holdout experiment:
  all eight result rows match the saved numeric metrics; 40 Mushroom fold
  scores are produced and the new temporary run reports status=complete.
- Original files and packaged copies remain unchanged after runtime checks.

Detailed check results and library versions are in VALIDATION_RESULTS.json.
Test outputs were kept outside the upload directory, preserving its original
result snapshot. Full Adult classifier retraining, historical Mushroom grid
search and a fresh online dependency installation were not repeated. This is
a validated relocation/package of existing work, not a newly tuned experiment.
