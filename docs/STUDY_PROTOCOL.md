# BreakCast study protocol

Written before any model was fitted. Changes go in the log at the bottom.

## Question

Can facts known before a candidate build predict whether changing one Maven dependency version breaks that project's build or tests, for projects the model has never seen?

## Target

One row is one version-only change: a project, a package, a from version and a to version.

| Outcome | Meaning | Used for training |
|---|---|---|
| `OBSERVED_PASS` | The baseline passed and the changed project passed the same verification profile | yes, label 0 |
| `OBSERVED_REGRESSION` | The baseline passed and the change caused a build, compile, dependency or test failure | yes, label 1 |
| `INCONCLUSIVE` | Broken baseline, missing tests, timeouts, unresolved versions, unstable results, or a policy-only rejection (licence, vulnerability) | no, kept for operations analysis |

A green CI run or a merged pull request is a candidate signal, never a label. Licence and vulnerability findings are not build failures.

## Label sources

- `bump_metadata`: a record of the BUMP benchmark (571 breaking updates, 153 projects, MIT licence). Its record says the update broke the build. It was not re-run by us, so it is marked as unverified locally. BUMP's passing pre-update build is a baseline and is never used as a pass example.
- `sentry_evidence`: an evidence bundle from a real before and after run, imported after schema validation. Only these can give `OBSERVED_PASS`.

## What is not available yet

BUMP holds no passing upgrades. Passing upgrades have to be mined and re-run (see the README for the job). Until at least the minimum counts below exist, training refuses to run. A model trained only on failures would predict "failure" for everything and prove nothing.

## Inclusion and exclusion

- Maven only. Version-only changes of one dependency.
- Excluded from training: pull requests that also repair source, disable tests or change several dependencies; rows whose versions cannot be parsed into a comparable pair; exact duplicates of project, package and version pair.

## Features (schema 1.0.0, metadata only)

Only facts available before the candidate build: version change category, size of the version jump, pre-release flags, dependency scope, where the dependency is declared, file type. Each row carries a list of missing features.

Never used as features: failure category, error text, test counts, verdicts, exit codes, project name, organisation, pull request author, pull request title, any outcome of the changed build, anything dated after the update.

Release age and API-difference features need Maven Central dates and the Java engine. They are listed as missing until collected and are not silently replaced by zero.

## Splits

- Group by project (organisation and project name, lower case). All rows of a project stay in one partition.
- Partition is chosen by a seeded hash of the group name, so it does not change when data is added: 60% train, 20% development, 20% test.
- The split is written to a manifest with a digest and checked on every run.
- Tuning and model choice use development data only. The test partition is evaluated once, by `evaluate --final`, and the result is written to the report. It is never used to promote or reject a model.

## Minimum data to train

At least 30 rows of each class and 10 projects in the training partition, and both classes present in the development partition. Below that, `train` stops with an error that names the missing counts.

## Models

1. Majority class (floor).
2. Rule: a major version jump predicts failure.
3. Logistic regression with fitted preprocessing.
4. Gradient boosting (scikit-learn histogram model, fixed seed).
5. Frozen text encoder plus a small head: pending, needs release notes or API diff text that is not collected yet.

## Metrics

PR-AUC (average precision, failure as the positive class), ROC-AUC, precision and recall for failure at the stated threshold, confusion matrix, Brier score and calibration bins, and a project-resampled interval for PR-AUC. A metric that cannot be defined (one class only, too few groups) is reported as undefined. A score is called a probability only after the calibration check is reported. The balanced benchmark prevalence is not the prevalence in real projects.

## Known biases

- BUMP failures come from projects that used automated update tools, so they are not a random sample of upgrades.
- Mined passes must come from the same projects and similar time windows, otherwise the model can learn which dataset a row came from.
- Metadata-only features are weak on purpose; if the models do not beat the major-jump rule, that is the reported result.

## Change log

- 1.0: first version, before any model run.
