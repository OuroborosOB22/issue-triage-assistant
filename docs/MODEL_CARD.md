# Model card

No real model has been released. This card describes what a release must state.

| Field | Value |
|---|---|
| Task | `maven_upgrade_build_failure` |
| Input | feature schema 1.0.0, metadata only |
| Scope | Maven, version-only changes of one dependency |
| Output | failure risk score between 0 and 1, not a calibrated probability |
| Training data | verified passes and failures, projects separated from the test projects |
| Evaluation | development partition for choice, test partition once, undefined metrics shown as undefined |
| Known limits | BUMP failures come from bot-updated projects, passes are mined and need the same verification, features are weak by design |
| Misuse | the score advises and never replaces the build; a low score is not a safety claim |

The folder `models/fixture` is a demonstration model on made-up rows. Its manifest has `is_fixture: true`.
