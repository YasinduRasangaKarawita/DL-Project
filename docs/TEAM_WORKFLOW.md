# Four-person team workflow

## Primary ownership

| Member | Model ownership | Cross-cutting ownership |
|---|---|---|
| Member 1 | Custom CNN | Dataset acquisition, provenance, validation, and grouped splits |
| Member 2 | ResNet-50 | Training engine, checkpoint/resume support, and reproducibility |
| Member 3 | EfficientNet-B0 | Metrics, statistical comparison, efficiency benchmark, and error analysis |
| Member 4 | MobileNetV3-Large | Colab integration, model release, Streamlit app, and demo video |

Ownership means leading the work, not becoming the only person who understands it. Every pull request should be reviewed by someone from another area, and every member should be able to explain all four architectures and the complete experimental protocol during the viva.

## Weekly rhythm

- Start of week: agree on issues, acceptance criteria, and dependencies.
- During week: make small focused commits and open reviewable pull requests.
- End of week: merge tested work, update the experiment register, and record decisions.
- Before final training: freeze data/splits and code/config versions.
- Before submission: jointly audit claims against raw metrics and rehearse the viva.

## Definition of done

A task is done when the implementation, tests, documentation, and evidence are merged. A model run is done only when its config, seed, commit SHA, split hash, runtime information, checkpoint hash, histories, and predictions are preserved.

