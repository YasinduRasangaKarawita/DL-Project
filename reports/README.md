# Assignment report workspace

Keep the report source and deliberately curated final tables/figures here. Do not copy raw or unverified experiment output.

Required report structure:

1. Introduction
2. Related Work
3. Dataset and Exploratory Data Analysis
4. Preprocessing
5. Experimental Design
6. Model Architectures
7. Results and Comparison
8. Critical Analysis
9. Conclusion
10. References

Recommended subdirectories:

```text
reports/
├── figures/
├── tables/
└── references/
```

Each included result should be traceable to a run ID, configuration, split hash, seed, Git SHA, and model checksum.

## Current drafts

| Draft | Owner | Status | Required reviewers |
|---|---|---|---|
| [`member1_report_draft.md`](member1_report_draft.md) | Member 1 | Design facts complete; experiment results intentionally pending | Members 2, 3, and 4 according to section ownership |

Supporting material:

- [`tables/member1_verified_facts.md`](tables/member1_verified_facts.md) records the source of every numerical design claim and the commands used to reproduce it.
- [`references/member1_references.bib`](references/member1_references.bib) contains verified citation metadata for the sources cited by the draft.

The Member 1 draft contains no accuracy, loss, timing, checkpoint, or test-set result. Add those claims only from accepted experiment records produced after the shared protocol is frozen.
