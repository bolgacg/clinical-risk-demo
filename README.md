# When should a risk model withhold its answer?

An interactive clinical risk console built for the SDU Health Informatics group (research
assistant application, job 4184). One patient at a time: the risk figure from a bootstrap
ensemble, exact per-patient attributions, a computed smallest-actionable-change answer, and a
first-class withheld state for patients outside the model's competence.

Live: https://bolgacg.github.io/clinical-risk-demo/

Real public data: the NIDDK Pima diabetes cohort (768 patients; 724 after excluding coded
missing values, stated on the page). Model: 30 bootstrap logistic regressions; attributions are
exact coefficient contributions; the counterfactual is the minimum-norm boundary crossing
restricted to clinically modifiable measurements. Calibration by decile and the referral
threshold as a capacity decision are computed offline and drawn from the shipped results.

## Reproduce

```
python3 study/study.py   # seeded, numpy only, ~2 s
```

No build step, no libraries in the browser. Bolgaç Gülen, August 2026.
