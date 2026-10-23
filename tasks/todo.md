# Model building plan

Goal: off-market hidden-test MdAPE well under the 17.7% bar; list-aware under 7.3%.
Deadline: Nov 8, 11:59 pm EST.

## Steps
- [ ] 1. Switch on `FeatureModel` for both tracks, rerun `validate` + `report`, record the numbers
- [ ] 2. Make validation match the test horizon: train on sales before a cutoff, predict the next 4 months
      (test = train through May 2026, predict homes listed up to Sep 2026)
- [ ] 3. Let features see past sales: `features(homes, history)`, with point-in-time stats for training rows
      (each row only sees sales sold before it was listed)
- [ ] 4. Off-market features: market index by type/month, community x type price per sq ft,
      time-adjusted nearest comps, lot area, house_style, municipality, garage, remarks keywords
- [ ] 5. Handle the time trend: target = log(price) - market index, so trees don't flatline past May 2026
- [ ] 6. List-aware: model log(sold / list) using the off-market estimate vs list price + market heat
- [ ] 7. Error analysis by price / type / community; prediction intervals (quantile models)
- [ ] 8. Final `predict` + `check`, freeze validation numbers for the report

## Review
(after each step: `python run.py report "what changed"`, then a notebook entry in report/notebook.tex)

- Experiment 0, starter baselines (monthly walk-forward): off-market MdAPE 20.84 / MAPE 27.43,
  list-aware 7.26 / 8.09. Off-market biased +8% and squeezed toward neighbourhood prices; list price
  ~5.7% above sale.
