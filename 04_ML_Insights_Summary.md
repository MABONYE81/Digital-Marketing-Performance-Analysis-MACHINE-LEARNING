# Machine Learning Section — Key Insights (One-Pager)
**Greenside Villas | Booking Prediction & Revenue Forecasting, Jan 2025–Jun 2026**

## The most important insight isn't a model — it's a limit
With 76 usable weeks and only 9 booking events, this is a small-data problem. The most honest and useful output of this section is recognizing where machine learning does and doesn't add value here — not forcing a model to look impressive.

## The 3 things that matter

1. **Random Forest and a neural net both learned to predict "no booking" every time.** That matches the 89.5% baseline exactly — with only 2 positive cases in the test set, there's no way for either model to learn real signal. This isn't a failed experiment; it's the correct, honest outcome to report.

2. **Revenue forecasting didn't beat "assume next month looks like the average."** Ridge regression ($181 MAE) and Random Forest ($211 MAE) both underperformed simply predicting the historical mean revenue ($149 MAE). With 18 months of highly volatile monthly revenue ($0 to $891), that's expected — and worth stating plainly rather than cherry-picking a metric that looks better.

3. **Deep learning was correctly ruled out, not just skipped.** A TensorFlow/PyTorch model on 76 rows would overfit within a few epochs. Demonstrating that judgment — knowing when not to reach for a bigger model — is arguably more valuable in a portfolio than running one anyway.

## Bottom line
This section's value is methodological: correct time-based splits, leakage-free lag features, baseline comparisons, and honest reporting when the fancier model doesn't win. That discipline is what separates a real ML workflow from one that just produces impressive-looking numbers.

*Companion files: `ml_analysis.py`, `charts/` (3 PNGs), `ML_Report.docx`.*
