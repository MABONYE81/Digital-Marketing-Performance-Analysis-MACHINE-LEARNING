# 05_MachineLearning — README

## Where's the dashboard?
Same answer as Python (`03_Python`): there isn't one interactive dashboard
— there are **3 chart images**, already generated, in `02_charts/`. Open
them directly like photos:

1. `01_confusion_matrix.png` — how well the booking-prediction model did
2. `02_feature_importance.png` — which features the model relied on most
3. `03_revenue_forecast.png` — predicted vs actual monthly revenue

You don't need to run anything to see these — they're already sitting in
this folder.

## What this section actually found (read before running)
With only 76 usable weeks and 9 booking events, the honest result is that
the fancier models (Random Forest, a small neural network) didn't beat a
simple baseline. That's explained in full in `03_ML_Report.docx` — it's a
deliberate, reported finding, not a bug.

## How to run the script yourself
1. Install **Python**: https://python.org (check "Add to PATH" during setup)
2. Install **VS Code**: https://code.visualstudio.com + its Python extension
3. Open this folder in VS Code, then open a terminal (Terminal → New Terminal)
4. Run:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn numpy
   python 01_ml_analysis.py
   ```
5. Text results (accuracy, precision, recall, MAE) print to the terminal;
   updated charts save alongside the script.

### Don't double-click the .py file
Same issue as every Python script in this portfolio — on Windows it opens
a console that closes itself instantly. Always run it from a terminal
(step 4 above), not by double-clicking.

## Files in this folder
1. `01_ml_analysis.py` — full scikit-learn script (classification + regression)
2. `02_charts/` — the 3 chart images (already generated — open these directly)
3. `03_ML_Report.docx` — full write-up, including the honest baseline comparisons
4. `04_ML_Insights_Summary.md` — one-page takeaways
