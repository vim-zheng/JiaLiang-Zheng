# SHINY_AKI_APP — Moderate-to-Severe AKI Risk Predictor after Type A Aortic Dissection

A standard Shiny risk predictor (bslib modern UI) built from the validation-best
model `BEST_XGBoost_model.rds` output by `03_run_8models.R`.

## How to run

In the R console（your personal computer）:

```r
shiny::runApp("D:/Rproject/Positron/01_prediction_of_AKI_after_AD/New_TAAD_AKI_term/thirdly_try_time_split/SHINY_AKI_APP")
```

Or open `app.R` and click "Run App" in the top-right of Positron.

## Features

- Reads the training data embedded in the rds and generates all input widgets
  automatically from the observed feature ranges (no hard-coded feature names)
- Optional SHAP individual-level explanation (waterfall plot, click the
  "Compute explanation" button)
- Reset to defaults / Random example buttons

## File structure

```
SHINY_AKI_APP/
├── app.R                        # Main Shiny application file
├── README.md
└── model/
    └── BEST_XGBoost_model.rds   # Best model (caret train object)
```

## Dependencies

shiny, bslib, ggplot2, shapviz, kernelshap, thematic (auto-installed if missing)

## Notes

- The risk cut-offs (0.3 / 0.6) are illustrative and not clinically validated
- If you replace the model rds, keep the file name consistent with `MODEL_FILE`
  in `app.R` (or edit that constant directly)
- This tool is for research purposes only and does not replace clinical judgment
