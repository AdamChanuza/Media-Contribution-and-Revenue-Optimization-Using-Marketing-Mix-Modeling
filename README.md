# Media Contribution and Revenue Optimization Using Marketing Mix Modeling

## Project Overview

This repository contains an end-to-end Marketing Mix Modeling (MMM) analysis using the Meta Robyn simulated weekly dataset. The project estimates model-based relationships between marketing activity and revenue, evaluates transformation parameters, and summarizes directional channel contribution and efficiency signals.

The results should be treated as model-based directional insights, not causal proof or evidence from a real company dataset.

## Business Problem

Marketing teams need to understand which channels are associated with revenue and how budget allocation could be improved. This project focuses on three practical questions: which channels show the strongest contribution signal, which appear most efficient relative to spend, and what budget allocation direction could be considered after validation.

## Dataset

The analysis uses the Meta Robyn simulated weekly dataset. The target variable is `revenue`. Media spend variables include `tv_S`, `ooh_S`, `print_S`, `facebook_S`, and `search_S`. Exposure and activity variables include `facebook_I` and `search_clicks_P`. Context variables include `competitor_sales_B` and `events`, while `newsletter` is treated as an organic marketing activity variable.

The Robyn demo dataset does not provide a full business data dictionary with exact units and measurement definitions. Variable interpretation is therefore based on the original Robyn input specification and naming conventions.

## Methodology

- Exploratory data analysis
- Revenue and media spend trend review
- Correlation analysis
- Adstock transformation for media carryover
- Hill saturation transformation for diminishing returns
- Grid search for adstock and saturation parameters
- Ridge Regression
- Time-aware 80/20 train/test split
- Model evaluation with R2, RMSE, and MAPE
- Channel contribution analysis
- Channel efficiency proxy analysis

## Repository Structure

```text
marketing-mix-modeling/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- NOTEBOOK_CLEANUP_REPORT.md
|-- data/
|   |-- raw/
|   |   `-- dt_simulated_weekly.RData
|   `-- processed/
|       `-- robyn_mmm_weekly.csv
|-- notebooks/
|   `-- Media Contribution and Revenue Optimization Using Marketing Mix Modeling.ipynb
|-- reports/
|   `-- Media_Contribution_and_Revenue_Optimization_Using_MMM.html
|-- scripts/
|   `-- export_notebook_to_html.py
|-- src/
|   |-- transformations.py
|   |-- model_selection.py
|   |-- evaluation.py
|   `-- plotting.py
`-- outputs/
    |-- figures/
    `-- tables/
```

## HTML Report

A polished HTML export of the final notebook is available at:

`reports/Media_Contribution_and_Revenue_Optimization_Using_MMM.html`

The HTML report includes a clickable table of contents, embedded figures and outputs, and collapsible code cells. To regenerate it after rerunning the notebook:

```bash
python scripts/export_notebook_to_html.py
```

## Key Results

- The final Ridge model achieves R2 of approximately 0.90 on the validation period.
- MAPE is approximately 7%.
- `tv_S` shows the strongest absolute model-based contribution signal.
- `facebook_S` shows the strongest efficiency proxy.
- `ooh_S` and `search_S` appear weaker in the current simplified specification.

These findings are directional and depend on the model specification, transformations, and validation approach.

## Business Recommendations

TV should be maintained as a scale-building channel because it has the strongest model-based contribution signal. Facebook deserves additional attention because it combines a positive contribution signal with the strongest efficiency proxy. OOH and Search should be reviewed before increasing budget.

Budget should not be reallocated directly from this notebook alone. The findings should be validated through experiments, lift tests, geo tests, or additional business review before final allocation decisions.

## Limitations

- The dataset is simulated.
- The implementation is a simplified MMM workflow.
- The grid search is intentionally small.
- The project does not implement the full Meta Robyn / Nevergrad optimization engine.
- The model is not calibrated with experimental results.
- The results do not provide causal proof.
- The efficiency proxy is not full ROAS.

## How to Run

1. Clone or download the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook from `notebooks/`.
4. Run all cells from top to bottom.
5. Generated outputs are stored in `outputs/`.
6. Regenerate the HTML report if needed:

```bash
python scripts/export_notebook_to_html.py
```

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook / Google Colab
- nbconvert
- BeautifulSoup

## Author

Author: [Your Name]  
Contact: [LinkedIn / GitHub]
