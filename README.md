# UK Telecom Cease Prediction

This project predicts the likelihood that a customer will place a cease (churn) so retention teams can prioritise outreach.

## Project Structure
- `app/` Streamlit app (single + batch scoring)
- `src/` reusable Python code (data prep, features, training, prediction)
- `notebooks/` EDA and experimentation
- `reports/` figures + slides
- `data/raw/` raw input data (excluded from git)

## Setup (Local)
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt