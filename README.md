# Criminal Justice AI Bias Analysis

**Group 2: Vidit, Ronaldo, Kaleemulla, Vishal**
**Course: Ethics, Sociology, and Governance of AI (MAIB AI 219)**

A comprehensive analysis of algorithmic bias in criminal justice AI systems, examining the COMPAS recidivism risk assessment tool and the NYPD Stop-Question-Frisk program.

## Overview

This project analyzes two landmark criminal justice datasets to demonstrate how AI systems can perpetuate and amplify racial disparities:

1. **COMPAS Dataset** - ProPublica's dataset exposing bias in recidivism prediction
2. **NYPD Stop-Question-Frisk 2012** - Records from NYC's controversial stop-and-frisk program

The analysis applies three critical lenses from the course:
- **Ethics**: Consequentialist, deontological, and virtue ethics perspectives
- **Sociology**: Power dynamics, who benefits/is harmed, systemic effects
- **Governance**: Regulatory frameworks, accountability mechanisms, recommendations

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- npm

### One-Command Setup (Recommended)

```bash
# 1. Create Python virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the analysis pipeline (generates all data and charts)
python scripts/generate_reports.py

# 3. Copy the PDF to public folder
cp 1906.04711v3.pdf web/public/

# 4. Install web dependencies and start dev server
cd web
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

### Building for Production

```bash
cd web
npm run build
```

The production build will be in `web/dist/`.

## Deploying to Netlify

### Option 1: Netlify CLI

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy (from project root)
netlify deploy --prod --dir=web/dist
```

### Option 2: Netlify Dashboard

1. Push this repository to GitHub
2. Log in to [Netlify](https://app.netlify.com)
3. Click "New site from Git"
4. Connect your repository
5. Build settings will be auto-detected from `netlify.toml`:
   - Base directory: `web`
   - Build command: `npm run build`
   - Publish directory: `web/dist`

### Important: Pre-generate Data

**Before deploying**, you must run the Python analysis locally:

```bash
python scripts/generate_reports.py
cp 1906.04711v3.pdf web/public/
```

The generated files in `web/public/data/` and `web/public/img/` must be committed and pushed to your repository. The web app is static and reads these pre-generated files.

## Project Structure

```
ESG-MidTerm-Bias-Code/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── scripts/
│   └── generate_reports.py      # Main analysis pipeline
├── web/
│   ├── package.json            # Node.js dependencies
│   ├── vite.config.js          # Vite configuration
│   ├── tailwind.config.js      # Tailwind CSS configuration
│   ├── netlify.toml            # Netlify deployment config
│   ├── index.html              # HTML entry point
│   ├── src/
│   │   ├── main.jsx            # React entry point
│   │   ├── App.jsx             # Main React application
│   │   └── index.css           # Tailwind CSS imports
│   └── public/
│       ├── favicon.svg         # Site favicon
│       ├── 1906.04711v3.pdf    # Context paper (copy manually)
│       ├── data/               # Generated JSON files
│       │   ├── dashboard_summary.json
│       │   ├── compas_analysis.json
│       │   ├── nypd_analysis.json
│       │   ├── comparison_analysis.json
│       │   └── charts_metadata.json
│       └── img/                # Generated chart images
│           ├── compas_*.png
│           ├── nypd_*.png
│           └── comparison_*.png
├── 2012.csv                    # NYPD SQF dataset
├── archive (1)/                # COMPAS dataset folder
│   └── propublicaCompassRecividism_data_fairml.csv/
│       └── propublica_data_for_fairml.csv
└── 1906.04711v3.pdf           # Context paper
```

## Analysis Details

### COMPAS Analysis

The COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) analysis includes:

- **Data Preprocessing**: Handling of one-hot encoded race features, binary target variable
- **Models Trained**:
  - Logistic Regression (interpretable baseline)
  - Random Forest (robust ensemble)
  - Gradient Boosting (high-performance)
- **Performance Metrics**: AUC, Accuracy, Precision, Recall, F1, Brier Score
- **Fairness Metrics**:
  - Demographic Parity Difference
  - Equalized Odds Difference
  - False Positive Rate by race
  - False Negative Rate by race
  - Calibration curves by race
- **Bias Mitigation**: Threshold optimization for equalized odds

### NYPD Stop-Question-Frisk Analysis

The NYPD SQF analysis includes:

- **Data Preprocessing**: Race code mapping, outcome variable selection (arrest/search/frisk)
- **Models Trained**: Same three-model comparison
- **Geographic Analysis**: Outcome rates by precinct, racial composition by precinct
- **Fairness Analysis**: Hit rates, selection rate disparities, error rate disparities
- **Confounding Visualization**: Race x Precinct x Outcome heatmaps

### Cross-Dataset Comparison

- Side-by-side base rate comparisons
- Fairness metric comparison across datasets
- Feedback loop visualization showing how biased data perpetuates bias

## Generated Visualizations

The analysis generates 15+ charts including:

1. Recidivism rate by race (COMPAS)
2. Prior convictions distribution by race (COMPAS)
3. Model comparison (COMPAS)
4. ROC curves (COMPAS)
5. Confusion matrix (COMPAS)
6. Fairness metrics by race (COMPAS)
7. Calibration curves by race (COMPAS)
8. Bias mitigation comparison (COMPAS)
9. Feature importance (COMPAS)
10. Stops by race (NYPD)
11. Outcome rate by race (NYPD)
12. Precinct analysis (NYPD)
13. Confounding heatmap (NYPD)
14. Age distribution (NYPD)
15. Model comparison (NYPD)
16. Fairness metrics (NYPD)
17. Cross-dataset base rate comparison
18. Cross-dataset fairness comparison
19. Feedback loop diagram

Each chart includes:
- Clear title and axis labels
- Detailed explanation (3-6 sentences)
- Speaker notes (bullet points for presentations)
- Talk track (20-30 second script)

## Technical Stack

### Python Analysis
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning models
- matplotlib - Visualization
- fairlearn - Fairness metrics and mitigation

### Web Application
- React 18 - UI framework
- Vite - Build tool
- Tailwind CSS - Styling
- Static JSON/PNG - No backend required

## Ethical Considerations

This project demonstrates that:

1. **Bias is systemic**: Both datasets show racial disparities that reflect historical discrimination
2. **Fairness has trade-offs**: Different fairness definitions are mathematically incompatible
3. **Technical fixes are insufficient**: Algorithmic interventions cannot fix fundamentally biased data
4. **Feedback loops amplify bias**: Biased predictions create biased data for future models

## References

1. Angwin et al. (2016). "Machine Bias" - ProPublica
2. Chouldechova (2017). "Fair Prediction with Disparate Impact"
3. Kleinberg et al. (2016). "Inherent Trade-Offs in Fair Risk Scores"
4. Goel et al. (2016). "Precinct or Prejudice? NYPD Stop-and-Frisk"
5. Corbett-Davies & Goel (2018). "Measure and Mismeasure of Fairness"
6. Floyd v. City of New York (2013)

## License

This project is for educational purposes as part of MAIB AI 219 coursework.

---

**Group 2: Vidit, Ronaldo, Kaleemulla, Vishal**
Ethics, Sociology, and Governance of AI (MAIB AI 219)
