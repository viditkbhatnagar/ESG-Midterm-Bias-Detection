import { useState, useEffect } from 'react'

// ============================================================================
// DATA FETCHING HOOK
// ============================================================================

function useData(filename) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`/data/${filename}`)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load ${filename}`)
        return res.json()
      })
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false))
  }, [filename])

  return { data, loading, error }
}

// ============================================================================
// ICONS
// ============================================================================

const Icons = {
  Menu: () => (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  ),
  Close: () => (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
  ChevronDown: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  ),
  ChevronUp: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
    </svg>
  ),
  Download: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
  ),
  ExternalLink: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
    </svg>
  ),
  Code: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
    </svg>
  ),
  Beaker: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
    </svg>
  ),
  LightBulb: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  ),
  Warning: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
}

// ============================================================================
// HEADER NAVIGATION
// ============================================================================

const navItems = [
  { id: 'overview', label: 'Overview' },
  { id: 'datasets', label: 'Data' },
  { id: 'methodology', label: 'Methods' },
  { id: 'model-explainer', label: 'Models' },
  { id: 'compas', label: 'COMPAS' },
  { id: 'nypd', label: 'NYPD' },
  { id: 'what-if', label: 'What-If' },
  { id: 'comparison', label: 'Compare' },
  { id: 'ethics', label: 'Ethics' },
  { id: 'limitations', label: 'Limits' },
  { id: 'references', label: 'Refs' },
  { id: 'artifacts', label: 'Download' },
]

function Header({ activeSection, onNavigate }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const handleNav = (id) => {
    onNavigate(id)
    setMobileMenuOpen(false)
    const element = document.getElementById(id)
    if (element) {
      const headerOffset = 70
      const elementPosition = element.getBoundingClientRect().top
      const offsetPosition = elementPosition + window.pageYOffset - headerOffset
      window.scrollTo({ top: offsetPosition, behavior: 'smooth' })
    }
  }

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-gray-200 shadow-sm">
      {/* Main header row */}
      <div className="flex items-center h-14 px-4">
        {/* Logo */}
        <div className="flex-shrink-0 mr-6">
          <h1 className="text-base font-bold text-gray-900">
            AI Bias
            <span className="hidden sm:inline text-gray-500 font-normal"> | Criminal Justice</span>
          </h1>
        </div>

        {/* Desktop Navigation - scrollable */}
        <nav className="hidden md:flex flex-1 items-center overflow-x-auto scrollbar-thin">
          <div className="flex items-center gap-1">
            {navItems.map(item => (
              <button
                key={item.id}
                onClick={() => handleNav(item.id)}
                className={`px-2.5 py-1.5 text-xs font-medium rounded-md whitespace-nowrap transition-colors ${
                  activeSection === item.id
                    ? 'text-blue-700 bg-blue-100'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </nav>

        {/* Team badge - desktop */}
        <div className="hidden lg:flex items-center ml-4 pl-4 border-l border-gray-200">
          <span className="text-xs text-gray-500">Group 2</span>
        </div>

        {/* Mobile menu button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden ml-auto p-2 rounded-lg hover:bg-gray-100"
        >
          {mobileMenuOpen ? <Icons.Close /> : <Icons.Menu />}
        </button>
      </div>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-white border-t border-gray-100 shadow-lg max-h-[70vh] overflow-y-auto">
          <nav className="p-3 grid grid-cols-3 gap-2">
            {navItems.map(item => (
              <button
                key={item.id}
                onClick={() => handleNav(item.id)}
                className={`px-2 py-2.5 text-xs font-medium rounded-lg text-center transition-colors ${
                  activeSection === item.id
                    ? 'text-blue-700 bg-blue-100'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>
          <div className="px-4 pb-3 border-t border-gray-100 pt-3 bg-gray-50">
            <p className="text-xs text-gray-600 font-medium">Group 2: Vidit, Ronaldo, Kaleemulla, Vishal</p>
            <p className="text-xs text-gray-400">MAIB AI 219 - Ethics, Sociology & Governance of AI</p>
          </div>
        </div>
      )}
    </header>
  )
}

// ============================================================================
// CODE BLOCK COMPONENT
// ============================================================================

function CodeBlock({ title, language = 'python', code, explanation, impact }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-gray-50 rounded-xl overflow-hidden my-4 border border-gray-200">
      <div
        className="flex items-center justify-between px-4 py-3 bg-gray-100 cursor-pointer border-b border-gray-200"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <span className="text-gray-600"><Icons.Code /></span>
          <span className="text-sm font-medium text-gray-800">{title}</span>
          <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded font-medium">{language}</span>
        </div>
        <button className="text-gray-500 hover:text-gray-800">
          {expanded ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
        </button>
      </div>

      {expanded && (
        <>
          <pre className="p-4 overflow-x-auto text-sm bg-white">
            <code className="text-gray-800 font-mono whitespace-pre">{code}</code>
          </pre>

          {explanation && (
            <div className="px-4 py-3 bg-blue-50 border-t border-blue-100">
              <p className="text-sm text-gray-700">
                <span className="text-blue-600 font-semibold">How it works: </span>
                {explanation}
              </p>
            </div>
          )}

          {impact && (
            <div className="px-4 py-3 bg-amber-50 border-t border-amber-100">
              <p className="text-sm text-gray-700">
                <span className="text-amber-600 font-semibold">Impact: </span>
                {impact}
              </p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ============================================================================
// WHAT-IF CARD COMPONENT
// ============================================================================

function WhatIfCard({ scenario, effect, code, beforeValue, afterValue, interpretation }) {
  const [showDetails, setShowDetails] = useState(false)

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-lg transition-shadow">
      <div className="p-5">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-purple-100 rounded-lg">
            <Icons.Beaker />
          </div>
          <div className="flex-1">
            <h4 className="font-semibold text-gray-900">{scenario}</h4>
            <p className="text-sm text-gray-600 mt-1">{effect}</p>
          </div>
        </div>

        {(beforeValue || afterValue) && (
          <div className="mt-4 flex items-center gap-4">
            {beforeValue && (
              <div className="flex-1 p-3 bg-red-50 rounded-lg border border-red-100">
                <p className="text-xs text-red-600 font-medium">BEFORE</p>
                <p className="text-lg font-bold text-red-700">{beforeValue}</p>
              </div>
            )}
            <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            {afterValue && (
              <div className="flex-1 p-3 bg-green-50 rounded-lg border border-green-100">
                <p className="text-xs text-green-600 font-medium">AFTER</p>
                <p className="text-lg font-bold text-green-700">{afterValue}</p>
              </div>
            )}
          </div>
        )}

        <button
          onClick={() => setShowDetails(!showDetails)}
          className="mt-4 text-sm text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1"
        >
          {showDetails ? 'Hide' : 'Show'} technical details
          {showDetails ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
        </button>
      </div>

      {showDetails && (
        <div className="border-t border-gray-100 bg-gray-50 p-5">
          {code && (
            <pre className="bg-gray-900 text-gray-300 p-4 rounded-lg text-xs overflow-x-auto mb-4">
              <code>{code}</code>
            </pre>
          )}
          {interpretation && (
            <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
              <p className="text-sm text-blue-800">
                <span className="font-semibold">Interpretation: </span>
                {interpretation}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// CHART CARD COMPONENT
// ============================================================================

function ChartCard({ chart, showCode = true }) {
  const [showNotes, setShowNotes] = useState(false)
  const [showAlgorithm, setShowAlgorithm] = useState(false)

  if (!chart) return null

  // Algorithm snippets for different chart types
  const algorithmSnippets = {
    'compas_recidivism_by_race': {
      code: `# Base Rate Calculation by Group
def calculate_base_rates(df, target_col, group_col):
    """
    Base rate = P(Y=1 | Group=g)
    This measures the observed outcome rate per group
    """
    return df.groupby(group_col)[target_col].mean()

# Key insight: Different base rates across groups
# make it mathematically impossible to satisfy
# multiple fairness criteria simultaneously
# (Chouldechova's Impossibility Theorem)`,
      explanation: "Base rates are calculated by computing the mean of the binary outcome variable within each racial group. This is the foundation of understanding disparate impact.",
      impact: "If base rates differ significantly (>5%), achieving equal FPR AND equal FNR simultaneously becomes mathematically impossible while maintaining calibration."
    },
    'compas_model_comparison': {
      code: `# Model Training with scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Logistic Regression - Interpretable baseline
lr = LogisticRegression(
    C=1.0,           # Regularization strength (1/lambda)
    max_iter=1000,   # Convergence iterations
    random_state=42  # Reproducibility
)

# Key hyperparameters that affect fairness:
# - C (regularization): Lower C = more regularization
#   = smaller coefficients = less reliance on
#   potentially biased features like priors
# - class_weight: 'balanced' can help with
#   imbalanced classes but may increase FPR`,
      explanation: "We train three models of increasing complexity. Logistic Regression provides interpretable coefficients, Random Forest captures non-linear interactions, and Gradient Boosting often achieves highest AUC.",
      impact: "Higher model complexity doesn't guarantee better fairness. In fact, complex models may learn subtle proxies for race more effectively, potentially increasing disparities."
    },
    'compas_fairness_by_race': {
      code: `# Fairness Metrics Calculation
def compute_fairness_metrics(y_true, y_pred, sensitive):
    metrics = {}

    for group in np.unique(sensitive):
        mask = sensitive == group
        y_t, y_p = y_true[mask], y_pred[mask]

        # Confusion matrix elements
        TP = ((y_t == 1) & (y_p == 1)).sum()
        FP = ((y_t == 0) & (y_p == 1)).sum()
        TN = ((y_t == 0) & (y_p == 0)).sum()
        FN = ((y_t == 1) & (y_p == 0)).sum()

        # False Positive Rate: P(Y_pred=1 | Y_true=0)
        # "Of innocent people, how many were wrongly flagged?"
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        # False Negative Rate: P(Y_pred=0 | Y_true=1)
        # "Of guilty people, how many were missed?"
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

        metrics[group] = {'FPR': FPR, 'FNR': FNR}

    # Equalized Odds requires: FPR_A = FPR_B AND FNR_A = FNR_B
    return metrics`,
      explanation: "FPR measures errors on non-recidivists (false accusations), while FNR measures errors on recidivists (missed predictions). ProPublica found COMPAS had higher FPR for Black defendants.",
      impact: "If FPR for Black defendants is 45% vs 23% for White defendants, Black non-recidivists are nearly TWICE as likely to be wrongly labeled high-risk."
    },
    'compas_calibration_curves': {
      code: `# Calibration Analysis
from sklearn.calibration import calibration_curve

def check_calibration(y_true, y_prob, group, n_bins=10):
    """
    Calibration = Do predicted probabilities match reality?
    If model says 70% risk, do 70% actually recidivate?

    Perfect calibration: points on diagonal line
    """
    mask = sensitive == group
    prob_true, prob_pred = calibration_curve(
        y_true[mask],
        y_prob[mask],
        n_bins=n_bins,
        strategy='uniform'  # Equal-width bins
    )

    # Brier Score: Mean squared error of probabilities
    # Lower = better calibrated
    brier = np.mean((y_prob[mask] - y_true[mask])**2)

    return prob_true, prob_pred, brier

# COMPAS argument: "Our scores ARE calibrated by race"
# ProPublica argument: "But error RATES differ by race"
# Both can be true simultaneously - this is the paradox`,
      explanation: "Calibration curves show whether predicted probabilities match observed frequencies. COMPAS company (Northpointe) argued their scores were equally calibrated across races.",
      impact: "Equal calibration across groups + different base rates = mathematically guaranteed unequal error rates. This is the core of the fairness impossibility theorem."
    },
    'nypd_stops_by_race': {
      code: `# Stop Distribution Analysis
def analyze_stop_distribution(df):
    """
    Compare stop rates to population demographics

    NYC 2012 Demographics (approx):
    - White: 33%
    - Hispanic: 29%
    - Black: 26%
    - Asian: 13%

    If stops were proportional to population,
    we'd expect similar percentages.
    """
    stop_counts = df['race'].value_counts(normalize=True)

    # Disparity ratio = Stop% / Population%
    # Ratio > 1 means over-representation
    disparity = {
        'Black': stop_counts.get('B', 0) / 0.26,
        'Hispanic': stop_counts.get('Q', 0) / 0.29,
        'White': stop_counts.get('W', 0) / 0.33,
    }

    return disparity

# Result: Black and Hispanic individuals are stopped
# at 2-3x their population proportion`,
      explanation: "We compare stop rates to baseline population demographics. Significant over-representation suggests either differential behavior OR differential policing (bias).",
      impact: "If Black residents are 26% of NYC but 55% of stops, the disparity ratio is 2.1x - meaning Black individuals are stopped at more than twice the rate expected from population alone."
    },
    'nypd_outcome_by_race': {
      code: `# Hit Rate Analysis
def calculate_hit_rates(df, outcome_col='arstmade'):
    """
    Hit Rate = P(Contraband/Arrest | Stop)

    If police have equally good 'reasonable suspicion'
    across groups, hit rates should be EQUAL.

    Lower hit rates for minorities suggests:
    - Lower threshold of suspicion applied
    - Stops based on race rather than behavior
    """
    hit_rates = df.groupby('race')[outcome_col].apply(
        lambda x: (x == 'Y').mean()
    )

    # Benchmark test (Knowles, Persico, Todd 2001):
    # Under optimal policing, hit rates equalize
    # across groups. Differences indicate bias.

    return hit_rates

# Finding: Similar or LOWER hit rates for minorities
# despite being stopped more often = evidence of bias`,
      explanation: "Hit rates measure how often stops result in finding contraband or making an arrest. Economic theory predicts that unbiased policing should equalize hit rates across groups.",
      impact: "If White individuals have 7% arrest rate when stopped but Black individuals have 5%, yet Black individuals are stopped 3x more, this suggests racial profiling - stopping based on race rather than suspicious behavior."
    },
    'comparison_feedback_loop': {
      code: `# Feedback Loop Dynamics
"""
The Predictive Policing Feedback Loop:

1. Historical Data Collection:
   arrests_data = collect_from_overpoliced_areas()
   # Data reflects WHERE police were, not WHERE crime is

2. Model Training:
   model.fit(arrests_data, labels=was_arrested)
   # Model learns: "these neighborhoods are high crime"
   # Reality: "these neighborhoods are heavily policed"

3. Prediction & Deployment:
   risk_scores = model.predict(current_population)
   # Directs more police to same areas

4. New Data Generation:
   new_arrests = police_activity_in_high_risk_areas()
   # More police = more arrests = "validates" the model

5. LOOP: Go to step 2 with new_arrests
   # Each iteration AMPLIFIES the original bias

Mathematical formulation:
P(arrest|location) = P(crime|location) * P(police|location)
                     * P(arrest|crime,police)

If P(police|minority_area) >> P(police|white_area),
then P(arrest|minority_area) is inflated regardless
of actual crime rates.
"""`,
      explanation: "Biased historical data creates biased predictions, which guide resource allocation, which generates new biased data. This is a self-reinforcing cycle that amplifies initial disparities.",
      impact: "Lum & Isaac (2016) showed that predictive policing in Oakland would direct police to minority neighborhoods even when using DRUG data that shows equal usage rates across races - purely due to historical arrest patterns."
    }
  }

  const snippet = algorithmSnippets[chart.chart_id]

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm hover:shadow-md transition-shadow">
      <div className="p-5 border-b border-gray-100">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="font-semibold text-gray-900">{chart.title}</h3>
            <span className="inline-block mt-2 text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full">
              {chart.dataset}
            </span>
          </div>
        </div>
      </div>

      {/* Chart Image */}
      <div className="p-4 bg-gray-50">
        <img
          src={`/img/${chart.filename}`}
          alt={chart.title}
          className="w-full h-auto rounded-lg"
          loading="lazy"
        />
      </div>

      {/* Explanation */}
      <div className="p-5">
        <p className="text-sm text-gray-600 leading-relaxed">{chart.explanation}</p>

        {/* Algorithm Code Section */}
        {showCode && snippet && (
          <div className="mt-4">
            <button
              onClick={() => setShowAlgorithm(!showAlgorithm)}
              className="flex items-center gap-2 text-sm font-medium text-purple-600 hover:text-purple-700"
            >
              <Icons.Code />
              {showAlgorithm ? 'Hide' : 'Show'} Algorithm Behind This
              {showAlgorithm ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
            </button>

            {showAlgorithm && (
              <CodeBlock
                title="Algorithm Implementation"
                code={snippet.code}
                explanation={snippet.explanation}
                impact={snippet.impact}
              />
            )}
          </div>
        )}

        {/* Speaker Notes */}
        {chart.speaker_notes && chart.speaker_notes.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-100">
            <button
              onClick={() => setShowNotes(!showNotes)}
              className="flex items-center gap-2 text-sm font-medium text-gray-600 hover:text-gray-800"
            >
              <Icons.LightBulb />
              {showNotes ? 'Hide' : 'Show'} Key Insights
              {showNotes ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
            </button>

            {showNotes && (
              <ul className="mt-3 space-y-2 pl-4">
                {chart.speaker_notes.map((note, idx) => (
                  <li key={idx} className="text-sm text-gray-600 flex gap-2">
                    <span className="text-blue-500 mt-1">•</span>
                    <span>{note}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================================================
// STAT CARD
// ============================================================================

function StatCard({ value, label, sublabel, color = 'blue' }) {
  const colors = {
    blue: 'border-l-blue-500 bg-blue-50/50',
    green: 'border-l-green-500 bg-green-50/50',
    red: 'border-l-red-500 bg-red-50/50',
    yellow: 'border-l-yellow-500 bg-yellow-50/50',
    purple: 'border-l-purple-500 bg-purple-50/50',
  }

  return (
    <div className={`p-5 rounded-xl border-l-4 ${colors[color]} border border-gray-100`}>
      <div className="text-3xl font-bold text-gray-900">{value}</div>
      <div className="text-sm text-gray-600 mt-1">{label}</div>
      {sublabel && <div className="text-xs text-gray-400 mt-0.5">{sublabel}</div>}
    </div>
  )
}

// ============================================================================
// SECTION: OVERVIEW
// ============================================================================

function OverviewSection({ dashboardData }) {
  return (
    <section id="overview" className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Overview</h2>
        <p className="text-lg text-gray-600 leading-relaxed">
          This analysis examines algorithmic bias in two landmark criminal justice systems:
          <strong> COMPAS</strong> (recidivism risk prediction) and <strong>NYPD Stop-Question-Frisk</strong>.
          Using machine learning, statistical analysis, and fairness metrics, we demonstrate how these systems
          produce <em>disparate outcomes</em> across racial groups.
        </p>
      </div>

      {/* Key Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard
          value={dashboardData?.datasets?.COMPAS?.sample_size?.toLocaleString() || '6,172'}
          label="COMPAS Records"
          sublabel="Broward County, FL"
          color="blue"
        />
        <StatCard
          value={dashboardData?.datasets?.NYPD?.sample_size?.toLocaleString() || '100,000'}
          label="NYPD Stops"
          sublabel="NYC 2012"
          color="green"
        />
        <StatCard
          value={`${((dashboardData?.datasets?.COMPAS?.base_rate || 0.455) * 100).toFixed(1)}%`}
          label="Recidivism Rate"
          sublabel="Two-year window"
          color="red"
        />
        <StatCard
          value="0.738"
          label="Best AUC"
          sublabel="COMPAS model"
          color="purple"
        />
        <StatCard
          value={dashboardData?.total_charts || 19}
          label="Visualizations"
          sublabel="With explanations"
          color="yellow"
        />
      </div>

      {/* Key Findings */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 bg-gray-50">
          <h3 className="font-semibold text-gray-900">Key Findings Summary</h3>
        </div>
        <div className="p-6 space-y-4">
          {(dashboardData?.key_findings || [
            { finding: 'Both datasets show significant racial disparities', implication: 'Algorithmic systems inherit these disparities' },
            { finding: 'Different fairness metrics reveal different biases', implication: 'No single metric captures all forms of bias' },
            { finding: 'Bias mitigation involves accuracy trade-offs', implication: 'Fairness requires value judgments' },
            { finding: 'Geographic and demographic factors are confounded', implication: 'Controlling for location does not eliminate disparity' },
          ]).slice(0, 4).map((finding, idx) => (
            <div key={idx} className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center font-bold text-sm">
                {idx + 1}
              </div>
              <div>
                <p className="font-medium text-gray-900">{finding.finding}</p>
                <p className="text-sm text-gray-500 mt-0.5">{finding.implication}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Context Paper */}
      <div className="p-5 bg-amber-50 border border-amber-200 rounded-xl">
        <div className="flex gap-4">
          <div className="text-amber-600">
            <Icons.Warning />
          </div>
          <div>
            <p className="font-medium text-amber-800">Context Paper Available</p>
            <p className="text-sm text-amber-700 mt-1">
              For the mathematical proofs behind fairness impossibility theorems (Chouldechova, Kleinberg et al.),
              see the accompanying research paper on why multiple fairness criteria cannot be simultaneously satisfied.
            </p>
            <a
              href="/1906.04711v3.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm font-medium text-amber-800 hover:text-amber-900 mt-2"
            >
              View PDF <Icons.ExternalLink />
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: DATASETS
// ============================================================================

function DatasetsSection({ compasData, nypdData }) {
  return (
    <section id="datasets" className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Datasets</h2>
        <p className="text-lg text-gray-600">
          Two foundational datasets in the algorithmic fairness literature.
        </p>
      </div>

      <div className="space-y-6">
        {/* COMPAS */}
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 bg-blue-50 border-b border-blue-100">
            <h3 className="text-lg font-semibold text-blue-900">COMPAS Dataset</h3>
            <p className="text-sm text-blue-700">ProPublica Investigation, 2016</p>
          </div>
          <div className="p-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <div className="space-y-4 min-w-0">
                <p className="text-sm text-gray-600">
                  Correctional Offender Management Profiling for Alternative Sanctions (COMPAS) is a
                  proprietary risk assessment tool used in US courts to predict recidivism likelihood.
                  ProPublica's 2016 investigation revealed significant racial disparities in false positive rates.
                </p>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-gray-500">Sample Size</p>
                    <p className="font-semibold">{compasData?.sample_size?.toLocaleString() || '6,172'}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-gray-500">Base Rate</p>
                    <p className="font-semibold">45.5%</p>
                  </div>
                </div>
              </div>
              <div className="min-w-0">
              <CodeBlock
                title="Data Schema"
                language="python"
                code={`# COMPAS dataset columns
schema = {
    'Two_yr_Recidivism': 'Target (0/1)',  # Recidivate in 2 years?
    'Number_of_Priors': 'int',             # Prior convictions
    'Age_Above_FourtyFive': 'binary',      # Age > 45
    'Age_Below_TwentyFive': 'binary',      # Age < 25
    'African_American': 'binary',          # Race indicator
    'Female': 'binary',                    # Sex indicator
    'Misdemeanor': 'binary',               # Charge type
}
# NOTE: Original COMPAS uses 137 features
# This public dataset is a subset released by ProPublica`}
              explanation="The dataset uses one-hot encoding for race. 'White' is the implicit reference category (all race flags = 0)."
              impact="Using race as a direct feature is illegal in many jurisdictions. However, removing it doesn't prevent discrimination - proxies like zip code and priors correlate with race."
            />
              </div>
            </div>
          </div>
        </div>

        {/* NYPD */}
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 bg-green-50 border-b border-green-100">
            <h3 className="text-lg font-semibold text-green-900">NYPD Stop-Question-Frisk</h3>
            <p className="text-sm text-green-700">Public Records, 2012</p>
          </div>
          <div className="p-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <div className="space-y-4 min-w-0">
                <p className="text-sm text-gray-600">
                  2012 was the peak year of NYC's Stop-Question-Frisk program with over 500,000 stops.
                  The program was ruled unconstitutional in Floyd v. City of New York (2013) due to
                  racial profiling. This data is now used to study algorithmic bias in policing.
                </p>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-gray-500">Sample Size</p>
                    <p className="font-semibold">{nypdData?.sample_size?.toLocaleString() || '100,000'}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-gray-500">Arrest Rate</p>
                    <p className="font-semibold">6.0%</p>
                  </div>
                </div>
              </div>
              <div className="min-w-0">
              <CodeBlock
                title="Data Schema"
                language="python"
                code={`# NYPD SQF key columns (112 total)
schema = {
    'race': 'B/W/Q/A/P/I/Z',   # B=Black, W=White, Q=Hispanic
    'arstmade': 'Y/N',         # Was arrest made? (Target)
    'frisked': 'Y/N',          # Was person frisked?
    'searched': 'Y/N',         # Was person searched?
    'contrabn': 'Y/N',         # Contraband found?
    'pct': 'int',              # Precinct number
    'cs_*': 'Y/N',             # Circumstance flags (10+)
    'pf_*': 'Y/N',             # Physical force flags (8)
}
# Critical: Only STOPPED people in data
# This is SELECTION BIAS`}
                explanation="The dataset only contains people who were stopped. We cannot observe the counterfactual - innocent people who weren't stopped but would have been if they were a different race."
                impact="Selection bias means our models learn 'who gets stopped' not 'who should be stopped'. Using this data for predictive policing perpetuates the bias."
              />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: METHODOLOGY
// ============================================================================

function MethodologySection() {
  return (
    <section id="methodology" className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Methodology</h2>
        <p className="text-lg text-gray-600">
          Our analysis pipeline combines machine learning, statistical testing, and algorithmic fairness evaluation.
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Models */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Models Trained</h3>
          <div className="space-y-3">
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="font-medium">Logistic Regression</p>
              <p className="text-sm text-gray-500">Linear, interpretable, coefficients show feature importance</p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="font-medium">Random Forest</p>
              <p className="text-sm text-gray-500">Ensemble of trees, captures non-linear patterns</p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="font-medium">Gradient Boosting</p>
              <p className="text-sm text-gray-500">Sequential trees, often highest accuracy</p>
            </div>
          </div>

          <CodeBlock
            title="Model Training Pipeline"
            language="python"
            code={`# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 30% for testing
    stratify=y,         # Preserve class balance
    random_state=42     # Reproducibility
)

# Why stratify?
# Without it, test set might have different
# class distribution than training set,
# giving misleading performance estimates`}
          />
        </div>

        {/* Performance Metrics */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Performance Metrics</h3>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between py-2 border-b border-gray-100">
              <span className="font-medium">AUC-ROC</span>
              <span className="text-gray-500">Ranking ability</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-100">
              <span className="font-medium">Accuracy</span>
              <span className="text-gray-500">(TP+TN) / Total</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-100">
              <span className="font-medium">Precision</span>
              <span className="text-gray-500">TP / (TP+FP)</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-100">
              <span className="font-medium">Recall</span>
              <span className="text-gray-500">TP / (TP+FN)</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-100">
              <span className="font-medium">F1 Score</span>
              <span className="text-gray-500">Harmonic mean</span>
            </div>
            <div className="flex justify-between py-2">
              <span className="font-medium">Brier Score</span>
              <span className="text-gray-500">Calibration quality</span>
            </div>
          </div>

          <CodeBlock
            title="AUC Calculation"
            language="python"
            code={`# AUC = Area Under ROC Curve
# Interpretation: Probability that a randomly
# chosen positive has higher score than
# a randomly chosen negative

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred_proba)

# AUC = 0.5: Random guessing
# AUC = 0.7: Acceptable
# AUC = 0.8: Good
# AUC = 1.0: Perfect (suspicious!)`}
          />
        </div>

        {/* Fairness Metrics */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Fairness Metrics</h3>
          <div className="space-y-3 text-sm">
            <div className="p-3 bg-red-50 rounded-lg border border-red-100">
              <p className="font-medium text-red-800">Demographic Parity</p>
              <p className="text-red-600 text-xs">P(Ŷ=1|A=a) = P(Ŷ=1|A=b)</p>
              <p className="text-red-700 mt-1">Equal selection rates across groups</p>
            </div>
            <div className="p-3 bg-blue-50 rounded-lg border border-blue-100">
              <p className="font-medium text-blue-800">Equalized Odds</p>
              <p className="text-blue-600 text-xs">FPR_a = FPR_b AND TPR_a = TPR_b</p>
              <p className="text-blue-700 mt-1">Equal error rates given true outcome</p>
            </div>
            <div className="p-3 bg-green-50 rounded-lg border border-green-100">
              <p className="font-medium text-green-800">Calibration</p>
              <p className="text-green-600 text-xs">P(Y=1|Ŷ=s,A=a) = P(Y=1|Ŷ=s,A=b) = s</p>
              <p className="text-green-700 mt-1">Same score means same probability</p>
            </div>
          </div>

          <CodeBlock
            title="Fairness Impossibility"
            language="python"
            code={`# Chouldechova's Impossibility Theorem (2017)
#
# IF base rates differ: P(Y=1|A=a) ≠ P(Y=1|A=b)
# THEN you CANNOT simultaneously have:
#   1. Equal FPR across groups
#   2. Equal FNR across groups
#   3. Equal PPV (calibration) across groups
#
# Proof: Algebraic constraint from Bayes' theorem
# This is MATH, not a technical limitation

# Implication: We must CHOOSE which fairness
# criterion to prioritize - this is a VALUES
# question, not a technical one.`}
          />
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: MODEL EXPLAINER
// ============================================================================

function ModelExplainerSection() {
  return (
    <section id="model-explainer" className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Model Explainer</h2>
        <p className="text-lg text-gray-600">
          Deep dive into how each algorithm works and which hyperparameters affect fairness outcomes.
        </p>
      </div>

      {/* Logistic Regression */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white">
          <h3 className="text-lg font-semibold">Logistic Regression</h3>
          <p className="text-blue-100 text-sm">The interpretable baseline</p>
        </div>
        <div className="p-6">
          <div className="grid lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-3">How It Works</h4>
              <p className="text-sm text-gray-600 mb-4">
                Logistic regression models the log-odds of the outcome as a linear combination of features.
                Each coefficient represents the change in log-odds for a one-unit increase in that feature,
                holding others constant.
              </p>

              <CodeBlock
                title="Mathematical Formulation"
                language="python"
                code={`# Logistic Regression Model
# P(Y=1|X) = σ(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ)
# where σ(z) = 1 / (1 + e^(-z))  [sigmoid function]

# In sklearn:
model = LogisticRegression(
    C=1.0,              # Inverse regularization strength
    penalty='l2',       # Ridge regularization
    solver='lbfgs',     # Optimization algorithm
    max_iter=1000       # Convergence iterations
)

# Coefficient interpretation:
# β_priors = 0.35 means:
# Each additional prior conviction increases
# log-odds of recidivism by 0.35
# Odds ratio = e^0.35 = 1.42 (42% higher odds)`}
              />
            </div>

            <div>
              <h4 className="font-medium text-gray-900 mb-3">Key Hyperparameters & Fairness</h4>
              <div className="space-y-3">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium">C (Regularization)</p>
                      <p className="text-sm text-gray-500 mt-1">
                        Lower C = stronger regularization = smaller coefficients
                      </p>
                    </div>
                    <span className="text-xs px-2 py-1 bg-yellow-100 text-yellow-700 rounded">Affects Fairness</span>
                  </div>
                  <div className="mt-3 p-3 bg-yellow-50 rounded border border-yellow-100">
                    <p className="text-xs text-yellow-800">
                      <strong>Fairness Impact:</strong> Strong regularization (C=0.01) shrinks all coefficients,
                      reducing reliance on any single feature including biased proxies. May reduce disparities
                      but also reduces overall accuracy.
                    </p>
                  </div>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium">class_weight</p>
                      <p className="text-sm text-gray-500 mt-1">
                        'balanced' adjusts weights inversely proportional to class frequencies
                      </p>
                    </div>
                    <span className="text-xs px-2 py-1 bg-red-100 text-red-700 rounded">High Impact</span>
                  </div>
                  <div className="mt-3 p-3 bg-red-50 rounded border border-red-100">
                    <p className="text-xs text-red-800">
                      <strong>Fairness Impact:</strong> Balanced weights increase recall (catch more positives)
                      but may increase FPR disproportionately for majority groups. Use with caution.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Random Forest */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-green-500 to-green-600 text-white">
          <h3 className="text-lg font-semibold">Random Forest</h3>
          <p className="text-green-100 text-sm">Ensemble of decision trees</p>
        </div>
        <div className="p-6">
          <div className="grid lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-3">How It Works</h4>
              <p className="text-sm text-gray-600 mb-4">
                Random Forest builds many decision trees on random subsets of data and features,
                then averages their predictions. This reduces overfitting and captures non-linear relationships.
              </p>

              <CodeBlock
                title="Random Forest Algorithm"
                language="python"
                code={`# Random Forest Ensemble
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Tree depth (None = unlimited)
    min_samples_split=2,   # Min samples to split a node
    min_samples_leaf=1,    # Min samples at leaf node
    max_features='sqrt',   # Features per tree: √n_features
    bootstrap=True,        # Sample with replacement
    random_state=42
)

# Prediction: Average of all tree predictions
# P(Y=1) = (1/n_trees) * Σ tree_i.predict(X)

# Feature importance: Mean decrease in impurity
# across all trees where feature is used`}
              />
            </div>

            <div>
              <h4 className="font-medium text-gray-900 mb-3">Key Hyperparameters & Fairness</h4>
              <div className="space-y-3">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="font-medium">max_depth</p>
                  <p className="text-sm text-gray-500 mt-1">
                    Deeper trees = more complex decision boundaries = can learn subtle patterns
                  </p>
                  <div className="mt-3 p-3 bg-orange-50 rounded border border-orange-100">
                    <p className="text-xs text-orange-800">
                      <strong>Fairness Impact:</strong> Very deep trees may memorize training data,
                      including any biases. Shallower trees (max_depth=5-10) may generalize better
                      and rely less on noisy correlations.
                    </p>
                  </div>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="font-medium">min_samples_leaf</p>
                  <p className="text-sm text-gray-500 mt-1">
                    Higher values prevent trees from creating very specific rules
                  </p>
                  <div className="mt-3 p-3 bg-blue-50 rounded border border-blue-100">
                    <p className="text-xs text-blue-800">
                      <strong>Fairness Impact:</strong> min_samples_leaf=50 ensures each prediction
                      is based on at least 50 training examples, reducing reliance on outliers
                      which may be biased edge cases.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Gradient Boosting */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-purple-500 to-purple-600 text-white">
          <h3 className="text-lg font-semibold">Gradient Boosting</h3>
          <p className="text-purple-100 text-sm">Sequential ensemble learning</p>
        </div>
        <div className="p-6">
          <div className="grid lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-3">How It Works</h4>
              <p className="text-sm text-gray-600 mb-4">
                Gradient Boosting builds trees sequentially, where each new tree corrects the errors
                of the previous ensemble. It optimizes a loss function using gradient descent in function space.
              </p>

              <CodeBlock
                title="Gradient Boosting Algorithm"
                language="python"
                code={`# Gradient Boosting (HistGradientBoosting for speed)
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    learning_rate=0.1,     # Step size for updates
    max_iter=100,          # Number of boosting stages
    max_depth=None,        # Individual tree depth
    min_samples_leaf=20,   # Regularization
    l2_regularization=0.0, # L2 penalty on leaf values
    random_state=42
)

# Update rule (simplified):
# F_m(x) = F_{m-1}(x) + η * h_m(x)
# where h_m fits the negative gradient of loss
# η = learning_rate (controls step size)

# Lower learning_rate + more iterations =
# more stable but slower training`}
              />
            </div>

            <div>
              <h4 className="font-medium text-gray-900 mb-3">Key Hyperparameters & Fairness</h4>
              <div className="space-y-3">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="font-medium">learning_rate</p>
                  <p className="text-sm text-gray-500 mt-1">
                    Controls how much each tree contributes to the final model
                  </p>
                  <div className="mt-3 p-3 bg-purple-50 rounded border border-purple-100">
                    <p className="text-xs text-purple-800">
                      <strong>Fairness Impact:</strong> Very low learning_rate (0.01) with many iterations
                      creates a more stable model that's less likely to overfit to biased patterns.
                      Higher rates (0.3+) may amplify biases in early trees.
                    </p>
                  </div>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="font-medium">early_stopping</p>
                  <p className="text-sm text-gray-500 mt-1">
                    Stop training when validation performance stops improving
                  </p>
                  <div className="mt-3 p-3 bg-green-50 rounded border border-green-100">
                    <p className="text-xs text-green-800">
                      <strong>Fairness Impact:</strong> Early stopping prevents overfitting.
                      Monitor fairness metrics on validation set alongside accuracy to catch
                      when the model starts learning biased patterns.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Threshold Selection */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Decision Threshold Selection</h3>
        <p className="text-sm text-gray-600 mb-4">
          Models output probabilities. We must choose a threshold to convert them to binary decisions.
          This choice dramatically affects fairness metrics.
        </p>

        <CodeBlock
          title="Threshold Impact Analysis"
          language="python"
          code={`# Default threshold = 0.5
y_pred = (y_prob >= 0.5).astype(int)

# But what if we lower it?
# threshold = 0.3 means: predict positive if P(Y=1) >= 30%

def analyze_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    # Lower threshold → More positive predictions
    # → Higher recall (catch more true positives)
    # → Higher FPR (more false positives too)

    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'fpr': false_positive_rate(y_true, y_pred),
    }

# FAIRNESS IMPLICATION:
# If score distributions differ by race,
# a single threshold produces different
# selection rates per group.
#
# Solution: Group-specific thresholds
# (ThresholdOptimizer in fairlearn)`}
          explanation="The decision threshold converts predicted probabilities into binary decisions. A threshold of 0.3 means anyone with ≥30% predicted risk is classified as 'high risk'."
          impact="If Black defendants have generally higher predicted scores due to biased features, lowering the threshold will classify even more of them as high risk - amplifying disparity."
        />
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: RESULTS (COMPAS & NYPD)
// ============================================================================

function ResultsSection({ title, sectionId, data, charts, filterDataset }) {
  const filteredCharts = charts?.filter(c => c.dataset === filterDataset) || []

  return (
    <section id={sectionId} className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">{title}</h2>
        <p className="text-lg text-gray-600">
          {sectionId === 'compas'
            ? 'Comprehensive analysis of COMPAS recidivism prediction with fairness evaluation.'
            : 'Analysis of NYPD Stop-Question-Frisk patterns and predictive modeling.'}
        </p>
      </div>

      {/* Model Performance Table */}
      {data?.models && (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 bg-gray-50 border-b border-gray-100">
            <h3 className="font-semibold text-gray-900">Model Performance Comparison</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left font-medium text-gray-700">Model</th>
                  <th className="px-4 py-3 text-right font-medium text-gray-700">AUC</th>
                  <th className="px-4 py-3 text-right font-medium text-gray-700">Accuracy</th>
                  <th className="px-4 py-3 text-right font-medium text-gray-700">F1</th>
                  <th className="px-4 py-3 text-right font-medium text-gray-700">Brier</th>
                  <th className="px-4 py-3 text-right font-medium text-gray-700">DP Diff</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {Object.entries(data.models).map(([name, m]) => (
                  <tr key={name} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-medium">{name}</td>
                    <td className="px-4 py-3 text-right font-mono">{m.auc?.toFixed(3)}</td>
                    <td className="px-4 py-3 text-right font-mono">{m.accuracy?.toFixed(3)}</td>
                    <td className="px-4 py-3 text-right font-mono">{m.f1?.toFixed(3)}</td>
                    <td className="px-4 py-3 text-right font-mono">{m.brier_score?.toFixed(3)}</td>
                    <td className="px-4 py-3 text-right">
                      <span className={`font-mono ${m.fairness?.selection_rate_difference > 0.1 ? 'text-red-600' : 'text-green-600'}`}>
                        {m.fairness?.selection_rate_difference?.toFixed(3)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid lg:grid-cols-2 gap-6">
        {filteredCharts.map(chart => (
          <ChartCard key={chart.chart_id} chart={chart} showCode={true} />
        ))}
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: WHAT-IF ANALYSIS
// ============================================================================

function WhatIfSection() {
  const scenarios = [
    {
      scenario: "Remove 'Number_of_Priors' Feature",
      effect: "Prior convictions is the strongest predictor but also carries historical bias from differential policing.",
      beforeValue: "AUC: 0.738",
      afterValue: "AUC: ~0.65",
      code: `# Removing a biased proxy variable
features_no_priors = [f for f in features if f != 'Number_of_Priors']
X_reduced = df[features_no_priors]

# Retrain model
model.fit(X_reduced, y)

# Expected effects:
# - AUC drops ~10-15% (priors is highly predictive)
# - Demographic parity difference may DECREASE
# - Model relies more on age, charge type
# - Trade-off: Less accurate but potentially fairer`,
      interpretation: "Removing priors reduces accuracy but may reduce racial disparity since priors correlate with race due to historical over-policing of minority communities. This is a classic accuracy-fairness trade-off."
    },
    {
      scenario: "Lower Decision Threshold (0.5 → 0.3)",
      effect: "More people classified as 'high risk' - catches more true positives but also more false positives.",
      beforeValue: "FPR: 32%",
      afterValue: "FPR: ~48%",
      code: `# Threshold sensitivity analysis
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    # Lower threshold effects:
    # - Selection rate INCREASES (more positives)
    # - Recall INCREASES (catch more recidivists)
    # - Precision DECREASES (more false positives)
    # - FPR INCREASES for ALL groups

    # Fairness: If score distributions differ by race,
    # disparity in selection rates may increase or
    # decrease depending on distribution shapes`,
      interpretation: "Lower threshold means being more 'cautious' - flagging more people. This increases recall but also increases false positives, potentially affecting minorities disproportionately if their score distribution is shifted higher."
    },
    {
      scenario: "Apply L1 Regularization (Lasso)",
      effect: "L1 penalty drives some coefficients to exactly zero, performing automatic feature selection.",
      beforeValue: "Features: 12",
      afterValue: "Features: ~7",
      code: `# L1 Regularization (Lasso)
from sklearn.linear_model import LogisticRegression

model_l1 = LogisticRegression(
    penalty='l1',        # L1 norm penalty
    C=0.1,               # Strong regularization
    solver='saga',       # Required for L1
    random_state=42
)

model_l1.fit(X_train, y_train)

# Check which features were zeroed out
zero_features = [f for f, c in zip(features, model_l1.coef_[0])
                 if abs(c) < 0.001]

# L1 may zero out:
# - Weakly predictive features
# - Redundant features (correlated with others)
# - Potentially race-proxy features if properly tuned`,
      interpretation: "L1 regularization can remove features that are weakly predictive or redundant. With careful tuning, it might remove race-proxy features, but it could also remove legitimately predictive features. No guarantee of fairness improvement."
    },
    {
      scenario: "Use Balanced Class Weights",
      effect: "Upweights minority class (recidivists) to combat class imbalance.",
      beforeValue: "Recall: 62%",
      afterValue: "Recall: ~78%",
      code: `# Class weight balancing
model_balanced = LogisticRegression(
    class_weight='balanced',  # Auto-balance
    random_state=42
)

# Effect: weight_class_i = n_samples / (n_classes * n_samples_i)
# If 45% positive, 55% negative:
# weight_positive = 1.11
# weight_negative = 0.91

# This increases penalty for misclassifying positives
# → Higher recall, lower precision
# → FPR likely increases for both groups

# DANGER: May increase FPR disparity if
# negative class distribution differs by race`,
      interpretation: "Balanced weights help when you want to catch more true positives (recidivists). But the increased false positives may fall disproportionately on the minority group if the negative class has different characteristics across groups."
    },
    {
      scenario: "Train on 70% Majority Group Only",
      effect: "Simulates what happens when training data is dominated by one demographic.",
      beforeValue: "FPR Black: 45%",
      afterValue: "FPR Black: ~55%",
      code: `# Data imbalance simulation
# What if 70% of training data is African American?

df_majority = df[df['African_American'] == 1].sample(frac=0.7)
df_minority = df[df['African_American'] == 0].sample(frac=0.3)
df_imbalanced = pd.concat([df_majority, df_minority])

model.fit(df_imbalanced[features], df_imbalanced[target])

# Model learns patterns from majority group
# Calibration for minority group suffers
# Score of 0.6 may mean different things
# for different racial groups`,
      interpretation: "When training data is skewed toward one group, the model learns that group's patterns better. Predictions for underrepresented groups become less reliable, often showing worse calibration and higher error rates."
    },
    {
      scenario: "Apply Threshold Optimizer (Fairlearn)",
      effect: "Post-processing technique that finds group-specific thresholds to equalize error rates.",
      beforeValue: "EO Diff: 0.22",
      afterValue: "EO Diff: ~0.05",
      code: `# Fairlearn ThresholdOptimizer
from fairlearn.postprocessing import ThresholdOptimizer

# Wrap the trained model
postprocess = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds",  # Equal FPR and TPR
    prefit=True
)

# Fit finds optimal thresholds per group
postprocess.fit(X_train, y_train, sensitive_features=race_train)

# Predict uses group-specific thresholds
y_pred_fair = postprocess.predict(X_test, sensitive_features=race_test)

# Result: Different threshold for each group
# e.g., threshold_black=0.42, threshold_white=0.58
# This EQUALIZES error rates across groups`,
      interpretation: "ThresholdOptimizer achieves equalized odds by using different decision thresholds for different groups. This is controversial - it treats individuals differently based on group membership. Some argue this is unfair; others argue it corrects for biased scores."
    },
  ]

  return (
    <section id="what-if" className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">What-If Analysis</h2>
        <p className="text-lg text-gray-600">
          Explore how changing model parameters, features, or data affects both accuracy and fairness.
          These scenarios illustrate the trade-offs inherent in algorithmic fairness.
        </p>
      </div>

      <div className="p-5 bg-purple-50 border border-purple-200 rounded-xl">
        <div className="flex gap-4">
          <div className="text-purple-600">
            <Icons.Beaker />
          </div>
          <div>
            <p className="font-medium text-purple-900">Interactive Experimentation</p>
            <p className="text-sm text-purple-700 mt-1">
              Each scenario below shows what happens when you modify a specific aspect of the model or data.
              Click "Show technical details" to see the code and deeper analysis.
            </p>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {scenarios.map((s, idx) => (
          <WhatIfCard key={idx} {...s} />
        ))}
      </div>

      {/* Summary */}
      <div className="bg-gray-900 text-white rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">Key Takeaways from What-If Analysis</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-200 mb-2">No Free Lunch</h4>
            <p className="text-sm text-gray-400">
              Every intervention involves trade-offs. Removing biased features reduces disparity but hurts accuracy.
              Lowering thresholds catches more positives but increases false positives. There is no purely technical
              solution that satisfies all criteria.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-gray-200 mb-2">Values Matter</h4>
            <p className="text-sm text-gray-400">
              Choosing which trade-off to accept is a values question, not a technical one. Should we prioritize
              equal error rates (equalized odds) or equal selection rates (demographic parity)? The answer depends
              on what we believe is fair, not what the math says.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: COMPARISON
// ============================================================================

function ComparisonSection({ comparisonData, charts }) {
  const filteredCharts = charts?.filter(c => c.dataset === 'Comparison') || []

  return (
    <section id="comparison" className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Cross-Dataset Comparison</h2>
        <p className="text-lg text-gray-600">
          Comparing COMPAS and NYPD reveals how bias manifests differently across criminal justice stages.
        </p>
      </div>

      {/* Comparison Table */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-gray-700">Dimension</th>
                <th className="px-4 py-3 text-center font-medium text-blue-700">COMPAS</th>
                <th className="px-4 py-3 text-center font-medium text-green-700">NYPD SQF</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              <tr>
                <td className="px-4 py-3 font-medium">Purpose</td>
                <td className="px-4 py-3 text-center">Predict recidivism risk</td>
                <td className="px-4 py-3 text-center">Document police stops</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-medium">Selection Bias</td>
                <td className="px-4 py-3 text-center">People in criminal system</td>
                <td className="px-4 py-3 text-center">People police chose to stop</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-medium">Label Bias</td>
                <td className="px-4 py-3 text-center">Re-arrest ≠ Reoffense</td>
                <td className="px-4 py-3 text-center">Arrest ≠ Guilt</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-medium">Disparity Type</td>
                <td className="px-4 py-3 text-center">Higher FPR for Black defendants</td>
                <td className="px-4 py-3 text-center">Over-representation in stops</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-medium">Best Model AUC</td>
                <td className="px-4 py-3 text-center font-mono">0.738</td>
                <td className="px-4 py-3 text-center font-mono">0.856</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        {filteredCharts.map(chart => (
          <ChartCard key={chart.chart_id} chart={chart} showCode={true} />
        ))}
      </div>

      {/* Key Findings */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Cross-Dataset Findings</h3>
        <div className="space-y-4">
          {(comparisonData?.key_findings || []).map((finding, idx) => (
            <div key={idx} className="flex gap-4 p-4 bg-gray-50 rounded-lg">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center font-bold text-sm">
                {idx + 1}
              </div>
              <div>
                <p className="font-medium text-gray-900">{finding.finding}</p>
                <p className="text-sm text-gray-500 mt-1"><strong>Evidence:</strong> {finding.evidence}</p>
                <p className="text-sm text-blue-600 mt-1"><strong>Implication:</strong> {finding.implication}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: ETHICS & GOVERNANCE (EXPANDED)
// ============================================================================

function EthicsSection() {
  return (
    <section id="ethics" className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Ethics, Sociology & Governance</h2>
        <p className="text-lg text-gray-600">
          Applying the three analytical lenses from our course to evaluate AI in criminal justice.
        </p>
      </div>

      {/* Ethics Lens - Expanded */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-red-500 to-red-600 text-white">
          <h3 className="text-lg font-semibold">Ethics Lens</h3>
          <p className="text-red-100 text-sm">Moral evaluation of algorithmic systems</p>
        </div>
        <div className="p-6 space-y-6">
          {/* Consequentialism */}
          <div className="border-l-4 border-red-400 pl-4">
            <h4 className="font-semibold text-gray-900">Consequentialist Analysis (Utilitarianism)</h4>
            <p className="text-sm text-gray-600 mt-2">
              <strong>Framework:</strong> Actions are right if they maximize overall well-being (Bentham, Mill).
              We must weigh aggregate harms and benefits.
            </p>
            <div className="mt-3 grid md:grid-cols-2 gap-4">
              <div className="p-3 bg-green-50 rounded-lg">
                <p className="text-sm font-medium text-green-800">Claimed Benefits</p>
                <ul className="text-xs text-green-700 mt-1 space-y-1">
                  <li>• Efficient resource allocation</li>
                  <li>• Consistent decision-making</li>
                  <li>• Reduced human bias (disputed)</li>
                  <li>• Public safety optimization</li>
                </ul>
              </div>
              <div className="p-3 bg-red-50 rounded-lg">
                <p className="text-sm font-medium text-red-800">Documented Harms</p>
                <ul className="text-xs text-red-700 mt-1 space-y-1">
                  <li>• Disparate error rates by race</li>
                  <li>• Unjust detention of false positives</li>
                  <li>• Erosion of community trust</li>
                  <li>• Feedback loops amplifying bias</li>
                </ul>
              </div>
            </div>
            <p className="text-sm text-gray-600 mt-3">
              <strong>Verdict:</strong> The aggregate harm—particularly the systematic disadvantaging of minority
              communities through higher false positive rates—outweighs efficiency gains. A utilitarian analysis
              that properly weights the severity of wrongful incarceration finds these systems wanting.
            </p>
          </div>

          {/* Deontology */}
          <div className="border-l-4 border-blue-400 pl-4">
            <h4 className="font-semibold text-gray-900">Deontological Analysis (Kantian Ethics)</h4>
            <p className="text-sm text-gray-600 mt-2">
              <strong>Framework:</strong> Actions must conform to universal moral laws. The Categorical Imperative:
              "Act only according to maxims you could will to be universal laws" (Kant).
            </p>
            <div className="mt-3 p-3 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Key Principles Violated:</strong>
              </p>
              <ul className="text-sm text-blue-700 mt-2 space-y-2">
                <li>
                  <strong>1. Respect for Persons:</strong> Using statistical group membership to make individual
                  decisions treats people as means, not ends. Each person has a right to be judged on their own actions.
                </li>
                <li>
                  <strong>2. Universalizability:</strong> Would we accept a system that assigns risk based on
                  characteristics we were born with, if it applied to our own demographic group?
                </li>
                <li>
                  <strong>3. Autonomy:</strong> Opaque algorithmic decisions deny individuals the ability to
                  understand and contest the basis for decisions affecting them.
                </li>
              </ul>
            </div>
            <p className="text-sm text-gray-600 mt-3">
              <strong>Verdict:</strong> From a Kantian perspective, these systems are fundamentally unjust regardless
              of their accuracy, because they use morally irrelevant characteristics (race-correlated features) to
              make consequential decisions about individuals.
            </p>
          </div>

          {/* Virtue Ethics */}
          <div className="border-l-4 border-purple-400 pl-4">
            <h4 className="font-semibold text-gray-900">Virtue Ethics Analysis (Aristotelian)</h4>
            <p className="text-sm text-gray-600 mt-2">
              <strong>Framework:</strong> Focus on character and what a virtuous person or institution would do.
              Key virtues: justice, prudence, temperance, courage (Aristotle).
            </p>
            <div className="mt-3 p-3 bg-purple-50 rounded-lg">
              <p className="text-sm text-purple-800">
                <strong>Virtues at Stake:</strong>
              </p>
              <ul className="text-sm text-purple-700 mt-2 space-y-2">
                <li>
                  <strong>Justice (Dikaiosyne):</strong> A just institution gives each person their due.
                  Algorithmic systems that produce systematically unequal treatment fail this virtue.
                </li>
                <li>
                  <strong>Prudence (Phronesis):</strong> Practical wisdom requires context-sensitive judgment.
                  Automated scoring strips away the nuance that justice requires.
                </li>
                <li>
                  <strong>Temperance (Sophrosyne):</strong> Moderation and humility about our knowledge.
                  The false precision of risk scores (e.g., "7 out of 10") belies deep uncertainty.
                </li>
              </ul>
            </div>
            <p className="text-sm text-gray-600 mt-3">
              <strong>Verdict:</strong> A virtuous criminal justice system would cultivate institutions that
              embody fairness as a core value. These algorithmic systems, by encoding historical discrimination,
              fail to express the virtue of justice that should characterize legitimate institutions.
            </p>
          </div>
        </div>
      </div>

      {/* Sociology Lens - Expanded */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white">
          <h3 className="text-lg font-semibold">Sociology Lens</h3>
          <p className="text-blue-100 text-sm">Power structures and social impact</p>
        </div>
        <div className="p-6 space-y-6">
          {/* Power Dynamics */}
          <div className="border-l-4 border-blue-400 pl-4">
            <h4 className="font-semibold text-gray-900">Power Dynamics (Critical Theory)</h4>
            <p className="text-sm text-gray-600 mt-2">
              <strong>Framework:</strong> Frankfurt School critical theory examines how power structures maintain
              social inequality. Technology is not neutral—it embodies and enforces existing power relations (Marcuse).
            </p>
            <div className="mt-3 p-3 bg-blue-50 rounded-lg">
              <ul className="text-sm text-blue-800 space-y-2">
                <li>
                  <strong>Designers:</strong> Predominantly white, affluent technologists with no lived experience
                  of the criminal justice system's impact on minority communities.
                </li>
                <li>
                  <strong>Deployers:</strong> Courts and police departments with institutional interests in appearing
                  "objective" and "data-driven."
                </li>
                <li>
                  <strong>Subjects:</strong> Disproportionately low-income people of color who lack resources to
                  understand or challenge algorithmic decisions.
                </li>
              </ul>
            </div>
          </div>

          {/* Who Benefits/Harms */}
          <div className="border-l-4 border-green-400 pl-4">
            <h4 className="font-semibold text-gray-900">Distribution of Benefits and Harms</h4>
            <div className="mt-3 grid md:grid-cols-2 gap-4">
              <div className="p-3 bg-green-50 rounded-lg">
                <p className="text-sm font-medium text-green-800">Who Benefits</p>
                <ul className="text-xs text-green-700 mt-1 space-y-1">
                  <li>• Private companies selling risk tools (Northpointe/Equivant)</li>
                  <li>• Politicians claiming "evidence-based" policy</li>
                  <li>• Courts seeking efficiency and consistency</li>
                  <li>• Majority communities (lower false positive burden)</li>
                </ul>
              </div>
              <div className="p-3 bg-red-50 rounded-lg">
                <p className="text-sm font-medium text-red-800">Who Is Harmed</p>
                <ul className="text-xs text-red-700 mt-1 space-y-1">
                  <li>• Black and Hispanic defendants (higher FPR)</li>
                  <li>• Low-income defendants (can't afford experts to challenge)</li>
                  <li>• Minority communities (erosion of trust)</li>
                  <li>• Democratic accountability (opaque decisions)</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Systemic Effects */}
          <div className="border-l-4 border-yellow-400 pl-4">
            <h4 className="font-semibold text-gray-900">Systemic Effects (Structural Racism)</h4>
            <p className="text-sm text-gray-600 mt-2">
              <strong>Framework:</strong> Algorithmic systems don't create racism—they operationalize existing
              structural racism into automated decisions (Benjamin, 2019; Noble, 2018).
            </p>
            <CodeBlock
              title="The Feedback Loop of Structural Racism"
              language="text"
              code={`Historical Discrimination
        ↓
Over-policing of minority neighborhoods
        ↓
Higher arrest rates (not crime rates)
        ↓
Training data reflects enforcement, not behavior
        ↓
Model learns: minority neighborhoods = high risk
        ↓
Directs more police to minority neighborhoods
        ↓
More arrests in minority neighborhoods
        ↓
"Validates" the model → Repeat

This is what Ruha Benjamin calls the "New Jim Code"`}
            />
          </div>
        </div>
      </div>

      {/* Governance Lens - Expanded */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-green-500 to-green-600 text-white">
          <h3 className="text-lg font-semibold">Governance Lens</h3>
          <p className="text-green-100 text-sm">Regulatory frameworks and accountability</p>
        </div>
        <div className="p-6 space-y-6">
          {/* Current Regulatory Landscape */}
          <div className="border-l-4 border-green-400 pl-4">
            <h4 className="font-semibold text-gray-900">Current Regulatory Landscape</h4>
            <div className="mt-3 grid md:grid-cols-2 gap-4">
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm font-medium text-gray-800">United States</p>
                <ul className="text-xs text-gray-600 mt-1 space-y-1">
                  <li>• No federal AI regulation</li>
                  <li>• State-level patchwork (Illinois BIPA, NYC Local Law 144)</li>
                  <li>• Equal Protection Clause (14th Amendment) - rarely applied to algorithms</li>
                  <li>• Due Process challenges largely unsuccessful</li>
                </ul>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm font-medium text-gray-800">European Union</p>
                <ul className="text-xs text-gray-600 mt-1 space-y-1">
                  <li>• EU AI Act (2024) - risk-based framework</li>
                  <li>• Criminal justice AI = "High Risk" category</li>
                  <li>• Mandatory transparency, human oversight</li>
                  <li>• GDPR Article 22: Right to human review of automated decisions</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Accountability Gaps */}
          <div className="border-l-4 border-red-400 pl-4">
            <h4 className="font-semibold text-gray-900">Accountability Gaps</h4>
            <div className="mt-3 space-y-3">
              <div className="p-3 bg-red-50 rounded-lg">
                <p className="text-sm text-red-800">
                  <strong>Opacity:</strong> COMPAS's algorithm is proprietary. Defendants cannot examine how their
                  score was calculated. Wisconsin Supreme Court ruled this acceptable (State v. Loomis, 2016),
                  requiring only that judges be warned about limitations.
                </p>
              </div>
              <div className="p-3 bg-red-50 rounded-lg">
                <p className="text-sm text-red-800">
                  <strong>Liability Gap:</strong> When an algorithm causes harm, who is responsible?
                  The vendor (Northpointe)? The state that purchased it? The judge who relied on it?
                  No clear legal framework exists.
                </p>
              </div>
              <div className="p-3 bg-red-50 rounded-lg">
                <p className="text-sm text-red-800">
                  <strong>Democratic Deficit:</strong> Affected communities have no input into the design,
                  procurement, or deployment of systems that govern their lives. This is a fundamental
                  failure of democratic governance.
                </p>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="border-l-4 border-blue-400 pl-4">
            <h4 className="font-semibold text-gray-900">Policy Recommendations</h4>
            <div className="mt-3 grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm font-medium text-blue-800">1. Mandatory Bias Audits</p>
                  <p className="text-xs text-blue-600">Independent testing for disparate impact before deployment</p>
                </div>
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm font-medium text-blue-800">2. Algorithmic Impact Assessments</p>
                  <p className="text-xs text-blue-600">Like environmental impact assessments for AI systems</p>
                </div>
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm font-medium text-blue-800">3. Right to Explanation</p>
                  <p className="text-xs text-blue-600">Meaningful explanation of factors affecting individual scores</p>
                </div>
              </div>
              <div className="space-y-2">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm font-medium text-blue-800">4. Community Oversight Boards</p>
                  <p className="text-xs text-blue-600">Affected communities in governance decisions</p>
                </div>
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm font-medium text-blue-800">5. Vendor Liability Framework</p>
                  <p className="text-xs text-blue-600">Clear legal responsibility for algorithmic harms</p>
                </div>
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm font-medium text-blue-800">6. Sunset Clauses</p>
                  <p className="text-xs text-blue-600">Regular re-evaluation and sunset provisions for AI systems</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: LIMITATIONS
// ============================================================================

function LimitationsSection() {
  return (
    <section id="limitations" className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Limitations</h2>
        <p className="text-lg text-gray-600">
          Critical limitations of our analysis and the underlying data.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Data Limitations */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
              <Icons.Warning />
            </span>
            Data Limitations
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-red-50 rounded-lg border border-red-100">
              <p className="font-medium text-red-800">Selection Bias</p>
              <p className="text-sm text-red-700 mt-1">
                Both datasets contain only people who entered the system—arrested or stopped.
                We cannot observe the counterfactual: people who would have been stopped/arrested
                if they were a different race. This fundamentally limits causal claims.
              </p>
            </div>
            <div className="p-4 bg-red-50 rounded-lg border border-red-100">
              <p className="font-medium text-red-800">Label Bias</p>
              <p className="text-sm text-red-700 mt-1">
                "Recidivism" means re-arrest, not reoffending. "Arrest" doesn't mean guilt.
                These labels reflect enforcement decisions, which are themselves biased.
                We're predicting system contact, not criminal behavior.
              </p>
            </div>
            <div className="p-4 bg-red-50 rounded-lg border border-red-100">
              <p className="font-medium text-red-800">Historical Context</p>
              <p className="text-sm text-red-700 mt-1">
                NYPD data is from 2012 (peak SQF). COMPAS data is from Broward County, Florida.
                Patterns may not generalize to other times, places, or policy contexts.
              </p>
            </div>
          </div>
        </div>

        {/* Methodological Limitations */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span className="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center">
              <Icons.Warning />
            </span>
            Methodological Limitations
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-100">
              <p className="font-medium text-yellow-800">Omitted Variables</p>
              <p className="text-sm text-yellow-700 mt-1">
                We cannot control for socioeconomic status, employment, education, neighborhood
                effects, or family circumstances. Race may be confounded with these factors,
                making causal interpretation difficult.
              </p>
            </div>
            <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-100">
              <p className="font-medium text-yellow-800">Fairness Metric Choice</p>
              <p className="text-sm text-yellow-700 mt-1">
                We report demographic parity, equalized odds, and calibration—but many other
                fairness definitions exist. Different metrics often conflict. Our choice
                reflects values, not objective truth.
              </p>
            </div>
            <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-100">
              <p className="font-medium text-yellow-800">Model Simplification</p>
              <p className="text-sm text-yellow-700 mt-1">
                The actual COMPAS algorithm uses 137 features; we use ProPublica's 12-feature
                public subset. Our models may not fully replicate the behavior of deployed systems.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Critical Epistemological Note */}
      <div className="bg-gray-900 text-white rounded-xl p-6">
        <h3 className="font-semibold mb-3">Critical Epistemological Note</h3>
        <p className="text-gray-300 text-sm leading-relaxed">
          <strong className="text-white">We can only measure disparities in system outcomes, not disparities in behavior.</strong>
          {' '}If the criminal justice system is biased at every stage—policing, prosecution, sentencing—then
          "recidivism" is a biased proxy for "reoffending." Even a model with equal error rates could perpetuate
          injustice if the underlying labels are biased. This is a fundamental limitation that no technical
          fix can address without broader institutional change.
        </p>
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: REFERENCES
// ============================================================================

function ReferencesSection() {
  const references = [
    {
      authors: "Angwin, J., Larson, J., Mattu, S., & Kirchner, L.",
      year: "2016",
      title: "Machine Bias",
      source: "ProPublica",
      type: "Investigation",
    },
    {
      authors: "Chouldechova, A.",
      year: "2017",
      title: "Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments",
      source: "Big Data, 5(2), 153-163",
      type: "Academic",
    },
    {
      authors: "Kleinberg, J., Mullainathan, S., & Raghavan, M.",
      year: "2016",
      title: "Inherent Trade-Offs in the Fair Determination of Risk Scores",
      source: "arXiv:1609.05807",
      type: "Academic",
    },
    {
      authors: "Benjamin, R.",
      year: "2019",
      title: "Race After Technology: Abolitionist Tools for the New Jim Code",
      source: "Polity Press",
      type: "Book",
    },
    {
      authors: "Noble, S. U.",
      year: "2018",
      title: "Algorithms of Oppression: How Search Engines Reinforce Racism",
      source: "NYU Press",
      type: "Book",
    },
    {
      authors: "Corbett-Davies, S., & Goel, S.",
      year: "2018",
      title: "The Measure and Mismeasure of Fairness: A Critical Review of Fair Machine Learning",
      source: "arXiv:1808.00023",
      type: "Academic",
    },
    {
      authors: "Floyd v. City of New York",
      year: "2013",
      title: "959 F. Supp. 2d 540 (S.D.N.Y. 2013)",
      source: "U.S. District Court",
      type: "Legal",
    },
    {
      authors: "State v. Loomis",
      year: "2016",
      title: "881 N.W.2d 749 (Wis. 2016)",
      source: "Wisconsin Supreme Court",
      type: "Legal",
    },
  ]

  return (
    <section id="references" className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">References</h2>
        <p className="text-lg text-gray-600">
          Academic, legal, and investigative sources informing this analysis.
        </p>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="space-y-4">
          {references.map((ref, idx) => (
            <div key={idx} className="flex gap-4 p-4 bg-gray-50 rounded-lg">
              <span className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center font-bold text-sm">
                {idx + 1}
              </span>
              <div className="flex-1">
                <p className="text-sm text-gray-900">
                  <span className="font-medium">{ref.authors}</span> ({ref.year}).
                  <em className="ml-1">{ref.title}</em>. {ref.source}.
                </p>
              </div>
              <span className={`text-xs px-2 py-1 rounded-full ${
                ref.type === 'Academic' ? 'bg-blue-100 text-blue-700' :
                ref.type === 'Legal' ? 'bg-purple-100 text-purple-700' :
                ref.type === 'Book' ? 'bg-green-100 text-green-700' :
                'bg-yellow-100 text-yellow-700'
              }`}>
                {ref.type}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: ARTIFACTS
// ============================================================================

function ArtifactsSection({ charts }) {
  return (
    <section id="artifacts" className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Download Artifacts</h2>
        <p className="text-lg text-gray-600">
          All analysis outputs available for download.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* JSON Files */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Data Files (JSON)</h3>
          <div className="space-y-3">
            {[
              { name: 'dashboard_summary.json', desc: 'Overview and key findings' },
              { name: 'compas_analysis.json', desc: 'COMPAS analysis results' },
              { name: 'nypd_analysis.json', desc: 'NYPD SQF analysis results' },
              { name: 'comparison_analysis.json', desc: 'Cross-dataset comparison' },
              { name: 'charts_metadata.json', desc: 'Chart descriptions' },
            ].map(file => (
              <a
                key={file.name}
                href={`/data/${file.name}`}
                download
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div>
                  <p className="font-medium text-gray-900 text-sm">{file.name}</p>
                  <p className="text-xs text-gray-500">{file.desc}</p>
                </div>
                <Icons.Download />
              </a>
            ))}
          </div>
        </div>

        {/* Context Paper */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Context Paper</h3>
          <a
            href="/1906.04711v3.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="block p-4 bg-blue-50 rounded-lg border border-blue-100 hover:bg-blue-100 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center text-white font-bold">
                PDF
              </div>
              <div>
                <p className="font-medium text-blue-900">Fairness Impossibility Theorems</p>
                <p className="text-sm text-blue-700">Background on fairness metrics trade-offs</p>
              </div>
            </div>
          </a>

          <h3 className="font-semibold text-gray-900 mb-4 mt-6">Chart Images</h3>
          <div className="grid grid-cols-3 gap-2">
            {(charts || []).slice(0, 9).map(chart => (
              <a
                key={chart.chart_id}
                href={`/img/${chart.filename}`}
                download
                className="aspect-video bg-gray-100 rounded overflow-hidden hover:opacity-80 transition-opacity"
              >
                <img
                  src={`/img/${chart.filename}`}
                  alt={chart.title}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
              </a>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Click any chart to download • {charts?.length || 0} total
          </p>
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// LANDING PAGE
// ============================================================================

function LandingPage({ onGetStarted }) {
  return (
    <div className="min-h-screen bg-white flex flex-col items-center justify-center px-4 py-12">
      {/* Course badge */}
      <p className="text-sm text-gray-400 mb-8 tracking-wide">
        MAIB AI 219 - Ethics, Sociology & Governance of AI
      </p>

      {/* Main title */}
      <h1 className="text-4xl sm:text-5xl md:text-6xl font-light text-center mb-6 text-gray-900">
        Criminal Justice
        <br />
        <span className="font-semibold">AI Bias Analysis</span>
      </h1>

      {/* Subtitle */}
      <p className="text-lg text-gray-500 text-center max-w-xl mb-12">
        Examining algorithmic fairness in COMPAS and NYPD Stop-Question-Frisk
      </p>

      {/* Get Started button */}
      <button
        onClick={onGetStarted}
        className="px-8 py-3 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 transition-colors flex items-center gap-2"
      >
        Get Started
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
        </svg>
      </button>

      {/* Divider */}
      <div className="w-16 h-px bg-gray-200 my-12"></div>

      {/* Group info */}
      <p className="text-xs text-gray-400 uppercase tracking-widest mb-6">Group 2</p>
      <div className="flex flex-wrap justify-center gap-8">
        {[
          { name: 'Vidit', role: 'Overview & Methods' },
          { name: 'Kaleemulla', role: 'COMPAS & NYPD' },
          { name: 'Ronaldo', role: 'What-If & Compare' },
          { name: 'Vishal', role: 'Ethics & Limits' },
        ].map((member) => (
          <div key={member.name} className="text-center">
            <p className="text-gray-900 font-medium">{member.name}</p>
            <p className="text-xs text-gray-400 mt-1">{member.role}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ============================================================================
// DASHBOARD (Previous Main Content)
// ============================================================================

function Dashboard({ activeSection, setActiveSection }) {
  const { data: dashboardData, error: errorDashboard } = useData('dashboard_summary.json')
  const { data: compasData } = useData('compas_analysis.json')
  const { data: nypdData } = useData('nypd_analysis.json')
  const { data: comparisonData } = useData('comparison_analysis.json')
  const { data: chartsData } = useData('charts_metadata.json')

  if (errorDashboard) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-xl shadow-lg p-8 text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Icons.Warning />
          </div>
          <h1 className="text-xl font-bold text-gray-900 mb-2">Analysis Data Not Found</h1>
          <p className="text-gray-600 mb-6">
            Run the Python analysis script first.
          </p>
          <div className="bg-gray-100 rounded-lg p-4 text-left">
            <code className="text-sm text-gray-700">python scripts/generate_reports.py</code>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header activeSection={activeSection} onNavigate={setActiveSection} />

      <main className="pt-16 pb-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-20">
          <OverviewSection dashboardData={dashboardData} />
          <DatasetsSection compasData={compasData} nypdData={nypdData} />
          <MethodologySection />
          <ModelExplainerSection />
          <ResultsSection
            title="COMPAS Analysis Results"
            sectionId="compas"
            data={compasData}
            charts={chartsData}
            filterDataset="COMPAS"
          />
          <ResultsSection
            title="NYPD Stop-Question-Frisk Results"
            sectionId="nypd"
            data={nypdData}
            charts={chartsData}
            filterDataset="NYPD SQF"
          />
          <WhatIfSection />
          <ComparisonSection comparisonData={comparisonData} charts={chartsData} />
          <EthicsSection />
          <LimitationsSection />
          <ReferencesSection />
          <ArtifactsSection charts={chartsData} />

          {/* Footer */}
          <footer className="border-t border-gray-200 pt-8 text-center">
            <p className="text-sm text-gray-600 font-medium">Criminal Justice AI Bias Analysis</p>
            <p className="text-sm text-gray-500 mt-1">Group 2: Vidit, Ronaldo, Kaleemulla, Vishal</p>
            <p className="text-sm text-gray-400">MAIB AI 219 - Ethics, Sociology, and Governance of AI</p>
          </footer>
        </div>
      </main>
    </div>
  )
}

// ============================================================================
// MAIN APP
// ============================================================================

function App() {
  const [showDashboard, setShowDashboard] = useState(false)
  const [activeSection, setActiveSection] = useState('overview')

  if (!showDashboard) {
    return <LandingPage onGetStarted={() => setShowDashboard(true)} />
  }

  return <Dashboard activeSection={activeSection} setActiveSection={setActiveSection} />
}

export default App
