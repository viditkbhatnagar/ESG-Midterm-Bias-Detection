# Speaker Notes: Criminal Justice AI Bias Analysis

**Group 2: Vidit, Ronaldo, Kaleemulla, Vishal**
**Course: MAIB AI 219 - Ethics, Sociology, and Governance of AI**

---

# VIDIT: Overview, Methods, Data, Models (~2-2.5 min)

## Section 1: Overview

### Opening Script
> "Good morning/afternoon. Our presentation analyzes algorithmic bias in criminal justice AI systems. We're examining two landmark cases: COMPAS, a recidivism prediction tool, and NYPD's Stop-Question-Frisk program. These systems affect millions of lives and have been at the center of debates about fairness in AI."

### Key Points to Hit
- AI in criminal justice isn't new - COMPAS has been used since 1998
- ProPublica's 2016 investigation brought bias into public consciousness
- These systems make high-stakes decisions: bail, sentencing, parole
- Our analysis applies technical ML methods AND the three course lenses

---

## Section 2: Datasets

### COMPAS Dataset Script
> "The COMPAS dataset comes from ProPublica's investigation. It contains over 6,000 defendants from Broward County, Florida. The target variable is whether someone recidivated within two years. Importantly, race is encoded as binary features - you're either African American, Hispanic, or the reference category White."

### NYPD Stop-Question-Frisk Script
> "The NYPD dataset is from 2012 - the peak year with over 500,000 stops. This program was ruled unconstitutional in Floyd v. City of New York for racial profiling. A critical point: this data only contains people who were STOPPED. We cannot see people who weren't stopped but would have been if they were a different race. This is called SELECTION BIAS."

---

## Section 3: Methodology

### Script
> "We trained three different model types to compare how different algorithms perform on biased data. We used a 70-30 stratified train-test split, meaning we preserve the class distribution in both sets. Let me explain the key technical concepts..."

---

## TECHNICAL JARGON DICTIONARY FOR VIDIT

### 1. AUC (Area Under the ROC Curve)

**Definition:**
AUC measures the model's ability to distinguish between classes. It represents the probability that a randomly chosen positive example ranks higher than a randomly chosen negative example.

**Range:** 0 to 1
- 0.5 = Random guessing (coin flip)
- 0.7-0.8 = Acceptable
- 0.8-0.9 = Good
- 0.9+ = Excellent

**What it tells us:**
How well the model RANKS predictions, regardless of the threshold used.

**How to increase AUC:**
- Add more informative features
- Use more complex models (ensemble methods)
- Better feature engineering
- More training data
- Handle class imbalance (oversampling, SMOTE)

**How to decrease AUC:**
- Remove predictive features
- Add noise to data
- Use simpler models
- Reduce training data

**Impact on algorithm/code:**
```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred_proba)
# y_pred_proba = probability scores, NOT binary predictions
# Higher AUC = model better separates classes
```

**Bias implications:**
A model can have high AUC overall but very different AUC for different racial groups. We need to check AUC BY SUBGROUP.

---

### 2. ROC Curve (Receiver Operating Characteristic)

**Definition:**
A plot showing the trade-off between True Positive Rate (sensitivity) and False Positive Rate at various threshold settings.

**Components:**
- X-axis: False Positive Rate (FPR) = FP / (FP + TN)
- Y-axis: True Positive Rate (TPR) = TP / (TP + FN)
- Diagonal line = random classifier

**What it tells us:**
How the model performs across ALL possible decision thresholds.

**How to read it:**
- Curve hugging top-left corner = good model
- Curve close to diagonal = poor model
- Each point on curve = one threshold value

**Code:**
```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
# thresholds array shows which probability cutoff gives each FPR/TPR pair
```

**Why it matters for fairness:**
Different groups may have different ROC curves. A threshold that's optimal for one group may be unfair to another.

---

### 3. Precision

**Definition:**
Of all the people we PREDICTED as positive (high risk), what fraction actually were positive?

**Formula:**
Precision = TP / (TP + FP)

**Range:** 0 to 1 (higher is better)

**What it tells us:**
How much we can TRUST a positive prediction.

**When to prioritize precision:**
- When false positives are costly
- When you want to be confident about positive predictions
- Example: If we predict "high risk" and they're detained, we want to be RIGHT

**How to increase precision:**
- Raise the decision threshold (be more conservative about predicting positive)
- Reduce false positives
- Use features that better identify true positives

**Trade-off:**
Increasing precision typically DECREASES recall (you'll miss more true positives)

**Code:**
```python
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)
# y_pred = binary predictions (0 or 1)
```

**Bias implications:**
If precision differs by race, we're making more mistakes for some groups than others.

---

### 4. Recall (Sensitivity, True Positive Rate)

**Definition:**
Of all the people who ARE actually positive (did recidivate), what fraction did we correctly identify?

**Formula:**
Recall = TP / (TP + FN)

**Range:** 0 to 1 (higher is better)

**What it tells us:**
How good we are at FINDING all the positive cases.

**When to prioritize recall:**
- When false negatives are costly
- When missing a positive case is dangerous
- Example: In medical screening, missing a cancer diagnosis is very costly

**How to increase recall:**
- Lower the decision threshold (predict positive more often)
- Add features that help identify positives
- Use ensemble methods

**Trade-off:**
Increasing recall typically DECREASES precision (more false positives)

**Code:**
```python
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)
```

**Bias implications:**
Equal recall across groups = "Equal Opportunity" - one definition of fairness.

---

### 5. F1 Score

**Definition:**
The harmonic mean of precision and recall. Balances both metrics.

**Formula:**
F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Range:** 0 to 1 (higher is better)

**Why harmonic mean (not arithmetic)?:**
Harmonic mean penalizes extreme imbalances. If precision is 1.0 and recall is 0.1, arithmetic mean = 0.55, but harmonic mean (F1) = 0.18. This better reflects poor performance.

**When to use F1:**
- When you need to balance precision and recall
- When class distribution is imbalanced
- When both false positives and false negatives matter

**Code:**
```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
```

---

### 6. Accuracy

**Definition:**
The fraction of all predictions that were correct.

**Formula:**
Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Range:** 0 to 1

**THE ACCURACY TRAP - IMPORTANT:**
Accuracy is MISLEADING with imbalanced classes!

**Example:**
If 95% of people don't recidivate, a model that predicts "no recidivism" for everyone gets 95% accuracy but is USELESS.

**When accuracy is useful:**
- Balanced classes (roughly 50-50)
- All errors are equally costly

**When accuracy is misleading:**
- Imbalanced classes
- Different error costs
- Fairness analysis (accuracy can be equal while error TYPES differ)

**Code:**
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
```

---

### 7. Brier Score

**Definition:**
Measures the mean squared error between predicted probabilities and actual outcomes. Tests CALIBRATION.

**Formula:**
Brier = (1/N) × Σ(predicted_probability - actual_outcome)²

**Range:** 0 to 1 (LOWER is better - opposite of other metrics!)

**What it tells us:**
How well-calibrated are the probability estimates? When we say "70% risk," do 70% of those people actually recidivate?

**Why calibration matters:**
- Judges interpret scores as probabilities
- "Score of 7" should mean roughly 70% chance
- Miscalibrated scores mislead decision-makers

**How to improve Brier Score:**
- Calibration techniques (Platt scaling, isotonic regression)
- Better probability estimation models
- Logistic regression often well-calibrated by default

**Code:**
```python
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_true, y_pred_proba)
```

**Bias implications:**
A model can be calibrated overall but MIS-calibrated for specific groups. This was ProPublica's main finding about COMPAS.

---

### 8. Confusion Matrix

**Definition:**
A 2x2 table showing all four possible prediction outcomes.

**Components:**
```
                    Actual Positive    Actual Negative
Predicted Positive       TP                 FP
Predicted Negative       FN                 TN
```

- **TP (True Positive):** Predicted high risk, actually recidivated
- **TN (True Negative):** Predicted low risk, didn't recidivate
- **FP (False Positive):** Predicted high risk, DIDN'T recidivate (WRONGLY DETAINED)
- **FN (False Negative):** Predicted low risk, DID recidivate (WRONGLY RELEASED)

**Why it matters for fairness:**
FP for Black defendants was 44.9%, for White defendants was 23.5%. This is the core ProPublica finding.

**Code:**
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
# Returns [[TN, FP], [FN, TP]]
```

---

### 9. False Positive Rate (FPR)

**Definition:**
Of all people who are actually NEGATIVE (won't recidivate), what fraction did we wrongly predict as positive?

**Formula:**
FPR = FP / (FP + TN) = FP / (Actual Negatives)

**Range:** 0 to 1 (LOWER is better)

**Why it matters in criminal justice:**
A false positive means an INNOCENT person is labeled high-risk. They may be:
- Denied bail
- Given longer sentence
- Denied parole

**The ProPublica finding:**
FPR for Black defendants: ~45%
FPR for White defendants: ~24%
Black defendants nearly TWICE as likely to be wrongly labeled high-risk.

**How to reduce FPR:**
- Raise decision threshold
- But this may increase FNR (miss actual recidivists)

---

### 10. False Negative Rate (FNR)

**Definition:**
Of all people who ARE actually positive (will recidivate), what fraction did we wrongly predict as negative?

**Formula:**
FNR = FN / (FN + TP) = FN / (Actual Positives)

**Range:** 0 to 1 (LOWER is better)

**Why it matters:**
A false negative means a person who WILL recidivate is labeled low-risk. They may be:
- Released when they pose genuine risk
- Given shorter sentence

**The trade-off:**
You can't minimize BOTH FPR and FNR simultaneously (with different base rates). This is Chouldechova's Impossibility Theorem.

---

### 11. Stratified Train-Test Split

**Definition:**
Dividing data into training and testing sets while PRESERVING the class distribution in both sets.

**Why stratify?**
```python
# Without stratification:
# Training set might have 45% positive
# Test set might have 52% positive
# Performance estimates will be wrong!

# With stratification:
# Both sets have same ~45.5% positive rate
```

**Code:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 30% for testing
    stratify=y,         # THIS preserves class balance
    random_state=42     # Reproducibility
)
```

**Why 70-30 split?**
- 70% gives enough data to train
- 30% gives enough data to estimate performance reliably
- Common convention, but can vary (80-20, etc.)

---

### 12. Logistic Regression

**Definition:**
A linear model that predicts probability of binary outcome using the logistic (sigmoid) function.

**How it works:**
1. Compute weighted sum: z = w₀ + w₁x₁ + w₂x₂ + ...
2. Apply sigmoid: P(y=1) = 1 / (1 + e^(-z))
3. Output is probability between 0 and 1

**Strengths:**
- Interpretable (coefficients show feature importance)
- Fast to train
- Well-calibrated probabilities
- Works well for linearly separable problems

**Weaknesses:**
- Assumes linear relationship between features and log-odds
- Can't capture complex interactions without feature engineering

**Code:**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Coefficients: model.coef_
# Intercept: model.intercept_
```

**Interpretation:**
If coefficient for "Number_of_Priors" is 0.5, then each additional prior conviction increases the LOG-ODDS of recidivism by 0.5.

---

### 13. Random Forest

**Definition:**
An ensemble of decision trees, each trained on random subsets of data and features. Final prediction is majority vote (classification) or average (regression).

**How it works:**
1. Create N decision trees (e.g., 100)
2. Each tree trained on bootstrap sample (random subset with replacement)
3. Each split considers random subset of features
4. Aggregate predictions across all trees

**Strengths:**
- Handles non-linear relationships
- Robust to outliers
- Feature importance built-in
- Reduces overfitting vs single tree

**Weaknesses:**
- Less interpretable than logistic regression
- Can be slow with many trees
- May still overfit with very deep trees

**Code:**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Feature importance: model.feature_importances_
```

**Why "random"?**
- Random sampling of data (bootstrap)
- Random sampling of features at each split
- This randomness reduces variance and overfitting

---

### 14. Gradient Boosting

**Definition:**
Builds trees sequentially, where each new tree tries to correct the errors of the previous trees.

**How it works:**
1. Train first tree on original data
2. Calculate residuals (errors)
3. Train second tree to predict residuals
4. Add second tree's predictions to first tree
5. Repeat, gradually reducing errors

**Strengths:**
- Often highest accuracy
- Handles complex patterns
- Can optimize various loss functions

**Weaknesses:**
- Prone to overfitting (need careful tuning)
- Slower to train than Random Forest
- Less interpretable

**Code:**
```python
from sklearn.ensemble import HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)
```

**Why HistGradientBoosting?**
- Faster than regular GradientBoosting
- Handles missing values natively
- Better memory efficiency

---

### 15. Demographic Parity (Statistical Parity)

**Definition:**
A fairness metric requiring that the SELECTION RATE (rate of positive predictions) be equal across groups.

**Formula:**
P(Ŷ=1 | A=a) = P(Ŷ=1 | A=b) for all groups a, b

**In plain English:**
"The same percentage of Black and White defendants should be labeled high-risk."

**Demographic Parity Difference:**
DPD = |Selection Rate (Group A) - Selection Rate (Group B)|
Goal: DPD close to 0

**Problems with this metric:**
- Ignores actual base rates
- If Group A actually has higher recidivism, equal selection rates mean we're either over-predicting Group B or under-predicting Group A
- Doesn't ensure ACCURATE predictions, just EQUAL predictions

**Code (Fairlearn):**
```python
from fairlearn.metrics import demographic_parity_difference
dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=race)
```

---

### 16. Equalized Odds

**Definition:**
A fairness metric requiring equal True Positive Rate AND equal False Positive Rate across groups.

**Formula:**
P(Ŷ=1 | Y=1, A=a) = P(Ŷ=1 | Y=1, A=b)  [Equal TPR]
P(Ŷ=1 | Y=0, A=a) = P(Ŷ=1 | Y=0, A=b)  [Equal FPR]

**In plain English:**
"Among people who WILL recidivate, equal % should be correctly identified across races. Among people who WON'T recidivate, equal % should be wrongly labeled high-risk across races."

**Why it's better than Demographic Parity:**
It conditions on the ACTUAL outcome, not just the prediction.

**Equalized Odds Difference:**
```python
from fairlearn.metrics import equalized_odds_difference
eod = equalized_odds_difference(y_true, y_pred, sensitive_features=race)
```

**The impossibility:**
You CANNOT have equal FPR, equal FNR, AND equal calibration simultaneously when base rates differ (Chouldechova, 2017).

---

### 17. Calibration

**Definition:**
A model is calibrated if when it predicts 70% probability, roughly 70% of those cases are actually positive.

**Perfect calibration:**
Calibration curve is a 45-degree diagonal line.

**How to check:**
1. Bin predictions (0-10%, 10-20%, etc.)
2. For each bin, calculate actual positive rate
3. Plot predicted vs actual

**Code:**
```python
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
# Plot prob_pred (x) vs prob_true (y)
```

**Why calibration matters for fairness:**
A score of "7" should mean the same thing for a Black defendant as for a White defendant. If the model is calibrated overall but NOT calibrated within groups, it's misleading decision-makers.

---

### 18. Threshold (Decision Boundary)

**Definition:**
The probability cutoff above which we classify as positive.

**Default:** 0.5 (predict positive if P > 0.5)

**How changing threshold affects metrics:**

| Lower Threshold (e.g., 0.3) | Higher Threshold (e.g., 0.7) |
|----------------------------|------------------------------|
| More positive predictions | Fewer positive predictions |
| Higher recall (catch more true positives) | Lower recall |
| Lower precision (more false positives) | Higher precision |
| Higher FPR | Lower FPR |
| Lower FNR | Higher FNR |

**Threshold optimization for fairness:**
```python
from fairlearn.postprocessing import ThresholdOptimizer
postprocess = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds"
)
postprocess.fit(X_train, y_train, sensitive_features=race_train)
# This finds DIFFERENT thresholds for each group to equalize error rates
```

---

### 19. One-Hot Encoding

**Definition:**
Converting categorical variables into binary (0/1) columns.

**Example:**
```
Original: Race = "Black", "White", "Hispanic"

One-Hot Encoded:
African_American | White | Hispanic
      1          |   0   |    0     (Black person)
      0          |   1   |    0     (White person)
      0          |   0   |    1     (Hispanic person)
```

**Why it's done:**
- ML algorithms need numerical input
- Can't use "Black"=1, "White"=2, "Hispanic"=3 because that implies ordering

**The reference category trap:**
In COMPAS, "White" is the IMPLICIT reference (all race columns = 0). This means the model's intercept is calibrated to White defendants.

---

### 20. Base Rate

**Definition:**
The overall rate of the positive class in the population.

**COMPAS base rate:** ~45.5% recidivism
**NYPD base rate:** ~6% arrest rate

**Why base rates matter for fairness:**

If Group A has 50% base rate and Group B has 30% base rate:
- Perfect calibration means different positive prediction rates
- Equal positive prediction rates means miscalibration
- You can't have both!

**Chouldechova's Impossibility Theorem:**
When base rates differ between groups, you CANNOT simultaneously achieve:
1. Equal FPR
2. Equal FNR
3. Equal calibration (predictive parity)

This is a MATHEMATICAL impossibility, not a technical limitation.

---

## Section 4: Models (Summary Script)

> "We compared three model families. Logistic Regression is our interpretable baseline - we can see exactly how each feature affects the prediction. Random Forest is an ensemble that handles non-linear patterns. Gradient Boosting often achieves the highest accuracy but is harder to interpret."

> "Key finding: ALL three models show similar fairness problems. This tells us the bias is in the DATA, not the algorithm choice. You can't fix biased data with a better algorithm."

---

# KALEEMULLA: COMPAS Analysis, NYPD Analysis (~2-2.5 min)

## Section 5: COMPAS Analysis

### Opening Script
> "Now let's dive into the COMPAS analysis. COMPAS stands for Correctional Offender Management Profiling for Alternative Sanctions. It's been used since 1998 to predict recidivism risk and inform bail, sentencing, and parole decisions."

### Key Findings to Present

**1. Recidivism Rate by Race Chart**
> "This chart shows raw recidivism rates. African Americans in this dataset have a 52% recidivism rate compared to 39% for White defendants. But here's the critical question: Is this difference because of actual behavior, or because of biased enforcement and charging practices?"

**2. False Positive Rate Disparity**
> "This is the ProPublica finding that made headlines. Among defendants who DID NOT recidivate, 44.9% of Black defendants were wrongly labeled high-risk, compared to only 23.5% of White defendants. That's nearly DOUBLE the false positive rate."

> "In human terms: If you're Black and you're NOT going to reoffend, you're twice as likely to be wrongly labeled dangerous."

**3. False Negative Rate (The Other Side)**
> "Interestingly, White defendants have HIGHER false negative rates - meaning actual recidivists are MORE likely to be labeled low-risk if they're White. The errors go in opposite directions for different groups."

**4. Model Performance Comparison**
> "All three of our models achieve similar AUC around 0.70-0.72. This is typical for recidivism prediction - it's a hard problem. But notice: accuracy looks similar across models, but fairness metrics vary."

**5. Calibration Curves**
> "This chart shows calibration BY RACE. A well-calibrated model should follow the diagonal. You can see the curves diverge - the model means different things for different groups."

### Bias Mitigation Script
> "We applied Fairlearn's ThresholdOptimizer to equalize odds across groups. The result: we CAN reduce the fairness gap, but at the cost of overall accuracy. This is the fundamental trade-off - there's no free lunch in fairness."

---

## Section 6: NYPD Stop-Question-Frisk Analysis

### Opening Script
> "The NYPD analysis is different - we're looking at policing INPUTS rather than sentencing OUTPUTS. In 2012, NYPD made over 500,000 stops. The vast majority - over 88% - resulted in NO arrest or summons. This was ruled unconstitutional racial profiling in Floyd v. City of New York."

### Key Findings to Present

**1. Stops by Race Chart**
> "Look at this distribution. Over 50% of stops were Black individuals, about 30% Hispanic, and under 15% White. But NYC's population is roughly 33% White, 29% Hispanic, 26% Black. Black New Yorkers were stopped at nearly DOUBLE their population share."

**2. Hit Rate / Outcome Rate**
> "Here's what's telling: the 'hit rate' - when a stop actually finds contraband or leads to arrest - is actually LOWEST for Black individuals. If stops were based purely on suspicious behavior, we'd expect similar hit rates. The lower hit rate suggests a lower threshold for stopping Black individuals."

**3. Precinct-Level Analysis**
> "When we break down by precinct, we see geographic concentration. High-stop precincts are majority-minority neighborhoods. This creates feedback loops: more policing → more arrests → 'higher crime' statistics → justifies more policing."

**4. Selection Bias - Critical Point**
> "This data has FUNDAMENTAL selection bias. We only see people who were stopped. We don't see the law-abiding Black person who wasn't stopped but would have been if they were White. We don't see the suspicious White person who wasn't stopped."

> "Any model trained on this data learns 'who gets stopped by NYPD' not 'who should be stopped.' Using this for predictive policing perpetuates the bias."

---

# RONALDO: What-If Analysis, Comparison (~2-2.5 min)

## Section 7: What-If Analysis

### Opening Script
> "What-If analysis lets us explore counterfactuals - what happens if we change parameters or remove features? This helps us understand which aspects of the system drive unfairness."

### Scenario Scripts

**1. Threshold Adjustment (0.5 → 0.7)**
> "If we raise the decision threshold from 0.5 to 0.7, we become more conservative - only labeling someone high-risk if we're very confident. This REDUCES false positives across all groups but INCREASES false negatives. We detain fewer innocent people but release more actual recidivists."

**2. Remove Race Features**
> "What if we remove explicit race features? Surprisingly, fairness metrics barely change. Why? Because other features like zip code, employment status, and prior arrests are PROXIES for race. They're correlated because of historical discrimination. Removing race doesn't remove bias."

**3. Balance Training Data**
> "What if we artificially balance the training data to have equal representation? This can help reduce some disparities, but it doesn't address the fundamental issue: the LABELS themselves (arrest, recidivism) reflect biased processes."

**4. Use Only Non-Sensitive Features**
> "What if we use only 'legitimate' features like prior convictions and charge severity? The model still shows bias because these features themselves reflect historical bias. More aggressive prosecution of Black defendants means more priors on record."

**5. Different Fairness Constraints**
> "What if we optimize for demographic parity instead of equalized odds? We get equal SELECTION rates but potentially unequal ERROR rates. Different fairness definitions lead to different outcomes - there's no single 'fair' solution."

**6. Retrain Without Historical Bias**
> "The ultimate question: what if we had UNBIASED training data? We can't create it - all criminal justice data reflects decades of biased enforcement. This is why technical solutions alone are insufficient."

---

## Section 8: Comparison Analysis

### Opening Script
> "Let's compare the two datasets to see patterns of bias in criminal justice AI more broadly."

### Key Points

**1. Base Rate Comparison**
> "COMPAS has a 45.5% base rate - nearly half of defendants recidivate. NYPD has a 6% arrest rate - 94% of stops were 'unsuccessful.' This tells us: NYPD was casting a VERY wide net with low precision."

**2. Similar Bias Patterns**
> "Despite different contexts - sentencing vs policing - both show the same pattern: Black individuals face higher false positive rates. Whether it's being wrongly labeled 'high risk' or being wrongly stopped, the error burden falls disproportionately on Black Americans."

**3. Feedback Loop Visualization**
> "Both systems create feedback loops. COMPAS: labeled high risk → detained → can't maintain job/housing → more likely to recidivate → 'confirms' the prediction. NYPD: area labeled high-crime → more patrols → more arrests → 'confirms' it's high-crime."

**4. The Common Thread**
> "Both systems encode the same historical pattern: over-policing and over-prosecution of Black communities. The algorithms don't CREATE this bias - they OPERATIONALIZE it, giving it a veneer of mathematical objectivity."

---

# VISHAL: Ethics/Sociology/Governance, Limitations (~2.5-3 min)

## Section 9: Ethics Lens

### Opening Script
> "Now we apply the three analytical lenses from our course. Starting with Ethics - we'll evaluate these systems through consequentialist, deontological, and virtue ethics frameworks."

### Consequentialist Analysis Script
> "From a utilitarian perspective, we weigh aggregate benefits against harms. Proponents claim these systems improve efficiency and consistency. But the documented harms - disparate error rates, unjust detention of innocents, erosion of community trust - are severe and fall disproportionately on minorities."

> "A proper utilitarian calculus must weight the SEVERITY of harms, not just count them. Wrongful detention is a severe harm. When this harm falls twice as often on Black defendants, the aggregate harm outweighs efficiency gains."

### Deontological Analysis Script
> "Kant's categorical imperative asks: can we universalize this maxim? Would we accept a system that judges individuals based on statistical group membership if it applied to OUR demographic group?"

> "Three Kantian principles are violated:
> 1. Respect for persons - using statistical patterns to judge individuals treats people as MEANS, not ends
> 2. Universalizability - we wouldn't accept this if it disadvantaged US
> 3. Autonomy - opaque algorithms deny people the ability to understand and contest decisions"

### Virtue Ethics Script
> "Aristotle asks: what would a VIRTUOUS institution do? The key virtues:
> - JUSTICE: giving each person their due - systematic inequality fails this
> - PRUDENCE: practical wisdom requiring context - automated scoring strips nuance
> - TEMPERANCE: humility about knowledge - false precision of '7 out of 10' belies uncertainty"

---

## Section 10: Sociology Lens

### Power Dynamics Script
> "Using Frankfurt School critical theory, we examine who designs, deploys, and is subjected to these systems."

> "DESIGNERS are predominantly white, affluent technologists with no lived experience of over-policing. DEPLOYERS are courts and police departments with institutional interests in appearing 'objective.' SUBJECTS are disproportionately low-income people of color who lack resources to challenge algorithmic decisions."

> "This is a clear power asymmetry - those most affected have the least input."

### Who Benefits/Harmed Script
> "Benefits flow to: private companies selling these tools, politicians claiming 'evidence-based' policy, courts seeking efficiency, and majority communities who bear lower false positive rates."

> "Harms concentrate on: Black and Hispanic defendants, low-income defendants who can't afford experts to challenge scores, minority communities whose trust in institutions erodes, and democratic accountability itself."

### Structural Racism Script
> "Ruha Benjamin calls this the 'New Jim Code' - algorithms don't CREATE racism, they OPERATIONALIZE existing structural racism into automated decisions."

> "The feedback loop: Historical discrimination → Over-policing → Higher arrest rates → Biased training data → Algorithm learns 'minority = high risk' → Directs more policing to minorities → More arrests → 'Validates' the algorithm. This is structural racism on autopilot."

---

## Section 11: Governance Lens

### Regulatory Landscape Script
> "The US has no federal AI regulation. We have a patchwork of state laws - Illinois BIPA for biometrics, NYC Local Law 144 for hiring algorithms. Courts have largely failed to apply Equal Protection to algorithms."

> "The EU is ahead with the AI Act classifying criminal justice AI as 'high risk,' requiring transparency and human oversight. GDPR Article 22 gives Europeans the right to human review of automated decisions."

### Accountability Gaps Script
> "Three critical gaps:
> 1. OPACITY - COMPAS is proprietary. Defendants can't examine how their score was calculated. Wisconsin Supreme Court said this is acceptable.
> 2. LIABILITY - When algorithms cause harm, who's responsible? The vendor? The state? The judge? No clear framework exists.
> 3. DEMOCRATIC DEFICIT - Affected communities have NO input into systems that govern their lives."

### Recommendations Script
> "We propose six reforms:
> 1. Mandatory bias audits before deployment
> 2. Algorithmic impact assessments like environmental impact assessments
> 3. Right to explanation of individual scores
> 4. Community oversight boards with real power
> 5. Clear vendor liability framework
> 6. Sunset clauses requiring regular re-evaluation"

---

## Section 12: Limitations

### Script
> "We must acknowledge our analysis's limitations:

> **Data limitations**: We used public subsets. Full COMPAS uses 137 features; we have ~10. Real NYPD data has more context we don't see.

> **Model limitations**: We trained simple models for illustration. Production systems may be more complex but face the same fundamental data bias.

> **Fairness limitations**: We focused on race. Intersectional analysis - race AND gender AND age - would reveal additional disparities.

> **Counterfactual limitations**: We can't observe the true counterfactual - what would have happened without these systems? Would human judges be more or less biased?

> **Scope limitations**: Two US datasets can't represent all criminal justice AI globally. Different contexts may show different patterns."

---

## Closing Script (Any Team Member)

> "In conclusion: algorithmic systems in criminal justice encode historical discrimination, giving it a veneer of mathematical objectivity. Technical fixes alone are insufficient - we need governance reforms that center affected communities, ensure transparency, and establish clear accountability."

> "The question isn't 'Can AI be fair?' but 'Should AI make these decisions at all?' And if so, under what conditions and oversight?"

> "Thank you. We welcome your questions."

---

# Q&A PREPARATION

## Likely Questions and Suggested Answers

**Q: Isn't some bias justified if one group actually commits more crime?**
> "The data shows ARREST rates, not crime rates. Higher arrests can reflect over-policing, not more crime. Self-report surveys show similar drug use across races, but Black Americans are 3.7x more likely to be arrested for marijuana. The 'base rate' itself is biased."

**Q: Can't we just remove race from the algorithm?**
> "We showed this in What-If analysis - removing race barely changes outcomes because other features are proxies. Zip code, employment, priors - all correlate with race due to historical discrimination. This is called 'redundant encoding.'"

**Q: What's the alternative? Human judges are biased too.**
> "Studies show human judges ARE biased. But algorithms scale bias consistently and hide it behind mathematical authority. The solution isn't necessarily more human judgment - it's transparency, accountability, and community input regardless of who (or what) decides."

**Q: Which fairness definition is correct?**
> "Chouldechova proved you can't satisfy all definitions simultaneously when base rates differ. The choice of fairness definition is a VALUE judgment, not a technical one. Different stakeholders may legitimately prioritize different definitions."

**Q: Has any jurisdiction banned these systems?**
> "Some cities have banned facial recognition. The EU AI Act restricts criminal justice AI. But COMPAS and similar tools remain widely used. Change is slow because of institutional inertia and vendor lobbying."

---

# TECHNICAL QUICK REFERENCE CARD

| Metric | Formula | Good Value | Use When |
|--------|---------|------------|----------|
| AUC | Area under ROC | > 0.7 | Ranking ability matters |
| Accuracy | (TP+TN)/Total | > 0.7 | Balanced classes |
| Precision | TP/(TP+FP) | Higher | FP costly |
| Recall | TP/(TP+FN) | Higher | FN costly |
| F1 | 2×P×R/(P+R) | Higher | Balance P and R |
| Brier | Mean(p-y)² | < 0.25 | Calibration matters |
| FPR | FP/(FP+TN) | Lower | Innocents matter |
| FNR | FN/(FN+TP) | Lower | Catching positives matters |
| DPD | \|Rate_A - Rate_B\| | ~0 | Equal selection |
| EOD | Max diff in TPR/FPR | ~0 | Equal errors |

---

*End of Speaker Notes*
