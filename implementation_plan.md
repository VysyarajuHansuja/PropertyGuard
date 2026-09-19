# Implementation Plan — PropertyGuard: Climate & Sustainability-Aware House Price Prediction & Conversational AI

Build and maintain a complete end-to-end Machine Learning, NASA Climate Awareness, and Conversational AI platform using Jupyter Notebooks for EDA & training (`clean_dataset.ipynb` & `train_model.ipynb`), a NASA microclimate engine (`climate_engine.py`), a conversational AI assistant (`ask_propertyguard.py`), and an interactive multi-tab **Streamlit** dashboard (`app.py`).

## Proposed Architecture & Workflow

```mermaid
flowchart TD
    A[Raw City Datasets 6 Cities] --> B[clean_dataset.ipynb]
    B -->|Fix 9 Bug, Remove Duplicates & Outliers| C[cleaned_house_prices_6cities.csv]
    C --> D[train_model.ipynb]
    D -->|Log Target, Locality Encoders, CatBoost/XGBoost| E[Saved Artifacts: model.joblib, encoders.joblib, locality_stats.json]
    
    E --> F[app.py Streamlit App]
    G[climate_engine.py NASA Climate Layer] --> F
    
    F -->|Property Context: Valuation + Climate + Benchmarks| H[ask_propertyguard.py Conversational Engine]
    H -->|Rule-Based Fallback / Optional LLM Hook| F
    
    F --> I[Interactive Dashboard: Valuation | NASA Climate | Buyer Advice | AI Chatbot]
```

---

## Core Components

### 1. Data Cleaning & Analysis Notebook
**File:** [`clean_dataset.ipynb`](file:///c:/Users/vhans/Desktop/Project_enigma/PrpertyGuard/clean_dataset.ipynb)
* Merges 6 major city CSV datasets (`Bangalore.csv`, `Chennai.csv`, `Delhi.csv`, `Hyderabad.csv`, `Kolkata.csv`, `Mumbai.csv`).
* Cell-by-cell Exploratory Data Analysis (EDA) & visual distributions.
* Resolves the dataset **"9" bug**: Maps `9` $\rightarrow$ `0` across all 35 amenity features, and engineers an `amenities_reported_count` feature.
* Drops duplicate listings (~3,828 redundant records).
* Filters extreme price-per-sq.ft. outliers ($< \text{₹}500/\text{sq.ft.}$ or $> \text{₹}1,50,000/\text{sq.ft.}$) and listing typo anomalies.
* Exports consolidated dataset: `cleaned_house_prices_6cities.csv`.

---

### 2. Machine Learning Model Training Notebook
**File:** [`train_model.ipynb`](file:///c:/Users/vhans/Desktop/Project_enigma/PrpertyGuard/train_model.ipynb)
* Loads `cleaned_house_prices_6cities.csv`.
* Prepares features: `City`, `Locality` (Frequency & Target Encoders), `Area`, `Bedrooms`, `Resale`, 35 Amenity indicators, and `amenities_count`.
* Applies $\log(\text{Price})$ target transformation to stabilize skewed distributions.
* Trains and compares ML models (CatBoost Regressor, XGBoost, Random Forest, LightGBM) with train/test splits and cross-validation.
* Evaluates metrics: MAE, RMSE, MAPE, $R^2$ (overall and city-wise breakdowns).
* Generates feature importance charts.
* Serializes production artifacts: `model.joblib`, `encoders.joblib`, and `locality_stats.json`.

---

### 3. NASA Climate & Sustainability Engine
**File:** [`climate_engine.py`](file:///c:/Users/vhans/Desktop/Project_enigma/PrpertyGuard/climate_engine.py)
* Maps 500+ localities across the 6 major Indian metros to latitude/longitude coordinates and NASA microclimate profiles.
* Integrates environmental and climate risk indicators derived from **NASA POWER**:
  * **Solar Insolation ($ALLSKY\_SWH\_DN$)**: Solar energy potential ($kWh/m^2/day$) and rooftop PV feasibility.
  * **Surface Temperature & Heat Index**: Peak summer maximum temperatures ($T2M\_MAX$), urban heat island intensity.
  * **Precipitation & Monsoon Risk ($PRECTOTCORR$)**: Annual rainfall (mm) and extreme precipitation flood vulnerability.
  * **Air Quality Index / Aerosol Optical Depth ($AOD\_550$)**: Seasonal PM2.5 / haze levels.
  * **Humidity ($RH2M$) & Green Cover %**: Microclimate heat retention.
* Computes composite **Sustainability & Living Score (0–100)** across 4 pillars: Solar Potential, Thermal Comfort, Air Quality, and Water & Flood Resilience.
* Generates actionable **Buyer Advice Cards** (Solar rooftop ROI, Rainwater harvesting necessity, Cooling/HVAC insulation, Flood risk warnings).

---

### 4. Ask PropertyGuard — Conversational AI Advisor
**File:** [`ask_propertyguard.py`](file:///c:/Users/vhans/Desktop/Project_enigma/PrpertyGuard/ask_propertyguard.py)
* **Property Context Aggregation (`build_property_context`)**:
  * Synthesizes property specs, ML fair value estimate, price per sq.ft., $\pm10\%$ valuation bounds, locality median benchmarks, 4-pillar NASA climate scores, and active buyer advice cards into structured context.
* **Modular LLM Hook (`call_llm`)**:
  * Clean plug-and-play hook with system prompt adherence for external LLM backends (e.g. IBM watsonx.ai Granite, OpenAI / Azure OpenAI, local Ollama / LLaMA-3).
  * Returns `None` by default to seamlessly activate the offline rule-based fallback without external API keys or cost.
* **Deterministic Grounded Rule-Based Engine (`_rule_based_response`)**:
  * Keyword-matched, data-grounded answers for home buyers.
  * Handles questions regarding sustainability scores, climate risks, due-diligence checklists, price benchmarks, solar potential, flood risks, and plain-language summaries.
  * Ensures zero hallucinations — only references calculated property data.
* **UI Starter Suggestions (`SUGGESTED_QUESTIONS`)**:
  * Provides quick-click starter prompts for buyers to explore property strengths, weaknesses, and seller inspection points.

---

### 5. Streamlit Interactive Web Application
**File:** [`app.py`](file:///c:/Users/vhans/Desktop/Project_enigma/PrpertyGuard/app.py)
* Modern dark-mode glassmorphism dashboard layout built with custom CSS.
* **Sidebar Controls**: City selector, Locality dropdown (dynamically populated), Area (sq.ft.), Bedrooms, Resale status, and 35 Amenity checkboxes.
* **Multi-Tab Interface**:
  * **Tab 1: 💰 Price Valuation**: Predicted price in ₹ Lakhs / Crores, price/sq.ft., fair-value confidence range ($\pm10\%$), and comparison with locality median benchmark.
  * **Tab 2: 🌍 NASA Climate & Sustainability**: Composite Sustainability Score (0–100) with color-coded gauge, individual pillar breakdown metrics (Solar, Thermal, Air, Water).
  * **Tab 3: 💡 Buyer Recommendations**: Tailored recommendations for energy efficiency, water security, HVAC needs, and a pre-purchase homebuyer inspection checklist.
  * **Tab 4: 🤖 Ask PropertyGuard**: Interactive chatbot tab powered by `ask_propertyguard.py`, featuring starter suggestion buttons, dynamic session-state chat history, and contextual AI answers.

---

### 6. GitHub Repository Packaging & Structure
* Ensure repository includes necessary configuration and setup guides for collaborators and recruiters:
  * `.gitignore`: Excludes Python cache (`__pycache__`), training logs (`catboost_info/`), virtual environments (`.venv`), and temporary checkpoints.
  * `requirements.txt`: Clean dependency list (`streamlit`, `catboost`, `scikit-learn`, `pandas`, `numpy`, `joblib`).
  * `README.md`: Project overview, architecture diagrams, key features, screenshot references (`Project_Screeshots.pdf`), and local installation steps.

---

## Verification Plan

### Automated Execution & Artifact Integrity
1. Check `clean_dataset.ipynb` and `train_model.ipynb` execute cleanly and output expected files.
2. Confirm `model.joblib`, `encoders.joblib`, and `locality_stats.json` load without deserialization errors.
3. Validate `climate_engine.py` returns valid metrics and scores for all 6 cities.
4. Verify `ask_propertyguard.py` response generation across all suggested questions.

### Interactive Streamlit Verification
1. Launch dashboard: `streamlit run app.py`.
2. Test end-to-end user journey across all 4 tabs:
   - Price valuation update when modifying property size or amenities.
   - Climate indicators refresh on locality change.
   - Pre-purchase checklist interaction.
   - Chatbot responses in Tab 4 for both suggestion chips and custom questions.
