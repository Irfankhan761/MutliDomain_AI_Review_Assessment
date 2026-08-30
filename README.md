# MultiDomain Review Analysis AI

This project is an Agentic AI review trust and risk assessment system.

It can analyse reviews from multiple domains, including:

- Mobile apps
- E-commerce products
- Hotels
- Restaurants
- Google Play reviews
- Google Maps reviews
- Single review text
- CSV datasets

The system uses multiple specialist agents instead of one single model. Each agent performs one part of the analysis, and the final orchestrator combines all results into an explainable trust and risk decision.

---

## 1. What This System Does

The system can:

1. Accept review data from a CSV file, single review text, Google Play URL, Google Play App ID, or Google Maps place URL.
2. Convert the input into one common review format.
3. Analyse review sentiment.
4. Predict the rating from review text.
5. Compare actual rating with predicted rating.
6. Detect review issues such as crash, login, payment, privacy, subscription, fake product, room quality, staff service, food quality, cleanliness and more.
7. Retrieve supporting evidence using RAG.
8. Calculate trust score, risk level and reliability level.
9. Generate review-level explanations.
10. Generate entity-level summaries for apps, products, hotels and restaurants.
11. Generate a final readable trust/risk summary using Groq.
12. Create evaluation results and dissertation figures.

---

## 2. Main Technologies Used

- Python
- Flask
- HTML, CSS and JavaScript
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Transformers
- Sentence Transformers
- FAISS
- Groq API
- Playwright / browser-based scraping support
- Google Play review scraping support

---

## 3. Main Folder Structure

```text
MultiDomain_Review_Analysis_Agentic_AI/
│
├── agents/
│   └── Specialist agent files
│
├── data/
│   ├── processed/
│   └── raw/
│
├── outputs/
│   └── Generated results, models, reports and figures
│
├── pipeline/
│   └── Main analysis pipeline files
│
├── scripts/
│   └── Setup, testing, evaluation and utility scripts
│
├── services/
│   └── Scraper and service helper files
│
├── static/
│   ├── app.css
│   └── app.js
│
├── templates/
│   └── index.html
│
├── app.py
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. Important Note About GitHub Files

Large files are not uploaded to GitHub.

The following folders are usually not pushed because they are generated again on the client system:

```text
venv/
.env
.browser_profiles/
outputs/models/
outputs/final_orchestrator_runs/
outputs/phase15_evaluation_results/
outputs/dissertation_figures/
outputs/vector_index/
data/raw/
```

This is normal.

You can rebuild these folders by running the commands given below.

---

## 5. First-Time Setup

Open the project folder in VS Code.

Then open the terminal inside the project root folder.

The terminal path should look similar to this:

```powershell
F:\...\MultiDomain_Review_Analysis_Agentic_AI>
```

---

## 6. Create Virtual Environment

Run this command:

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(venv)
```

---

## 7. Install Requirements

Run:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs the required Python packages.

---

## 8. Environment File Setup

Copy this file:

```text
.env.example
```

Rename the copy to:

```text
.env
```

Inside `.env`, add your private Groq API key:

```env
GROQ_API_KEY=YOUR_PRIVATE_GROQ_API_KEY
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-120b
```

Do not upload `.env` to GitHub because it contains a private API key.

---

## 9. Download Local Models

The project uses local models so that normal analysis does not call Hugging Face again and again.

Run:
```powershell
python scripts/train_distilbert_sentiment.py
```
```powershell
python scripts/download_local_models.py
```

This creates the following folders:

```text
outputs/models/distilbert_sentiment/
outputs/models/all-MiniLM-L6-v2/
outputs/models/nlptown_bert_rating/
```

### Purpose of Each Model

| Model Folder | Purpose |
|---|---|
| `outputs/models/distilbert_sentiment/` | Predicts review sentiment |
| `outputs/models/all-MiniLM-L6-v2/` | Used for semantic issue matching and RAG evidence retrieval |
| `outputs/models/nlptown_bert_rating/` | Predicts star rating from review text |

---

## 10. Test Local Offline Models

Run:

```powershell
python scripts/test_local_offline_models.py
```

Expected result:

```text
OFFLINE LOCAL MODEL TEST PASSED
```

This confirms that the models are loading locally.

---

## 11. Run the Flask Web App

Run:

```powershell
python app.py
```

Then open this URL in the browser:

```text
http://127.0.0.1:5000
```

The web app supports:

- CSV Dataset
- Single Review
- Google Play URL
- Google Play App ID
- Google Maps review analysis

---

## 12. Run Final Orchestrator from Command Line

The final orchestrator is the main backend pipeline.

### Quick Demo Run

```powershell
python scripts/run_final_orchestrator_offline.py --csv data/processed/combined_multidomain_reviews.csv --sample-size 200
```

### Full Dataset Run

```powershell
python scripts/run_final_orchestrator_offline.py --csv data/processed/combined_multidomain_reviews.csv --sample-size 0
```

### Run Without Groq Final Summary

```powershell
python scripts/run_final_orchestrator_offline.py --csv data/processed/combined_multidomain_reviews.csv --sample-size 200 --no-groq
```

### Run Without RAG and Groq

```powershell
python scripts/run_final_orchestrator_offline.py --csv data/processed/combined_multidomain_reviews.csv --sample-size 200 --no-rag --no-groq
```

---

## 13. Final Orchestrator Output

Every final run creates a new folder here:

```text
outputs/final_orchestrator_runs/<run_id>/
```

Example:

```text
outputs/final_orchestrator_runs/20260805_145802_8d5ee6fc/
```

Important files inside this folder:

| File / Folder | Purpose |
|---|---|
| `orchestrator_state.json` | Stores the full workflow state |
| `prepared_standardised_dataset.csv` | Standardised input dataset |
| `analysis_pipeline/` | Main analytical outputs |
| `analysis_pipeline/multidomain_review_level_results.csv` | Review-level results |
| `analysis_pipeline/multidomain_entity_level_summary.csv` | Entity-level summary |
| `final_groq_report.txt` | Final readable trust/risk summary |
| `final_groq_context_payload.json` | Structured data used for Groq summary |

---

## 14. Review-Level Result File

Main file:

```text
outputs/final_orchestrator_runs/<run_id>/analysis_pipeline/multidomain_review_level_results.csv
```

This file contains:

- Review text
- Actual rating
- Predicted sentiment
- Sentiment confidence
- Predicted star rating
- Rating-review mismatch status
- Primary issue
- Issue severity
- RAG evidence
- Trust score
- Risk level
- Reliability level
- Recommendation level
- Explanation text

This is the most important file for checking each review result.

---

## 15. Entity-Level Summary File

Main file:

```text
outputs/final_orchestrator_runs/<run_id>/analysis_pipeline/multidomain_entity_level_summary.csv
```

This file contains summary results for each entity.

Examples of entities:

- Mobile app
- Product
- Hotel
- Restaurant
- Google Maps location

Important columns:

- Entity name
- Entity type
- Total reviews
- Average rating
- Average trust score
- Overall risk level
- High-risk percentage
- Mismatch percentage
- Top issues
- Recommendation

---

## 16. Google Play Review Analysis

### Option 1: Use Web UI

Run:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Select:

```text
Google Play URL
```

Paste a Google Play app URL, for example:

```text
https://play.google.com/store/apps/details?id=com.example.app
```

Set maximum reviews and run the analysis.

### Option 2: Use App ID

Select:

```text
App ID
```

Enter an app ID, for example:

```text
com.example.app
```

The system will collect public Google Play reviews, convert them to the common schema and run the full agentic analysis.

---

## 17. Google Maps Review Analysis

Google Maps review scraping uses a browser workflow.

### Step 1: Start Chrome Bridge

```powershell
python scripts/start_google_maps_chrome.py
```

Keep the opened Chrome window running.

### Step 2: Check Bridge Connection

```powershell
python scripts/check_google_maps_bridge.py
```

### Step 3: Start Web App

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Paste the exact Google Maps place URL and run the analysis.

The system will:

1. Open the Google Maps place page.
2. Open the reviews panel.
3. Scroll reviews.
4. Expand review text where possible.
5. Save collected reviews.
6. Convert reviews into the common schema.
7. Run the trust and risk analysis pipeline.

---

## 18. Evaluation and Results

After running the final orchestrator, run:

```powershell
python scripts/run_phase15_evaluation_results.py --latest
```

This uses the latest final orchestrator run and creates:

```text
outputs/phase15_evaluation_results/<timestamp>/
```

Important evaluation files:

| File | Purpose |
|---|---|
| `sentiment_confusion_matrix.csv` | Sentiment confusion matrix |
| `sentiment_metrics.csv` | Sentiment accuracy, precision, recall and F1 |
| `rating_confusion_matrix.csv` | Actual rating vs predicted rating |
| `domain_performance.csv` | Domain-wise performance results |
| `risk_distribution.csv` | Risk level counts |
| `risk_by_domain.csv` | Risk levels by domain |
| `issue_distribution.csv` | Issue counts |
| `issue_by_domain.csv` | Issues by domain |
| `severity_distribution.csv` | Severity counts |
| `severity_by_domain.csv` | Severity by domain |
| `dominant_factor_distribution.csv` | Main risk-driving factors |
| `rag_metrics.csv` | RAG evidence metrics |
| `top_risky_entities.csv` | Highest-risk entities |
| `phase15_evaluation_report.md` | Full evaluation report |
| `phase15_chapter5_results_summary.txt` | Dissertation-ready results summary |

---

## 19. Generate Dissertation Figures

Run:

```powershell
python scripts/generate_dissertation_figures.py --latest
```

This creates:

```text
outputs/dissertation_figures/
```

Typical figures:

- Sentiment confusion matrix
- Rating confusion matrix
- Risk level distribution
- Risk by domain
- Issue distribution
- Severity distribution
- Trust score by domain
- Domain performance chart

These figures can be used in the dissertation findings/results chapter.

---

## 20. Processed Dataset Files

Processed datasets are stored here:

```text
data/processed/
```

Important files:

| File | Purpose |
|---|---|
| `amazon_normalized.csv` | E-commerce reviews |
| `hotel_normalized.csv` | Hotel reviews |
| `mobile_app_normalized.csv` | Mobile app reviews |
| `yelp_restaurant_normalized.csv` | Restaurant reviews |
| `combined_multidomain_reviews.csv` | Combined dataset used for full testing |
| `orchestrator_single_review.csv` | Small single-review test file |
| `orchestrator_hotel_sample.csv` | Small hotel test file |
| `orchestrator_multidomain_sample.csv` | Small multi-domain test file |

If processed files need to be rebuilt from raw datasets, place raw files in:

```text
data/raw/
```

Then run:

```powershell
python scripts/build_multidomain_dataset.py
```

---

## 21. Development Test Scripts

These scripts were used during development and testing.

| Script | Purpose |
|---|---|
| `scripts/test_phase6_transformer_pipeline.py` | Tests transformer sentiment pipeline |
| `scripts/test_phase7_discrepancy_pipeline.py` | Tests rating prediction and discrepancy checking |
| `scripts/test_phase8_semantic_issue_pipeline.py` | Tests semantic issue mining |
| `scripts/test_phase9_rag_pipeline.py` | Tests RAG evidence retrieval |
| `scripts/test_phase10_risk_scoring_pipeline.py` | Tests risk scoring |
| `scripts/test_phase11_explainability_summary_pipeline.py` | Tests explainability and entity summary |
| `scripts/test_phase12_groq_final_summary.py` | Tests Groq final summary |
| `scripts/run_phase15_evaluation_results.py` | Generates evaluation results |
| `scripts/generate_dissertation_figures.py` | Generates dissertation figures |

It does not need to run all development test scripts. They are available only for checking individual modules.

---

## 22. Recommended User Run Order

For a fresh system, run these commands in order:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_local_models.py
python scripts/test_local_offline_models.py
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

For command-line testing:

```powershell
python scripts/run_final_orchestrator_offline.py --csv data/processed/combined_multidomain_reviews.csv --sample-size 200
python scripts/run_phase15_evaluation_results.py --latest
python scripts/generate_dissertation_figures.py --latest
```

---

## 23. Common Issues and Fixes

### Issue: PowerShell blocks venv activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### Issue: Groq API key not found

Check that `.env` exists and contains:

```env
GROQ_API_KEY=YOUR_PRIVATE_GROQ_API_KEY
```

### Issue: Local models missing

Run:

```powershell
python scripts/download_local_models.py
```

Then test:

```powershell
python scripts/test_local_offline_models.py
```

### Issue: Web app not opening

Make sure Flask is running:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Issue: Google Maps scraper not working

Start the Chrome bridge first:

```powershell
python scripts/start_google_maps_chrome.py
```

Then check:

```powershell
python scripts/check_google_maps_bridge.py
```

Then run:

```powershell
python app.py
```

---

## 24. Final Notes

- Use `sample-size 200` for a quick demo.
- Use `sample-size 0` for the full dataset.
- Local models are downloaded into `outputs/models/`.
- Generated results are saved into `outputs/final_orchestrator_runs/`.
- Evaluation results are saved into `outputs/phase15_evaluation_results/`.
- Dissertation figures are saved into `outputs/dissertation_figures/`.
- Do not upload `.env`, `venv`, raw private data, browser profiles or generated output folders to GitHub.
- The final trust score is calculated from multiple agent outputs, not from one model only.
- Groq is used only for the final readable summary.
- The system still works without Groq, but the final natural-language summary will not be generated.
