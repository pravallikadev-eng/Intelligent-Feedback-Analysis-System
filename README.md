# 🎫 Intelligent Feedback Analysis System — Capstone Project

## What This Does
A multi-agent AI system that reads user feedback from CSV files, classifies it,
extracts technical details, creates structured tickets, and quality-reviews them.

## Project Structure
```
capstone/
├── data/
│   ├── app_store_reviews.csv       ← 20 mock reviews (bugs, features, praise, spam)
│   ├── support_emails.csv          ← 12 mock support emails
│   └── expected_classifications.csv ← Ground truth for accuracy testing
├── agents/
│   └── pipeline.py                 ← Multi-agent CrewAI pipeline (6 agents)
├── output/                         ← Auto-created when pipeline runs
│   ├── generated_tickets.csv
│   ├── processing_log.csv
│   └── metrics.csv
├── app.py                          ← Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
# For Anthropic Claude:
export ANTHROPIC_API_KEY="your-key-here"

# OR for OpenAI:
export OPENAI_API_KEY="your-key-here"
```

### 3. Run the Streamlit dashboard
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

### 4. Run the pipeline from command line (optional)
```bash
cd capstone
python agents/pipeline.py
```

## The 6 Agents

| Agent | Role |
|-------|------|
| CSV Reader | Loads and validates feedback from CSV files |
| Feedback Classifier | Categorizes: Bug / Feature Request / Praise / Complaint / Spam |
| Bug Analyzer | Extracts device, OS, steps to reproduce, severity |
| Feature Extractor | Summarizes feature request, estimates user impact |
| Ticket Creator | Generates structured tickets, writes to CSV |
| Quality Critic | Reviews tickets, assigns quality score 0-10 |

## Using the Dashboard

1. **Dashboard** — KPI cards, charts, recent tickets
2. **Tickets** — View, filter, and manually edit all tickets
3. **Raw Feedback** — Browse the original CSV data
4. **Run Pipeline** — Trigger the agents with your API key
5. **Analytics** — Accuracy vs expected + quality scores

## Configuration (Sidebar)
- Switch between Claude / GPT-4o / Gemini models
- Adjust batch size (how many feedback items per run)
- Set classification confidence threshold

## Output Files
- `generated_tickets.csv` — Final tickets ready for Jira/Linear import
- `processing_log.csv` — Every agent decision logged
- `metrics.csv` — Run history with timing and counts
