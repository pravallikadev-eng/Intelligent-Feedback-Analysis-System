"""
Intelligent User Feedback Analysis and Action System
Multi-Agent Pipeline using CrewAI + Claude/OpenAI
"""

import os
import json
import csv
import uuid
import logging
from datetime import datetime
from typing import Optional
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd

# ─────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("output/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CONFIG = {
    "model": "claude-opus-4-5",          # or "gpt-4o"
    "classification_threshold": 0.7,
    "priority_rules": {
        "crash": "Critical",
        "data loss": "Critical",
        "login": "High",
        "billing": "High",
        "slow": "Medium",
        "feature": "Medium",
        "praise": "Low",
        "spam": "Low",
    },
    "input_files": {
        "reviews": "data/app_store_reviews.csv",
        "emails": "data/support_emails.csv",
        "expected": "data/expected_classifications.csv",
    },
    "output_files": {
        "tickets": "output/generated_tickets.csv",
        "log": "output/processing_log.csv",
        "metrics": "output/metrics.csv",
    }
}

os.makedirs("output", exist_ok=True)

# ─────────────────────────────────────────
# Tools
# ─────────────────────────────────────────

class CSVReaderTool(BaseTool):
    name: str = "csv_reader"
    description: str = "Reads feedback data from a CSV file and returns it as JSON."

    def _run(self, filepath: str) -> str:
        try:
            df = pd.read_csv(filepath)
            df = df.fillna("")
            records = df.to_dict(orient="records")
            logger.info(f"CSVReaderTool: loaded {len(records)} rows from {filepath}")
            return json.dumps(records[:5])  # Return first 5 for demo
        except Exception as e:
            logger.error(f"CSVReaderTool error: {e}")
            return json.dumps({"error": str(e)})


class TicketWriterTool(BaseTool):
    name: str = "ticket_writer"
    description: str = "Writes a generated ticket dict to the output CSV file."

    def _run(self, ticket_json: str) -> str:
        try:
            ticket = json.loads(ticket_json)
            filepath = CONFIG["output_files"]["tickets"]
            file_exists = os.path.exists(filepath)
            with open(filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "ticket_id", "source_id", "source_type", "category",
                    "priority", "title", "description", "steps_to_reproduce",
                    "platform", "app_version", "quality_score",
                    "created_at", "status"
                ])
                if not file_exists:
                    writer.writeheader()
                ticket.setdefault("ticket_id", f"TKT-{str(uuid.uuid4())[:8].upper()}")
                ticket.setdefault("created_at", datetime.now().isoformat())
                ticket.setdefault("status", "Open")
                writer.writerow(ticket)
            logger.info(f"TicketWriterTool: wrote ticket {ticket.get('ticket_id')}")
            return f"Ticket {ticket.get('ticket_id')} written successfully."
        except Exception as e:
            logger.error(f"TicketWriterTool error: {e}")
            return f"Error: {e}"


class LogWriterTool(BaseTool):
    name: str = "log_writer"
    description: str = "Logs a processing decision to the processing log CSV."

    def _run(self, log_json: str) -> str:
        try:
            entry = json.loads(log_json)
            filepath = CONFIG["output_files"]["log"]
            file_exists = os.path.exists(filepath)
            with open(filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "log_id", "source_id", "agent", "action",
                    "result", "confidence", "timestamp"
                ])
                if not file_exists:
                    writer.writeheader()
                entry["log_id"] = str(uuid.uuid4())[:8]
                entry["timestamp"] = datetime.now().isoformat()
                writer.writerow(entry)
            return "Log entry written."
        except Exception as e:
            return f"Log error: {e}"


# ─────────────────────────────────────────
# Agent Definitions
# ─────────────────────────────────────────

def build_agents():
    csv_reader = Agent(
        role="CSV Reader Agent",
        goal="Read and parse all feedback from input CSV files accurately.",
        backstory=(
            "You are a data ingestion specialist. You load user feedback from "
            "CSV files and present each record cleanly for downstream agents."
        ),
        tools=[CSVReaderTool()],
        verbose=True,
        allow_delegation=False,
    )

    classifier = Agent(
        role="Feedback Classifier Agent",
        goal=(
            "Classify each feedback item into exactly one of: "
            "Bug, Feature Request, Praise, Complaint, Spam. "
            "Assign a confidence score 0.0–1.0."
        ),
        backstory=(
            "You are an NLP classification expert trained on thousands of "
            "user feedback examples. You are precise, consistent, and fast."
        ),
        tools=[LogWriterTool()],
        verbose=True,
        allow_delegation=False,
    )

    bug_analyzer = Agent(
        role="Bug Analysis Agent",
        goal=(
            "For items classified as Bug: extract device info, OS version, "
            "app version, steps to reproduce, and assign severity "
            "(Critical / High / Medium / Low)."
        ),
        backstory=(
            "You are a senior QA engineer. You read bug reports and extract "
            "structured technical details that developers need to reproduce and fix issues."
        ),
        tools=[LogWriterTool()],
        verbose=True,
        allow_delegation=False,
    )

    feature_extractor = Agent(
        role="Feature Extractor Agent",
        goal=(
            "For Feature Request items: summarize the requested feature, "
            "estimate user impact (High/Medium/Low), and suggest a priority."
        ),
        backstory=(
            "You are a product manager who reads feature requests and evaluates "
            "their business value, user demand, and implementation complexity."
        ),
        tools=[LogWriterTool()],
        verbose=True,
        allow_delegation=False,
    )

    ticket_creator = Agent(
        role="Ticket Creator Agent",
        goal=(
            "Generate a complete, structured ticket for each feedback item. "
            "Include: ticket_id, source_id, source_type, category, priority, "
            "title, description, steps_to_reproduce, platform, app_version. "
            "Write each ticket to the output CSV."
        ),
        backstory=(
            "You are a project management expert who writes clean, actionable "
            "engineering tickets. Every ticket you create is clear, complete, "
            "and immediately actionable by a developer."
        ),
        tools=[TicketWriterTool(), LogWriterTool()],
        verbose=True,
        allow_delegation=False,
    )

    quality_critic = Agent(
        role="Quality Critic Agent",
        goal=(
            "Review each generated ticket for: completeness, accurate priority, "
            "correct category, and actionable title. "
            "Assign a quality_score 0–10 and flag any issues."
        ),
        backstory=(
            "You are a quality assurance lead who reviews tickets before they "
            "enter the engineering backlog. You ensure consistency and completeness."
        ),
        tools=[LogWriterTool()],
        verbose=True,
        allow_delegation=False,
    )

    return csv_reader, classifier, bug_analyzer, feature_extractor, ticket_creator, quality_critic


# ─────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────

def build_tasks(agents, feedback_batch: list[dict]) -> list[Task]:
    csv_reader, classifier, bug_analyzer, feature_extractor, ticket_creator, quality_critic = agents

    feedback_text = json.dumps(feedback_batch, indent=2)

    task_read = Task(
        description=(
            f"Parse and validate the following feedback records. "
            f"Return them as a clean JSON list, one record per item:\n\n{feedback_text}"
        ),
        expected_output="A clean JSON list of feedback records with all fields present.",
        agent=csv_reader,
    )

    task_classify = Task(
        description=(
            "For each feedback record from the previous task, classify it into exactly one of: "
            "Bug, Feature Request, Praise, Complaint, Spam. "
            "Return JSON list with added fields: category, confidence (0.0-1.0)."
        ),
        expected_output="JSON list with category and confidence added to each record.",
        agent=classifier,
        context=[task_read],
    )

    task_bug_analyze = Task(
        description=(
            "For every record classified as 'Bug', extract: "
            "device_info, os_version, app_version, steps_to_reproduce, severity. "
            "For non-bugs, pass through unchanged. Return full JSON list."
        ),
        expected_output="JSON list with bug technical details added where applicable.",
        agent=bug_analyzer,
        context=[task_classify],
    )

    task_feature_extract = Task(
        description=(
            "For every record classified as 'Feature Request', extract: "
            "feature_summary, user_impact (High/Medium/Low), suggested_priority. "
            "For non-features, pass through unchanged. Return full JSON list."
        ),
        expected_output="JSON list with feature request details added where applicable.",
        agent=feature_extractor,
        context=[task_bug_analyze],
    )

    task_create_tickets = Task(
        description=(
            "For EACH feedback record, create a structured ticket and write it using "
            "the ticket_writer tool. Each ticket must include: "
            "ticket_id (generate as TKT-XXXXX), source_id, source_type, category, "
            "priority, title (clear and actionable, start with [BUG]/[FEATURE]/[PRAISE]/[COMPLAINT]/[SPAM]), "
            "description (2-3 sentences), steps_to_reproduce (for bugs), "
            "platform, app_version, quality_score (leave blank for critic to fill). "
            "Write each ticket as a JSON string to the ticket_writer tool."
        ),
        expected_output="Confirmation that all tickets have been written to CSV.",
        agent=ticket_creator,
        context=[task_feature_extract],
    )

    task_quality_review = Task(
        description=(
            "Review all tickets created in the previous task. For each ticket, "
            "evaluate: (1) Is the title actionable? (2) Is the priority correct? "
            "(3) Is the category accurate? (4) Is the description complete? "
            "Assign a quality_score out of 10. Flag any tickets scoring below 7. "
            "Return a quality report summarizing findings."
        ),
        expected_output=(
            "A quality report with: overall_score, tickets_reviewed, "
            "tickets_flagged, list of any issues found."
        ),
        agent=quality_critic,
        context=[task_create_tickets],
    )

    return [task_read, task_classify, task_bug_analyze, task_feature_extract,
            task_create_tickets, task_quality_review]


# ─────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────

def load_all_feedback() -> list[dict]:
    """Load and combine feedback from both CSV files."""
    records = []

    # Load app store reviews
    try:
        df = pd.read_csv(CONFIG["input_files"]["reviews"]).fillna("")
        for _, row in df.iterrows():
            r = row.to_dict()
            r["source_type"] = "app_store_review"
            r["source_id"] = r.pop("review_id", f"R{len(records):03d}")
            r["text"] = r.pop("review_text", "")
            records.append(r)
        logger.info(f"Loaded {len(df)} app store reviews")
    except Exception as e:
        logger.error(f"Error loading reviews: {e}")

    # Load support emails
    try:
        df = pd.read_csv(CONFIG["input_files"]["emails"]).fillna("")
        for _, row in df.iterrows():
            r = row.to_dict()
            r["source_type"] = "support_email"
            r["source_id"] = r.pop("email_id", f"E{len(records):03d}")
            r["text"] = r.get("subject", "") + " " + r.get("body", "")
            records.append(r)
        logger.info(f"Loaded {len(df)} support emails")
    except Exception as e:
        logger.error(f"Error loading emails: {e}")

    return records


def write_metrics(result: str, total: int, start_time: datetime):
    """Write run metrics to CSV."""
    elapsed = (datetime.now() - start_time).total_seconds()
    tickets_written = 0
    if os.path.exists(CONFIG["output_files"]["tickets"]):
        try:
            tickets_written = len(pd.read_csv(CONFIG["output_files"]["tickets"]))
        except:
            pass

    filepath = CONFIG["output_files"]["metrics"]
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "timestamp", "feedback_processed",
            "tickets_generated", "elapsed_seconds", "status"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "feedback_processed": total,
            "tickets_generated": tickets_written,
            "elapsed_seconds": round(elapsed, 2),
            "status": "Success" if "error" not in result.lower() else "Error"
        })


def run_pipeline(batch_size: int = 5) -> dict:
    """Run the full multi-agent pipeline."""
    start = datetime.now()
    logger.info("=" * 50)
    logger.info("Starting Feedback Analysis Pipeline")
    logger.info("=" * 50)

    # Load feedback
    all_feedback = load_all_feedback()
    if not all_feedback:
        return {"status": "error", "message": "No feedback loaded"}

    # Process in batches
    batch = all_feedback[:batch_size]
    logger.info(f"Processing batch of {len(batch)} items")

    # Build agents and tasks
    agents = build_agents()
    tasks = build_tasks(agents, batch)

    # Build and run crew
    crew = Crew(
        agents=list(agents),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = crew.kickoff()
        result_str = str(result)
        logger.info("Pipeline completed successfully")
    except Exception as e:
        result_str = f"Pipeline error: {e}"
        logger.error(result_str)

    write_metrics(result_str, len(batch), start)

    return {
        "status": "success",
        "items_processed": len(batch),
        "result_summary": result_str[:500],
        "elapsed": str(datetime.now() - start),
        "output_files": CONFIG["output_files"]
    }


if __name__ == "__main__":
    # Set your API key before running
    # os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
    # os.environ["OPENAI_API_KEY"] = "your-key-here"

    result = run_pipeline(batch_size=5)
    print("\n" + "="*50)
    print("PIPELINE RESULT:")
    print(json.dumps(result, indent=2))
