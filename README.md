# Resume AI Technical Interview

A Python + FastAPI web application that parses a resume, generates resume-based technical interview questions, conducts an AI-powered mock interview, evaluates candidate responses, and provides detailed scoring and feedback.

---

# Features

## Resume Analysis

* Upload PDF or TXT resumes
* Parse resume content automatically
* Extract technical skills
* Extract project-related information
* Identify technologies, frameworks, and tools used

## AI Interview Generation

* Generate up to 7 technical interview questions
* Create project-specific questions
* Create skill-based technical questions
* Customize questions based on the target role

## Interview Experience

* Ask questions one at a time
* Voice recording support in the browser
* Typed answers as a fallback option
* Real-time interview workflow

## Answer Evaluation

* Semantic similarity scoring
* Technical correctness scoring
* Confidence scoring
* Overall performance scoring

## Reporting

* Detailed interview report
* Strengths and weaknesses analysis
* Improvement recommendations
* Interview history storage using SQLite

---

# Project Structure

```text
outputs/
│
├── resume_interview_api.py      # Main FastAPI application
├── run_interview_site.py        # Application launcher
├── requirements.txt            # Core dependencies
├── requirements-ml.txt         # Optional ML dependencies
└── README.md                   # Project documentation
```

---

# Setup

## Create Virtual Environment (Optional)

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## Install Core Dependencies

```bash
python -m pip install -r outputs/requirements.txt
```

---

## Install Optional ML Dependencies

For Whisper speech transcription and advanced embeddings:

```bash
python -m pip install -r outputs/requirements-ml.txt
```

The application can run without these dependencies.

If Whisper is unavailable, users can submit typed answers instead.

---

# Running The Application

From the project root directory:

```bash
python outputs/run_interview_site.py
```

You should see:

```text
Uvicorn running on http://127.0.0.1:8000
```

---

# Open The Application

Application:

```text
http://127.0.0.1:8000/
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

# How To Use

## Step 1: Upload Resume

Upload a resume in:

* PDF format
* TXT format

---

## Step 2: Select Target Role

Examples:

* AI Engineer
* Backend Engineer
* Data Scientist
* DevOps Engineer
* Machine Learning Engineer
* Full Stack Developer

---

## Step 3: Generate Interview

Click:

```text
Generate Interview
```

The system will:

* Parse the resume
* Extract skills and projects
* Generate technical interview questions

---

## Step 4: Answer Questions

For each question:

### Option A: Voice Answer

* Record your answer
* Browser microphone permission required

### Option B: Typed Answer

* Enter answer manually
* Useful when speech transcription is unavailable

---

## Step 5: Review Results

After completing all questions:

* View question-wise scores
* View overall interview score
* Review strengths and weaknesses
* Receive improvement suggestions

---

# Optional LLM Integration

The application works without an LLM API key.

Without an API key, the system uses:

* Local question generation
* Local answer evaluation

---

## Using Groq

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
$env:GROQ_MODEL="llama-3.1-70b-versatile"
```

### macOS / Linux

```bash
export GROQ_API_KEY="your_groq_api_key"
export GROQ_MODEL="llama-3.1-70b-versatile"
```

Restart the application after setting environment variables.

---

# Alternative LLM Setup (Gemini)

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="YOUR_GEMINI_API_KEY"
)
```

The Gemini integration can be used for:

* Technical question generation
* Ideal answer generation
* Feedback generation
* Career recommendations

---

# API Endpoints

## Health Check

```http
GET /health
```

---

## Home

```http
GET /
```

---

## Start Interview

```http
POST /interview/start
```

---

## Get Interview Session

```http
GET /interview/{session_id}
```

---

## Submit Answer

```http
POST /interview/{session_id}/answer
```

---

## Generate Final Report

```http
GET /interview/{session_id}/report
```

---

# Scoring System

Each answer receives three independent scores:

## Semantic Score

Measures similarity between:

* Candidate answer
* Ideal answer

Uses:

* Sentence Transformers
* Cosine Similarity

---

## Technical Score

Measures:

* Technical accuracy
* Completeness
* Depth of explanation

---

## Confidence Score

Measures:

* Speaking fluency
* Response quality
* Communication indicators

---

## Final Score Formula

```python
final_score = (
    semantic_score * 0.45 +
    technical_score * 0.40 +
    confidence_score * 0.15
)
```

---

# Technologies Used

## Backend

* FastAPI
* Uvicorn

## NLP

* spaCy

## Resume Parsing

* pdfplumber

## Speech-to-Text

* Whisper (Optional)

## Embeddings

* Sentence Transformers
* all-MiniLM-L6-v2

## LLMs

* Gemini 2.5 Flash
* Groq Llama 3.1 (Optional)

## Database

* SQLite

---

# Local Data Storage

The application stores runtime data locally:

```text
work/
├── resume_interview.sqlite3

interview_audio/
├── recorded_answers/
```

---

# Troubleshooting

## Website Is Not Opening

Make sure the server is running:

```bash
python outputs/run_interview_site.py
```

Then open:

```text
http://127.0.0.1:8000/
```

Do not open the Python file directly in the browser.

---

## No Module Named "fastapi"

Install dependencies:

```bash
python -m pip install -r outputs/requirements.txt
```

---

## Voice Recording Not Working

Use a modern browser:

* Google Chrome
* Microsoft Edge

Allow microphone permissions when prompted.

---

## Whisper Not Installed

Install ML dependencies:

```bash
python -m pip install -r outputs/requirements-ml.txt
```

Or use the typed-answer option.

---

# Security Notes

Do NOT commit the following to a public repository:

* Personal resumes
* Audio recordings
* API keys
* SQLite database files
* Environment variable files (.env)

Use environment variables for all secret credentials.

---

# Future Enhancements

* Webcam-based confidence analysis
* Eye-contact detection
* ATS Resume Scoring
* Company-specific interview modes
* Coding interview module
* System design interview mode
* Personalized learning roadmap
* Career recommendation engine

---

# License

This project is intended for educational and research purposes.
