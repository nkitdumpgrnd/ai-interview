Resume AI Technical Interview
A Python FastAPI web app that parses a resume, generates resume-based technical interview questions, asks them on screen, records or accepts answers, and gives a score with feedback.
Features
Upload a PDF or TXT resume
Extract technical skills from the resume
Extract project-related details
Generate up to 7 technical interview questions
Ask questions one by one on screen
Record voice answers in the browser
Accept typed answers if voice transcription is unavailable
Compare candidate answers with ideal answers
Score semantic match, technical match, confidence, and final performance
Save interview sessions locally with SQLite
Project Structure
outputs/
  resume_interview_api.py   Main FastAPI app
  run_interview_site.py     App launcher
  requirements.txt          Core dependencies
  requirements-ml.txt       Optional ML dependencies
  README.md                 Project instructions
Setup
Create and activate a virtual environment if you want an isolated setup:
python -m venv .venv
Windows:
.venv\Scripts\activate
macOS/Linux:
source .venv/bin/activate
Install core dependencies:
python -m pip install -r outputs/requirements.txt
Optional: install ML dependencies for Whisper and better embeddings:
python -m pip install -r outputs/requirements-ml.txt
The app can still run without the optional ML dependencies. If Whisper is not installed, use the typed answer box.
Run The App
From the project root, run:
python outputs/run_interview_site.py
Keep the terminal open. You should see:
Uvicorn running on http://127.0.0.1:8000
Open the website:
http://127.0.0.1:8000/
API docs:
http://127.0.0.1:8000/docs
How To Use
Open http://127.0.0.1:8000/.
Upload your resume as a PDF or TXT file.
Enter your target role, such as:AI Engineer
Backend Engineer
Data Scientist
DevOps Engineer

Click Generate Interview.
Answer each technical question.
Use voice recording or type your answer.
Submit each answer.
View your final score and feedback after all questions are complete.
Optional LLM Setup
The app works without an LLM key by using fallback question generation and local answer scoring.
To use Groq for better generated questions, set:
Windows PowerShell:
$env:GROQ_API_KEY="your_groq_api_key"
$env:GROQ_MODEL="llama-3.1-70b-versatile"
macOS/Linux:
export GROQ_API_KEY="your_groq_api_key"
export GROQ_MODEL="llama-3.1-70b-versatile"
Then run the app again.
API Routes
GET  /
GET  /health
POST /interview/start
GET  /interview/{session_id}
POST /interview/{session_id}/answer
GET  /interview/{session_id}/report
Scoring
Each answer receives:
Semantic score
Technical score
Confidence score
Final score
Formula:
final_score =
  semantic_score * 0.45
  + technical_score * 0.40
  + confidence_score * 0.15
Troubleshooting
Website is not opening
Make sure the app is running:
python outputs/run_interview_site.py
Then open:
http://127.0.0.1:8000/
Do not open the .py file directly in the browser.
No module named fastapi
Install the core dependencies:
python -m pip install -r outputs/requirements.txt
Voice recording does not work
Use a modern browser such as Chrome or Edge and allow microphone permission.
If Whisper is not installed, type your answer instead.
Whisper is not installed
Install the optional ML dependencies:
python -m pip install -r outputs/requirements-ml.txt
Local Data
The app stores local runtime data here:
work/
  resume_interview.sqlite3
  interview_audio/
Do not commit personal resumes, audio recordings, API keys, or database files to a public repository.
