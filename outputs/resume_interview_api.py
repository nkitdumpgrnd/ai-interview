import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import httpx
except Exception:
    httpx = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import spacy
except Exception:
    spacy = None

try:
    import whisper
except Exception:
    whisper = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


APP_NAME = "Resume AI Interview"
BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR.parent / "work"
AUDIO_DIR = WORK_DIR / "interview_audio"
DB_PATH = WORK_DIR / "resume_interview.sqlite3"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MAX_QUESTIONS = 7

SKILLS = sorted(
    {
        "python",
        "fastapi",
        "flask",
        "django",
        "sql",
        "postgresql",
        "mongodb",
        "sqlite",
        "redis",
        "docker",
        "kubernetes",
        "aws",
        "git",
        "rest api",
        "javascript",
        "typescript",
        "react",
        "machine learning",
        "deep learning",
        "nlp",
        "spacy",
        "bert",
        "transformers",
        "sentence transformers",
        "whisper",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "pandas",
        "numpy",
        "langchain",
        "rag",
        "vector database",
        "microservices",
        "ci/cd",
        "linux",
        "data structures",
        "algorithms",
        "system design",
    }
)

ROLE_SKILLS = {
    "ai": ["python", "machine learning", "nlp", "transformers", "rag", "vector database", "fastapi"],
    "backend": ["python", "fastapi", "postgresql", "mongodb", "rest api", "docker", "microservices"],
    "data": ["python", "sql", "pandas", "numpy", "scikit-learn", "machine learning"],
    "devops": ["linux", "docker", "kubernetes", "aws", "ci/cd", "git"],
    "software": ["python", "javascript", "data structures", "algorithms", "system design", "git"],
}


class ParsedResume(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    projects: List[Dict[str, Any]] = Field(default_factory=list)
    raw_text: str


class Question(BaseModel):
    id: int
    question: str
    ideal_answer: str
    topic: str
    source: str


class Evaluation(BaseModel):
    question_id: int
    transcript: str
    semantic_score: float
    technical_score: float
    confidence_score: float
    final_score: float
    feedback: str
    missing_concepts: List[str] = Field(default_factory=list)


class Session(BaseModel):
    id: str
    created_at: str
    target_role: str
    resume: ParsedResume
    questions: List[Question]
    answers: List[Evaluation] = Field(default_factory=list)
    status: str = "in_progress"
    final_score: Optional[float] = None
    final_feedback: Optional[str] = None


class HashEmbedder:
    def __init__(self, size: int = 384) -> None:
        self.size = size

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = True) -> List[List[float]]:
        return [self.embed(text, normalize_embeddings) for text in texts]

    def embed(self, text: str, do_norm: bool) -> List[float]:
        vec = [0.0] * self.size
        for token, count in Counter(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}", text.lower())).items():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.size
            vec[idx] += float(count)
        if do_norm:
            norm = sum(v * v for v in vec) ** 0.5
            if norm:
                vec = [v / norm for v in vec]
        return vec


class Resources:
    def __init__(self) -> None:
        self._nlp = None
        self._embedder = None
        self._whisper = None

    @property
    def nlp(self):
        if self._nlp is None and spacy is not None:
            try:
                self._nlp = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"))
            except OSError:
                self._nlp = spacy.blank("en")
                self._nlp.add_pipe("sentencizer")
        return self._nlp

    @property
    def embedder(self):
        if self._embedder is None:
            if SentenceTransformer is None:
                self._embedder = HashEmbedder()
            else:
                self._embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        return self._embedder

    @property
    def whisper(self):
        if self._whisper is None and whisper is not None:
            self._whisper = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
        return self._whisper


resources = Resources()
sessions: Dict[str, Session] = {}
app = FastAPI(title=APP_NAME, version="1.0.0")


def init_db() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )


@app.on_event("startup")
async def startup() -> None:
    init_db()


def to_dict(model: BaseModel) -> Dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def save_session(session: Session) -> None:
    sessions[session.id] = session
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, payload) VALUES (?, ?)",
            (session.id, json.dumps(to_dict(session), ensure_ascii=False)),
        )


def get_session(session_id: str) -> Session:
    if session_id in sessions:
        return sessions[session_id]
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT payload FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = Session(**json.loads(row[0]))
    sessions[session_id] = session
    return session


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


async def file_to_text(file: UploadFile) -> str:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Resume file is empty.")
    ext = Path(file.filename or "").suffix.lower()
    if ext == ".txt":
        return clean(data.decode("utf-8", errors="ignore"))
    if ext == ".pdf":
        if pdfplumber is None:
            raise HTTPException(status_code=500, detail="pdfplumber is not installed.")
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return clean("\n".join(page.extract_text() or "" for page in pdf.pages))
    raise HTTPException(status_code=400, detail="Upload a PDF or TXT resume.")


def extract_skills(text: str) -> List[str]:
    lower = norm(text)
    found = []
    for skill in SKILLS:
        if re.search(rf"(?<![a-z0-9+#.]){re.escape(skill)}(?![a-z0-9+#.])", lower):
            found.append(skill)
    return found


def extract_projects(text: str, skills: List[str]) -> List[Dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    project_lines = []
    capture = False
    for line in lines:
        heading = norm(re.sub(r"[:\-]+$", "", line))
        if heading in {"projects", "project", "personal projects", "academic projects"}:
            capture = True
            continue
        if capture and heading in {"skills", "education", "experience", "certifications"}:
            break
        if capture:
            project_lines.append(line)
    if not project_lines:
        project_lines = [line for line in lines if any(w in norm(line) for w in ["project", "built", "developed", "implemented"])][:10]
    text_block = " ".join(project_lines)
    chunks = re.split(r"\s*(?:\n|[-•*])\s*", text_block)
    projects = []
    for idx, chunk in enumerate([c for c in chunks if len(c.split()) > 5][:5], start=1):
        tech = [skill for skill in skills if skill in norm(chunk)]
        projects.append({"name": f"Project {idx}", "description": chunk[:700], "technologies": tech})
    return projects


def parse_resume(text: str) -> ParsedResume:
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    doc = resources.nlp(text[:50000]) if resources.nlp is not None else None
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"] if doc else []
    skills = extract_skills(text)
    return ParsedResume(
        name=people[0] if people else (lines[0] if lines else None),
        email=email.group(0) if email else None,
        skills=skills,
        projects=extract_projects(text, skills),
        raw_text=text,
    )


def role_skills(role: str) -> List[str]:
    role_lower = norm(role)
    for key, skills in ROLE_SKILLS.items():
        if key in role_lower:
            return skills
    return ROLE_SKILLS["software"]


async def llm_json(prompt: str) -> Optional[Any]:
    if httpx is None:
        return None
    if os.getenv("GROQ_API_KEY"):
        try:
            async with httpx.AsyncClient(timeout=40) as client:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                    json={
                        "model": os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
                        "messages": [
                            {"role": "system", "content": "Return valid JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                    },
                )
            content = res.json()["choices"][0]["message"]["content"]
            return json.loads(content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        except Exception:
            return None
    return None


async def make_questions(resume: ParsedResume, target_role: str) -> List[Question]:
    prompt = f"""
Generate exactly 7 technical interview questions from this resume.
Target role: {target_role}
Skills: {resume.skills}
Projects: {resume.projects}
Rules: only technical/project/skill questions, no HR questions.
Return JSON array with question, ideal_answer, topic, source.
"""
    result = await llm_json(prompt)
    questions = []
    if isinstance(result, list):
        for idx, item in enumerate(result[:MAX_QUESTIONS], 1):
            if isinstance(item, dict) and item.get("question") and item.get("ideal_answer"):
                questions.append(
                    Question(
                        id=idx,
                        question=str(item["question"]),
                        ideal_answer=str(item["ideal_answer"]),
                        topic=str(item.get("topic", "technical")),
                        source=str(item.get("source", "resume")),
                    )
                )
    return questions if len(questions) == MAX_QUESTIONS else fallback_questions(resume, target_role)


def fallback_questions(resume: ParsedResume, target_role: str) -> List[Question]:
    base_skills = resume.skills or role_skills(target_role)
    questions = []
    for project in resume.projects[:3]:
        tech = ", ".join(project.get("technologies") or base_skills[:3])
        questions.append(
            Question(
                id=len(questions) + 1,
                question=f"Explain the architecture and technical flow of {project.get('name', 'one project')} from your resume.",
                ideal_answer=f"A strong answer explains the goal, components, data flow, APIs, database/model choices, trade-offs, and how {tech} was used.",
                topic="project architecture",
                source="project",
            )
        )
    templates = [
        "How have you used {skill} in a real project?",
        "What are common debugging issues with {skill}, and how would you solve them?",
        "How would you test a feature built with {skill}?",
        "What performance or scalability concerns exist when using {skill}?",
        "Explain the trade-offs of choosing {skill}.",
        "What security or reliability checks would you add around {skill}?",
        "How would you explain {skill} implementation to a junior developer?",
    ]
    for template, skill in zip(templates, base_skills):
        if len(questions) >= MAX_QUESTIONS:
            break
        questions.append(
            Question(
                id=len(questions) + 1,
                question=template.format(skill=skill),
                ideal_answer=f"A strong answer defines {skill}, connects it to a project, explains implementation steps, testing, trade-offs, and concrete technical details.",
                topic="skill",
                source="resume skill",
            )
        )
    while len(questions) < MAX_QUESTIONS:
        questions.append(
            Question(
                id=len(questions) + 1,
                question="Describe one technical decision from your resume and justify it.",
                ideal_answer="A strong answer names the decision, alternatives, trade-offs, implementation details, and measurable result.",
                topic="technical decision",
                source="resume",
            )
        )
    return questions[:MAX_QUESTIONS]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    denom = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    return 0.0 if not denom else sum(x * y for x, y in zip(a, b)) / denom


def clamp(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def semantic(candidate: str, ideal: str) -> float:
    cand, ref = resources.embedder.encode([candidate, ideal], normalize_embeddings=True)
    return clamp((cosine(cand, ref) + 1) * 50)


def confidence(text: str) -> float:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    fillers = ["um", "uh", "like", "actually", "basically", "you know"]
    filler_count = sum(text.lower().count(word) for word in fillers)
    length_score = min(100, len(words) / 80 * 100)
    return clamp(length_score - min(35, filler_count * 5))


def missing_concepts(ideal: str, answer: str) -> List[str]:
    answer_lower = norm(answer)
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{3,}", ideal.lower())
    ignore = {"strong", "answer", "explains", "technical", "project", "details"}
    missing = [word for word in words if word not in answer_lower and word not in ignore]
    return [word for word, _ in Counter(missing).most_common(5)]


async def evaluate(question: Question, answer: str) -> Evaluation:
    sem = semantic(answer, question.ideal_answer)
    conf = confidence(answer)
    missing = missing_concepts(question.ideal_answer, answer)
    technical = sem
    final = sem * 0.45 + technical * 0.40 + conf * 0.15
    if final >= 80:
        feedback = "Good technical answer. You matched most expected concepts."
    elif final >= 60:
        feedback = "Decent answer. Add more implementation details, trade-offs, and examples."
    else:
        feedback = "Answer is too shallow. Explain the architecture, implementation, and reasoning more clearly."
    if missing:
        feedback += " Missing concepts: " + ", ".join(missing[:4]) + "."
    return Evaluation(
        question_id=question.id,
        transcript=answer,
        semantic_score=sem,
        technical_score=technical,
        confidence_score=conf,
        final_score=clamp(final),
        feedback=feedback,
        missing_concepts=missing,
    )


async def transcribe(audio: UploadFile, session_id: str, question_id: int) -> str:
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    suffix = Path(audio.filename or "answer.webm").suffix or ".webm"
    path = AUDIO_DIR / f"{session_id}_q{question_id}{suffix}"
    path.write_bytes(data)
    model = resources.whisper
    if model is None:
        raise HTTPException(status_code=400, detail="Whisper is not installed. Type your answer instead.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return clean(model.transcribe(tmp_path).get("text", ""))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def finish_if_done(session: Session) -> None:
    if len(session.answers) >= len(session.questions):
        session.status = "completed"
        session.final_score = round(sum(a.final_score for a in session.answers) / len(session.answers), 2)
        weak = [q.question_id for q in session.answers if q.final_score < 70]
        session.final_feedback = (
            f"Final Score: {session.final_score}/100\n\n"
            "Strengths: You completed the technical interview and answered resume-based questions.\n"
            "Improve: Add more implementation depth, trade-offs, testing details, and measurable results.\n"
            f"Lower-scoring question IDs: {weak or 'None'}"
        )


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Resume AI Interview</title>
  <style>
    body{margin:0;font-family:Arial,Helvetica,sans-serif;background:#f6f8fb;color:#17202a}
    header{background:white;border-bottom:1px solid #d8e0e8;padding:24px 30px 16px}
    main{display:grid;grid-template-columns:minmax(320px,420px) 1fr;gap:20px;padding:22px 30px}
    section{background:white;border:1px solid #d8e0e8;border-radius:8px;padding:18px}
    input,textarea,button{width:100%;font:inherit;box-sizing:border-box}
    input,textarea{border:1px solid #d8e0e8;border-radius:6px;padding:11px}
    textarea{min-height:130px;resize:vertical}
    label{display:block;font-weight:700;margin:14px 0 7px}
    button{margin-top:12px;border:0;border-radius:6px;padding:12px;background:#0f766e;color:white;font-weight:700;cursor:pointer}
    button.secondary{background:#334155}.danger{background:#b91c1c}.hidden{display:none}
    .badge{display:inline-block;border:1px solid #d8e0e8;border-radius:999px;padding:5px 9px;margin:0 6px 6px 0;color:#617080;font-size:12px}
    .question{border:1px solid #d8e0e8;border-radius:8px;padding:16px;background:#fbfdff;margin-bottom:14px}
    .score{display:inline-flex;width:104px;height:104px;border-radius:999px;border:9px solid #0f766e;align-items:center;justify-content:center;font-size:28px;font-weight:800}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.metric{border:1px solid #d8e0e8;border-radius:6px;padding:10px}.metric strong{display:block;font-size:20px}
    pre{white-space:pre-wrap;background:#f8fafc;border:1px solid #d8e0e8;border-radius:6px;padding:12px}
    @media(max-width:900px){main{grid-template-columns:1fr;padding:18px}}
  </style>
</head>
<body>
<header><h1>Resume AI Technical Interview</h1><p>Upload resume, generate 7 technical questions, answer by voice or text, and get scored feedback.</p></header>
<main>
<section>
  <form id="startForm">
    <label>Resume file</label><input name="resume" type="file" accept=".pdf,.txt" required />
    <label>Target role</label><input name="target_role" placeholder="AI Engineer, Backend Engineer" required />
    <button id="startBtn">Generate Interview</button>
  </form>
  <p id="status"></p><div id="summary"></div>
</section>
<section><div id="box"><h2>Interview</h2><p>Generate an interview to begin.</p></div></section>
</main>
<script>
const form=document.getElementById('startForm'), statusEl=document.getElementById('status'), box=document.getElementById('box'), summary=document.getElementById('summary');
let sessionId=null, questions=[], index=0, recorder=null, chunks=[], blob=null;
const esc=v=>String(v||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
const badges=items=>(items&&items.length?items.map(x=>`<span class="badge">${esc(x)}</span>`).join(''):'<span class="badge">None found</span>');
form.onsubmit=async e=>{
 e.preventDefault(); statusEl.textContent='Generating questions...';
 const res=await fetch('/interview/start',{method:'POST',body:new FormData(form)}); const data=await res.json();
 if(!res.ok){statusEl.textContent=data.detail||'Failed';return}
 sessionId=data.session_id; questions=data.questions; index=0;
 summary.innerHTML=`<h3>Resume Summary</h3><p><b>Name:</b> ${esc(data.parsed_resume.name)}</p><p><b>Skills:</b></p>${badges(data.parsed_resume.skills)}`;
 statusEl.textContent=`Generated ${questions.length} questions.`; renderQuestion();
};
function renderQuestion(){blob=null; const q=questions[index]; box.innerHTML=`
 <h2>Question ${index+1} of ${questions.length}</h2><div class="question"><span class="badge">${esc(q.topic)}</span><p><b>${esc(q.question)}</b></p></div>
 <button id="rec">Start Recording</button><button id="stop" class="danger hidden">Stop Recording</button>
 <label>Typed answer fallback</label><textarea id="answer" placeholder="Type answer here if you do not use voice."></textarea>
 <button id="submit">Submit Answer</button><div id="answerStatus"></div>`;
 document.getElementById('rec').onclick=startRecording; document.getElementById('stop').onclick=stopRecording; document.getElementById('submit').onclick=submitAnswer;
}
async function startRecording(){const s=document.getElementById('answerStatus');try{const stream=await navigator.mediaDevices.getUserMedia({audio:true});chunks=[];recorder=new MediaRecorder(stream);recorder.ondataavailable=e=>chunks.push(e.data);recorder.onstop=()=>{blob=new Blob(chunks,{type:'audio/webm'});stream.getTracks().forEach(t=>t.stop());s.innerHTML='<p>Recording ready.</p>'};recorder.start();document.getElementById('rec').classList.add('hidden');document.getElementById('stop').classList.remove('hidden');s.innerHTML='<p>Recording...</p>'}catch(err){s.innerHTML='<p>'+esc(err.message)+'</p>'}}
function stopRecording(){if(recorder&&recorder.state!=='inactive')recorder.stop();document.getElementById('rec').classList.remove('hidden');document.getElementById('stop').classList.add('hidden')}
async function submitAnswer(){const text=document.getElementById('answer').value.trim(), s=document.getElementById('answerStatus'); if(!blob&&!text){s.innerHTML='<p>Record or type an answer first.</p>';return} s.innerHTML='<p>Evaluating...</p>'; const fd=new FormData();fd.append('question_id',questions[index].id);fd.append('answer_text',text);if(blob)fd.append('audio',blob,'answer.webm'); const res=await fetch(`/interview/${sessionId}/answer`,{method:'POST',body:fd});const data=await res.json(); if(!res.ok){s.innerHTML='<p>'+esc(data.detail||'Failed')+'</p>';return} renderEval(data)}
function renderEval(data){const e=data.evaluation;if(data.completed){box.innerHTML=`<h2>Complete</h2><div class="score">${Math.round(data.final_score)}</div><h3>Final Feedback</h3><pre>${esc(data.final_feedback)}</pre>`;return}box.innerHTML=`<h2>Answer Score</h2><div class="score">${Math.round(e.final_score)}</div><div class="grid"><div class="metric">Semantic<strong>${Math.round(e.semantic_score)}%</strong></div><div class="metric">Technical<strong>${Math.round(e.technical_score)}%</strong></div><div class="metric">Confidence<strong>${Math.round(e.confidence_score)}%</strong></div><div class="metric">Question<strong>${index+1}/${questions.length}</strong></div></div><h3>Transcript</h3><pre>${esc(e.transcript)}</pre><h3>Feedback</h3><pre>${esc(e.feedback)}</pre><button id="next">Next Question</button>`;document.getElementById('next').onclick=()=>{index++;renderQuestion()}}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return HTML


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "app": APP_NAME, "whisper": whisper is not None, "sentence_transformers": SentenceTransformer is not None}


@app.post("/interview/start")
async def start_interview(resume: UploadFile = File(...), target_role: str = Form(...)) -> Dict[str, Any]:
    text = await file_to_text(resume)
    if len(text) < 80:
        raise HTTPException(status_code=400, detail="Resume text is too short.")
    parsed = parse_resume(text)
    questions = await make_questions(parsed, target_role)
    session = Session(id=str(uuid.uuid4()), created_at=datetime.utcnow().isoformat(), target_role=target_role, resume=parsed, questions=questions)
    save_session(session)
    return {"session_id": session.id, "parsed_resume": to_dict(parsed), "questions": [to_dict(q) for q in questions]}


@app.get("/interview/{session_id}")
async def read_interview(session_id: str) -> Dict[str, Any]:
    return to_dict(get_session(session_id))


@app.post("/interview/{session_id}/answer")
async def answer_question(session_id: str, question_id: int = Form(...), answer_text: str = Form(""), audio: Optional[UploadFile] = File(None)) -> Dict[str, Any]:
    session = get_session(session_id)
    if session.status == "completed":
        raise HTTPException(status_code=400, detail="Interview already completed.")
    question = next((q for q in session.questions if q.id == question_id), None)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found.")
    if any(a.question_id == question_id for a in session.answers):
        raise HTTPException(status_code=400, detail="Question already answered.")
    transcript = clean(answer_text)
    if audio is not None and audio.filename:
        transcript = await transcribe(audio, session_id, question_id)
    if len(transcript.split()) < 5:
        raise HTTPException(status_code=400, detail="Answer is too short.")
    ev = await evaluate(question, transcript)
    session.answers.append(ev)
    await finish_if_done(session)
    save_session(session)
    answered = {a.question_id for a in session.answers}
    next_q = next((q for q in session.questions if q.id not in answered), None)
    return {"evaluation": to_dict(ev), "next_question": to_dict(next_q) if next_q else None, "completed": session.status == "completed", "final_score": session.final_score, "final_feedback": session.final_feedback}


@app.get("/interview/{session_id}/report")
async def report(session_id: str) -> Dict[str, Any]:
    session = get_session(session_id)
    return {"session_id": session.id, "status": session.status, "final_score": session.final_score, "final_feedback": session.final_feedback, "answers": [to_dict(a) for a in session.answers]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("resume_interview_api:app", host="127.0.0.1", port=8000, reload=False)
