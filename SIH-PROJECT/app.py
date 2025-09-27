from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import sqlite3
import os
import pickle
from dotenv import load_dotenv
import requests
import numpy as np
import re
from werkzeug.utils import secure_filename
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
# Load .env file
load_dotenv()

# Read keys with validation
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Validate required API keys
if not all([GOOGLE_API_KEY, SERP_API_KEY, GEMINI_API_KEY, GOOGLE_CSE_ID]):
    print("WARNING: Some API keys are missing. Check your .env file.")

app = Flask(__name__, instance_relative_config=True)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey-change-in-production")

# File upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt','pdf','docx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fetch_ncvct_data(query):
    if not SERP_API_KEY:
        return {}
    
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": f"{query} NSQF qualification site:nielit.gov.in",
        "api_key": SERP_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"SERP API error: {e}")
    return {}

# Ensure DB folder
os.makedirs(app.instance_path, exist_ok=True)
DB_PATH = os.path.join(app.instance_path, "db.sqlite3")

# Load ML model with error handling
try:
    with open("model.pkl", "rb") as f:
        clf, mlb = pickle.load(f)
    print("✅ ML model loaded successfully")
except FileNotFoundError:
    print("❌ model.pkl not found. Run train_model.py first.")
    clf, mlb = None, None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    clf, mlb = None, None

# ----------------- Database -----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users table with username + email
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT
    )""")

    # Progress table
    c.execute("""CREATE TABLE IF NOT EXISTS user_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        topic TEXT,
        step TEXT
    )""")

    # Profile table - extended for richer user profiling
    c.execute("""CREATE TABLE IF NOT EXISTS user_profile (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE,
        interest TEXT,
        time_per_week TEXT,
        current_education TEXT,
        skills_want_to_learn TEXT,
        career_goal TEXT,
        preferred_learning_style TEXT,   -- e.g. "visual", "hands-on", "reading"
        experience_level TEXT            -- e.g. "beginner", "intermediate", "advanced"
    )""")

    # NEW: User Goals Table
    c.execute("""CREATE TABLE IF NOT EXISTS user_goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        skill_name TEXT NOT NULL,
        target_date DATE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'active',
        completed_at DATETIME
    )""")

    # NEW: Roadmap Steps Table
    c.execute("""CREATE TABLE IF NOT EXISTS user_roadmap_steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_id INTEGER NOT NULL,
        step_title TEXT NOT NULL,
        description TEXT,
        estimated_hours INTEGER,
        order_index INTEGER,
        completed BOOLEAN DEFAULT 0,
        completed_at DATETIME,
        FOREIGN KEY(goal_id) REFERENCES user_goals(id) ON DELETE CASCADE
    )""")
    
    conn.commit()
    conn.close()

def upgrade_user_profile_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Add new columns if they don’t already exist
    try:
        c.execute("ALTER TABLE user_profile ADD COLUMN preferred_learning_style TEXT;")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        c.execute("ALTER TABLE user_profile ADD COLUMN experience_level TEXT;")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def save_progress(user_id, topic, step):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO user_progress (user_id, topic, step) VALUES (?, ?, ?)", (user_id, topic, step))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Database error in save_progress: {e}")
        return False

def get_progress(user_id, topic):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT step FROM user_progress WHERE user_id=? AND topic=?", (user_id, topic))
        rows = c.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except sqlite3.Error as e:
        print(f"Database error in get_progress: {e}")
        return []

def create_user(username, email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        hashed_pw = generate_password_hash(password)  # hash password
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                  (username, email, hashed_pw))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None
    
def get_user(email, password=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, username, password FROM users WHERE email=?", (email,))
        row = c.fetchone()
        conn.close()

        if row:
            user_id, username, hashed_pw = row
            if password is None or check_password_hash(hashed_pw, password):
                return (user_id, username)
        return None
    except sqlite3.Error as e:
        print(f"Database error in get_user: {e}")
        return None
# ----------------- Skill Extraction -----------------
def extract_skills(text):
    skills_db = [
        "python", "sql", "java", "c++", "unity", "blender", "flask", 
        "html", "css", "javascript", "machine learning", "c#", "react",
        "nodejs", "php", "mysql", "tensorflow", "deep learning", "r",
        "statistics", "data analysis", "android", "swift", "ios",
        "aws", "docker", "kubernetes", "azure", "linux", "figma",
        "adobe xd", "solidity", "blockchain", "excel", "salesforce"
    ]
    text_lower = text.lower()
    found_skills = []
    for skill in skills_db:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills

# ----------------- ML Recommendation -----------------
def ml_recommend_career(skills, top_n=3):
    if not clf or not mlb:
        # Fallback recommendations if model not loaded
        return [
            {"career": "Web Developer", "score": 0.75},
            {"career": "Data Analyst", "score": 0.65},
            {"career": "Backend Developer", "score": 0.60}
        ]
    
    try:
        # Transform input skills into binary features
        skills_enc = mlb.transform([skills])
        
        # Get probabilities for all classes
        probs = clf.predict_proba(skills_enc)[0]
        
        # Get top N careers
        top_indices = np.argsort(probs)[-top_n:][::-1]
        results = [
            {"career": clf.classes_[i], "score": round(float(probs[i]), 2)}
            for i in top_indices
        ]
        
        return results
    except Exception as e:
        print(f"ML prediction error: {e}")
        return [{"career": "General IT", "score": 0.50}]
    
def classify_learning_pace(goal):
    """
    goal = dict with total_steps, completed_steps, created_at, target_date
    Returns: Quick, Medium, or Slow Learner
    """
    try:
        if not goal["total_steps"]:
            return "No progress yet"

        from datetime import datetime
        now = datetime.now()
        created = datetime.strptime(goal["created_at"], "%Y-%m-%d %H:%M:%S")
        target = datetime.strptime(goal["target_date"], "%Y-%m-%d") if goal["target_date"] else None

        elapsed_days = (now - created).days + 1
        total_days = (target - created).days if target else 30  # default: 30 days
        progress_fraction = goal["completed_steps"] / goal["total_steps"]
        time_fraction = elapsed_days / total_days if total_days > 0 else 1

        pace_ratio = progress_fraction / time_fraction

        if pace_ratio >= 1.2:
            return "⚡ Quick Learner"
        elif pace_ratio >= 0.8:
            return "📈 Medium Learner"
        else:
            return "🐢 Slow Learner"
    except Exception as e:
        print("Error in classify_learning_pace:", e)
        return "Unknown"
    
def build_user_profile(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Fetch basic profile
    c.execute("""
        SELECT interest, time_per_week, current_education, skills_want_to_learn, career_goal, preferred_learning_style, experience_level
        FROM user_profile
        WHERE user_id=?
    """, (user_id,))
    row = c.fetchone()

    profile = {
        "interest": row[0] if row else None,
        "time_per_week": row[1] if row else None,
        "education": row[2] if row else None,
        "skills_want_to_learn": row[3] if row else None,
        "career_goal": row[4] if row else None,
        "learning_style": row[5] if row else None,
        "experience_level": row[6] if row else None
    }

    # Fetch active goals and compute pace
    c.execute("""
        SELECT id, skill_name, target_date, created_at,
               (SELECT COUNT(*) FROM user_roadmap_steps WHERE goal_id = g.id) as total_steps,
               (SELECT COUNT(*) FROM user_roadmap_steps WHERE goal_id = g.id AND completed = 1) as completed_steps
        FROM user_goals g
        WHERE user_id=? AND status='active'
    """, (user_id,))

    goals = []
    for g in c.fetchall():
        goals.append({
            "skill_name": g[1],
            "progress_percent": int(g[5]/g[4]*100) if g[4] else 0,
            "pace": classify_learning_pace({
                "total_steps": g[4],
                "completed_steps": g[5],
                "created_at": g[3],
                "target_date": g[2]
            })
        })

    profile["active_goals"] = goals

    # Fetch completed skills
    c.execute("SELECT skill_name FROM user_goals WHERE user_id=? AND status='completed'", (user_id,))
    profile["completed_skills"] = [r[0] for r in c.fetchall()]

    conn.close()
    return profile
@app.route("/learning_test", methods=["GET", "POST"])
def learning_test():
    if "user_id" not in session:
        return redirect(url_for("login"))

    questions = [
        {
            "id": 1,
            "text": "How many new concepts can you grasp in 1 hour?",
            "options": ["1–2 concepts", "3–5 concepts", "More than 5 concepts"]
        },
        {
            "id": 2,
            "text": "How quickly do you complete a practice exercise after reading instructions?",
            "options": ["Slowly", "Moderate", "Quickly"]
        },
        {
            "id": 3,
            "text": "How often do you need to revise the material before remembering it?",
            "options": ["Often", "Sometimes", "Rarely"]
        },
        {
            "id": 4,
            "text": "How comfortable are you with switching between different learning tasks?",
            "options": ["Difficult", "Medium", "Very comfortable"]
        }
    ]

    if request.method == "POST":
        total_score = 0
        for q in questions:
            answer = request.form.get(f"q{q['id']}")
            if answer == "Fast" or answer == "More than 5 concepts" or answer == "Quickly" or answer == "Rarely" or answer == "Very comfortable":
                total_score += 3
            elif answer == "Medium" or answer == "3–5 concepts" or answer == "Moderate" or answer == "Sometimes":
                total_score += 2
            else:
                total_score += 1

        if total_score >= 12:
            pace = "⚡ Quick Learner"
        elif total_score >= 8:
            pace = "⏳ Medium Learner"
        else:
            pace = "🐢 Slow Learner"

        # Save to user_profile
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE user_profile SET preferred_learning_style=? WHERE user_id=?", (pace, session["user_id"]))
        conn.commit()
        conn.close()

        flash(f"Your learning pace is classified as: {pace}", "success")
        return redirect(url_for("profile"))

    return render_template("learning_test.html", questions=questions)


@app.route("/api/profile_data")
def profile_data():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 403

    profile = build_user_profile(session["user_id"])
    return jsonify(profile)


# ----------------- Google Search Function -----------------
def google_search(query):
    """Google Custom Search implementation"""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("Google search API keys missing")
        return []

    try:
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "num": 10
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("items", [])
        else:
            print(f"Google search API error: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Google search error: {e}")
        return []

# ----------------- Gemini Summarizer Function -----------------
def summarize_trends(text):
    """Gemini AI summarization implementation"""
    if not text.strip():
        return "No content available to summarize."
        
    try:
        if not GEMINI_API_KEY:
            print("GEMINI_API_KEY missing. Returning raw text.")
            return f"Summary unavailable. Here's the raw content:\n{text[:500]}..."

        body = {
            "contents": [
                {
                    "parts": [
                        {"text": f"Summarize the key trends and emerging skills from these search snippets:\n{text}"}
                    ]
                }
            ]
        }

        model_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

        response = requests.post(
            model_url,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=30
        )

        if not response.ok:
            if response.status_code == 429:
                print("Gemini quota exceeded. Returning raw text instead.")
            else:
                print(f"HTTP error from Gemini: {response.status_code} {response.reason}")
            return f"Summary unavailable. Here's the raw content:\n{text[:500]}..."

        data = response.json()

        # Extract summary from response
        summary = (data.get("candidates", [{}])[0]
                  .get("content", {})
                  .get("parts", [{}])[0]
                  .get("text", ""))

        if not summary:
            print("Gemini returned empty summary. Using raw text.")
            return f"Summary unavailable. Here's the raw content:\n{text[:500]}..."

        return summary

    except requests.exceptions.RequestException as e:
        print(f"Network error in summarize_trends: {e}")
        return f"Summary unavailable due to network error. Here's the raw content:\n{text[:500]}..."
    except Exception as e:
        print(f"Error in summarize_trends: {e}")
        return f"Summary unavailable due to an error. Here's the raw content:\n{text[:500]}..."
    
def generate_roadmap_for_goal(goal_id, skill_name, cursor):
    """Generate roadmap steps for a goal using predefined templates"""
    
    DEFAULT_ROADMAPS = {
        "Python": [
            {"title": "Variables & Data Types", "desc": "Learn basics of variables, strings, numbers", "hours": 5, "order": 1},
            {"title": "Control Flow", "desc": "If/else, loops, functions", "hours": 8, "order": 2},
            {"title": "Data Structures", "desc": "Lists, dicts, tuples, sets", "hours": 6, "order": 3},
            {"title": "OOP Concepts", "desc": "Classes, inheritance, methods", "hours": 10, "order": 4},
            {"title": "Projects", "desc": "Build 2 small projects", "hours": 15, "order": 5}
        ],
        "JavaScript": [
            {"title": "Syntax & DOM Basics", "desc": "Variables, functions, selecting elements", "hours": 6, "order": 1},
            {"title": "Events & Forms", "desc": "Handling clicks, inputs, validation", "hours": 7, "order": 2},
            {"title": "ES6+ Features", "desc": "Arrow functions, destructuring, modules", "hours": 8, "order": 3},
            {"title": "Async JS", "desc": "Promises, async/await, fetch API", "hours": 10, "order": 4},
            {"title": "Mini Projects", "desc": "Build a calculator, todo app", "hours": 12, "order": 5}
        ],
        "SQL": [
            {"title": "SELECT & WHERE", "desc": "Basic queries, filtering data", "hours": 4, "order": 1},
            {"title": "JOINs & GROUP BY", "desc": "Combine tables, aggregate data", "hours": 6, "order": 2},
            {"title": "Subqueries & CTEs", "desc": "Nested queries, WITH clauses", "hours": 7, "order": 3},
            {"title": "Indexes & Optimization", "desc": "Speed up queries", "hours": 5, "order": 4},
            {"title": "Real-world Practice", "desc": "Solve 10+ practice problems", "hours": 10, "order": 5}
        ]
    }

    # Fallback generic roadmap
    GENERIC_ROADMAP = [
        {"title": "Basics", "desc": "Learn fundamental concepts", "hours": 5, "order": 1},
        {"title": "Intermediate Topics", "desc": "Dive deeper into core features", "hours": 8, "order": 2},
        {"title": "Advanced Concepts", "desc": "Master complex techniques", "hours": 10, "order": 3},
        {"title": "Practice Projects", "desc": "Apply knowledge in real scenarios", "hours": 12, "order": 4},
        {"title": "Review & Polish", "desc": "Revise, refactor, and perfect", "hours": 5, "order": 5}
    ]

    roadmap = DEFAULT_ROADMAPS.get(skill_name.strip().title(), GENERIC_ROADMAP)

    for step in roadmap:
        cursor.execute("""
            INSERT INTO user_roadmap_steps (goal_id, step_title, description, estimated_hours, order_index)
            VALUES (?, ?, ?, ?, ?)
        """, (goal_id, step["title"], step["desc"], step["hours"], step["order"]))

def generate_ai_roadmap_for_goal(goal_id, skill_name, user_id, cursor):
    """Generate roadmap using Gemini AI based on user's full profile"""
    
    if not GEMINI_API_KEY:
        print("Gemini API key missing. Using fallback.")
        return False

    try:
        # Fetch full profile
        cursor.execute("""
            SELECT interest, time_per_week, current_education, skills_want_to_learn, career_goal
            FROM user_profile WHERE user_id = ?
        """, (user_id,))
        profile = cursor.fetchone()

        if not profile:
            return False

        interest, time_per_week, education, skills_str, career_goal = profile

        # Build context prompt
        prompt = f"""
You are a world-class learning coach. Create a personalized, step-by-step learning roadmap for a user who wants to learn "{skill_name}".

User Context:
- Interest: {interest or 'Not specified'}
- Available time per week: {time_per_week or 'Not specified'}
- Current education level: {education or 'Not specified'}
- Other skills they want to learn: {skills_str or 'None'}
- Long-term career goal: {career_goal or 'Not specified'}

Instructions:
1. Generate 5–8 clear, ordered learning steps.
2. Each step should have:
   - title (string)
   - description (1–2 sentences)
   - estimated_hours (integer, total hours needed for this step)
   - order_index (integer starting from 1)
3. Adjust depth and pace based on available time and education level.
4. If career goal is provided, align later steps toward it.
5. Return ONLY valid JSON in this exact format:
{{
  "roadmap": [
    {{
      "title": "Step 1 Title",
      "description": "Description here...",
      "estimated_hours": 5,
      "order_index": 1
    }}
  ]
}}

Do NOT add any other text or explanation.
"""

        body = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        model_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

        response = requests.post(
            model_url,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=30
        )

        if not response.ok:
            print(f"Gemini API error: {response.status_code} {response.text}")
            return False

        data = response.json()
        text_response = (data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", ""))

        if not text_response:
            print("Empty response from Gemini")
            return False

        # Clean and parse JSON (Gemini sometimes wraps in ```json ... ```)
        text_response = text_response.strip()
        if text_response.startswith("```json"):
            text_response = text_response[7:]  # Remove ```json
        if text_response.endswith("```"):
            text_response = text_response[:-3]

        roadmap_data = json.loads(text_response)

        # Validate structure
        if "roadmap" not in roadmap_data or not isinstance(roadmap_data["roadmap"], list):
            print("Invalid roadmap structure from AI")
            return False

        # Insert steps into DB
        for step in roadmap_data["roadmap"]:
            cursor.execute("""
                INSERT INTO user_roadmap_steps (goal_id, step_title, description, estimated_hours, order_index)
                VALUES (?, ?, ?, ?, ?)
            """, (
                goal_id,
                step.get("title", "Untitled Step")[:200],
                step.get("description", "")[:500],
                int(step.get("estimated_hours", 5)),
                int(step.get("order_index", 1))
            ))

        print(f"✅ AI roadmap generated for goal {goal_id}: {skill_name}")
        return True

    except Exception as e:
        print(f"Error in AI roadmap generation: {e}")
        return False
    
@app.route("/generate_career_roadmap", methods=["GET", "POST"])
def generate_career_roadmap():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        career_goal = request.form.get("career_goal", "").strip()

        if not career_goal:
            flash("Please enter a career goal.", "error")
            return render_template("career_roadmap_form.html")

        try:
            # Fetch user profile
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT interest, time_per_week, current_education, skills_want_to_learn, career_goal
                FROM user_profile WHERE user_id = ?
            """, (session["user_id"],))
            profile = c.fetchone()
            conn.close()

            if not profile:
                flash("Please complete your profile first.", "error")
                return render_template("career_roadmap_form.html")

            # Generate roadmap via Gemini
            roadmap_data = generate_career_roadmap_with_ai(career_goal, profile)

            if not roadmap_data:
                flash("Could not generate roadmap. Please try again.", "error")
                return render_template("career_roadmap_form.html")

            # Store in session temporarily (or cache in DB if needed later)
            session['temp_career_roadmap'] = {
                'target_career': career_goal,
                'phases': roadmap_data.get('phases', []),
                'generated_at': datetime.now().isoformat()
            }

            return redirect(url_for("view_career_roadmap"))

        except Exception as e:
            print(f"Error generating career roadmap: {e}")
            flash("An error occurred. Please try again.", "error")

    return render_template("career_roadmap_form.html")


def generate_career_roadmap_with_ai(target_career, profile_tuple):
    if not GEMINI_API_KEY:
        return None

    interest, time_per_week, education, skills_str, _ = profile_tuple

    prompt = f"""
You are a top-tier career coach. Create a comprehensive, phase-based learning roadmap for someone who wants to become a "{target_career}".

User Context:
- Interest: {interest or 'Not specified'}
- Available time per week: {time_per_week or 'Not specified'}
- Current education level: {education or 'Not specified'}
- Skills they already want to learn: {skills_str or 'None'}

Instructions:
1. Break the journey into 3–5 logical PHASES (e.g., "Foundation", "Core Skills", "Advanced Topics", "Portfolio & Job Prep").
2. Each phase should contain 2–4 KEY SKILLS to master.
3. For each skill, include:
   - skill_name (string)
   - description (1 sentence)
   - estimated_hours (integer)
4. Adjust depth and pacing based on available time and education level.
5. End with job-ready/portfolio advice if relevant.
6. Return ONLY valid JSON in this exact format:
{{
  "phases": [
    {{
      "phase_name": "Phase 1: Foundation",
      "skills": [
        {{
          "skill_name": "HTML & CSS",
          "description": "Build static web pages with structure and style.",
          "estimated_hours": 20
        }}
      ]
    }}
  ]
}}

Do NOT add any other text.
"""

    try:
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1  # Keep it deterministic
            }
        }

        model_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(model_url, headers={"Content-Type": "application/json"}, json=body, timeout=30)

        if not response.ok:
            print(f"Gemini error: {response.status_code} {response.text}")
            return None

        text_response = (response.json()
                        .get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", ""))

        # Clean JSON
        text_response = text_response.strip()
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]

        return json.loads(text_response)

    except Exception as e:
        print(f"AI generation error: {e}")
        return None
    
@app.route("/view_career_roadmap")
def view_career_roadmap():
    if "user_id" not in session:
        return redirect(url_for("login"))

    roadmap = session.get('temp_career_roadmap')
    if not roadmap:
        flash("No roadmap found. Please generate one first.", "error")
        return redirect(url_for("generate_career_roadmap"))

    return render_template("career_roadmap_display.html", roadmap=roadmap)

@app.route("/add_skill_to_goals", methods=["POST"])
def add_skill_to_goals():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 403

    skill_name = request.form.get("skill_name", "").strip()
    description = request.form.get("description", "").strip()
    estimated_hours = request.form.get("estimated_hours", 0)

    if not skill_name:
        flash("Skill name is required.", "error")
        return redirect(url_for("view_career_roadmap"))

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Insert new goal
        c.execute("""
            INSERT INTO user_goals (user_id, skill_name, target_date, status)
            VALUES (?, ?, NULL, 'active')
        """, (session["user_id"], skill_name))
        goal_id = c.lastrowid

        # Insert first step using description
        c.execute("""
            INSERT INTO user_roadmap_steps (goal_id, step_title, description, estimated_hours, order_index, completed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (goal_id, "Start Learning", description, int(estimated_hours), 1, 0))

        conn.commit()
        conn.close()

        flash(f"✅ Added '{skill_name}' to your learning goals!", "success")
        return redirect(url_for("profile"))

    except Exception as e:
        print(f"Error adding skill to goals: {e}")
        flash("Error adding skill. Please try again.", "error")
        return redirect(url_for("view_career_roadmap"))

# ----------------- Routes -----------------
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Handle profile form submission
    if request.method == "POST":
        interest = request.form.get("interest", "")
        time_per_week = request.form.get("time_per_week", "")
        current_education = request.form.get("current_education", "")
        skills_want_to_learn = request.form.get("skills_want_to_learn", "")
        career_goal = request.form.get("career_goal", "")

        c.execute("""
            INSERT OR REPLACE INTO user_profile
            (user_id, interest, time_per_week, current_education, skills_want_to_learn, career_goal)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session["user_id"], interest, time_per_week, current_education, skills_want_to_learn, career_goal))
        conn.commit()

    # Fetch profile including learning pace
    c.execute("""
        SELECT interest, time_per_week, current_education, skills_want_to_learn, career_goal, preferred_learning_style
        FROM user_profile
        WHERE user_id=?
    """, (session["user_id"],))
    profile_data = c.fetchone()

    # Fetch active goals
    c.execute("""
        SELECT id, skill_name, target_date,
               (SELECT COUNT(*) FROM user_roadmap_steps WHERE goal_id = g.id) as total_steps,
               (SELECT COUNT(*) FROM user_roadmap_steps WHERE goal_id = g.id AND completed = 1) as completed_steps
        FROM user_goals g
        WHERE user_id=? AND status='active'
    """, (session["user_id"],))
    active_goals = []
    for g in c.fetchall():
        progress_percent = int(g[4]/g[3]*100) if g[3] else 0
        active_goals.append({
            "id": g[0],
            "skill_name": g[1],
            "target_date": g[2],
            "total_steps": g[3],
            "completed_steps": g[4],
            "progress_percent": progress_percent
        })

    # Fetch completed goals
    c.execute("SELECT skill_name, completed_at FROM user_goals WHERE user_id=? AND status='completed'", (session["user_id"],))
    completed_goals = [{"skill_name": r[0], "completed_at": r[1]} for r in c.fetchall()]

    conn.close()

    return render_template("profile.html",
                           profile=profile_data,
                           active_goals=active_goals,
                           completed_goals=completed_goals)

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not all([username, email, password]):
            flash("All fields are required.", "error")
            return render_template("signup.html")

        user_id = create_user(username, email, password)
        if user_id:
            session["user_id"] = user_id
            session["username"] = username
            session["email"] = email
            flash("Account created successfully!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Username or Email already exists!", "error")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not all([email, password]):
            flash("Email and password are required.", "error")
            return render_template("login.html")

        user = get_user(email, password)
        if user:
            session["user_id"] = user[0]
            session["username"] = user[1]
            session["email"] = email
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials!", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    # Use get() instead of pop() to keep the data in session
    analysis_results = session.get('analysis_results', None)
    
    if analysis_results:
        skills = analysis_results.get('skills', [])
        careers = analysis_results.get('careers', [])
        aspiration = analysis_results.get('aspiration', '')
    else:
        skills = []
        careers = []
        aspiration = ""
    
    return render_template("dashboard.html", skills=skills, careers=careers, aspiration=aspiration)

@app.route("/clear_analysis")
def clear_analysis():
    """Clear analysis results and return to clean dashboard"""
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    # This explicitly clears the analysis results
    session.pop('analysis_results', None)
    return redirect(url_for("dashboard"))
@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if 'file' not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("dashboard"))

    file = request.files['file']
    if file.filename == '':
        flash("No file selected.", "error")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename):
        flash("Only .txt files are allowed.", "error")
        return redirect(url_for("dashboard"))

    try:
        # Read file content
        text = file.read().decode("utf-8", errors="ignore").strip()
        
        if not text:
            flash("File is empty or unreadable.", "error")
            return redirect(url_for("dashboard"))

        # Auto-detect if it's a skill list or resume
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]

        if len(lines) > 3 and all(len(line.split()) <= 3 for line in lines):
            # Looks like a skills list
            skills = lines[:20]  # Limit to prevent overflow
        else:
            # Looks like a resume
            skills = extract_skills(text)

        aspiration = request.form.get("aspiration", "").strip()

        # ML Predictions
        careers = ml_recommend_career(skills, top_n=3) if skills else []

        # Store the results in session
        session['analysis_results'] = {
            'skills': skills,
            'careers': careers,
            'aspiration': aspiration
        }

        flash(f"Analysis complete! Found {len(skills)} skills.", "success")
        
        # Redirect to dashboard (PRG pattern)
        return redirect(url_for("dashboard"))

    except Exception as e:
        print(f"File processing error: {e}")
        flash("Error processing file. Please try again.", "error")
        return redirect(url_for("dashboard"))
@app.route("/roadmap/<topic>")
def show_roadmap(topic):
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Career roadmaps data
    career_roadmaps = {
        "Web Developer": ["HTML", "CSS", "JavaScript", "Python", "SQL"],
        "Data Scientist": ["Python", "SQL", "Machine Learning", "Deep Learning"],
        "Game Developer": ["C++", "Unity", "Blender"],
        "AR/VR Developer": ["Unity", "Blender", "C#"],
        "Backend Developer": ["Python", "Flask", "SQL"],
        "Data Analyst": ["SQL", "Python", "Visualization Tools"]
    }

    roadmap = career_roadmaps.get(topic, [])
    completed = get_progress(session["user_id"], topic)
    
    # Fetch NCVCT data
    data = fetch_ncvct_data(topic)

    results = []
    if "organic_results" in data:
        for r in data["organic_results"][:10]:  # Limit results
            snippet = r.get("snippet", "")
            title = r.get("title", "")
            link = r.get("link", "")
            text = f"{title} {snippet}"

            # Extract QP Code
            match_qp = re.search(r"QP\s*Code[:\s]*([A-Za-z0-9/]+)", text, re.IGNORECASE)
            qp_code = match_qp.group(1) if match_qp else "N/A"

            # Extract NSQF Level
            match_level = re.search(r"NSQF\s*Level[:\s]*([0-9]+)", text, re.IGNORECASE)
            nsqf_level = match_level.group(1) if match_level else "N/A"

            results.append({
                "qualification": title,
                "qp_code": qp_code,
                "nsqf_level": nsqf_level,
                "details": snippet,
                "link": link
            })

    return render_template(
        "roadmap.html",
        topic=topic,
        roadmap=roadmap,
        completed=completed,
        user=session.get("username", "User"),
        roadmap_data=results
    )

@app.route("/save_progress", methods=["POST"])
def save_progress_route():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 403

    try:
        data = request.get_json()
        topic = data.get("topic", "").strip()
        step = data.get("step", "").strip()
        
        if not topic or not step:
            return jsonify({"error": "Missing topic or step"}), 400
            
        success = save_progress(session["user_id"], topic, step)
        if success:
            return jsonify({"status": "saved"})
        else:
            return jsonify({"error": "Database error"}), 500
    except Exception as e:
        print(f"Save progress error: {e}")
        return jsonify({"error": "Server error"}), 500
    

@app.route("/set_learning_goal", methods=["GET", "POST"])
def set_learning_goal():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Fetch profile to suggest defaults
    c.execute("SELECT interest, skills_want_to_learn, career_goal FROM user_profile WHERE user_id=?", (session["user_id"],))
    profile_suggestion = c.fetchone()
    conn.close()

    suggested_skill = ""
    if profile_suggestion:
        # Suggest first skill from "skills_want_to_learn" or use interest
        skills_str = profile_suggestion[1] or profile_suggestion[0] or ""
        if skills_str:
            suggested_skill = skills_str.split(",")[0].strip()

    if request.method == "POST":
        skill_name = request.form.get("skill_name", "").strip()
        target_date = request.form.get("target_date", "").strip() or None

        if not skill_name:
            flash("Skill name is required.", "error")
            return render_template("set_learning_goal.html", suggested_skill=suggested_skill)

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()

            # Insert new goal
            c.execute("""
                INSERT INTO user_goals (user_id, skill_name, target_date)
                VALUES (?, ?, ?)
            """, (session["user_id"], skill_name, target_date))
            goal_id = c.lastrowid

            # 👇 NEW: Generate AI roadmap using full profile context
            success = generate_ai_roadmap_for_goal(goal_id, skill_name, session["user_id"], c)
            
            if not success:
                # Fallback to template
                generate_roadmap_for_goal(goal_id, skill_name, c)

            conn.commit()
            conn.close()

            flash(f"🎯 Smart roadmap generated for {skill_name}!", "success")
            return redirect(url_for("learning_roadmap", goal_id=goal_id))

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            flash("Error setting goal. Please try again.", "error")

    return render_template("set_learning_goal.html", suggested_skill=suggested_skill)

@app.route("/learning_roadmap/<int:goal_id>")
def learning_roadmap(goal_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Fetch goal with created_at
        c.execute("""
            SELECT id, skill_name, target_date, status, created_at,
                   (SELECT COUNT(*) FROM user_roadmap_steps WHERE goal_id = user_goals.id) as total_steps,
                   (SELECT COUNT(*) FROM user_roadmap_steps WHERE goal_id = user_goals.id AND completed = 1) as completed_steps
            FROM user_goals 
            WHERE id = ? AND user_id = ?
        """, (goal_id, session["user_id"]))
        goal_row = c.fetchone()

        if not goal_row:
            flash("Goal not found or access denied.", "error")
            return redirect(url_for("profile"))

        goal = {
            "id": goal_row[0],
            "skill_name": goal_row[1],
            "target_date": goal_row[2],
            "status": goal_row[3],
            "created_at": goal_row[4],
            "total_steps": goal_row[5],
            "completed_steps": goal_row[6],
            "progress_percent": int(goal_row[6] / goal_row[5] * 100) if goal_row[5] > 0 else 0,
            "pace": classify_learning_pace({
                "total_steps": goal_row[5],
                "completed_steps": goal_row[6],
                "created_at": goal_row[4],
                "target_date": goal_row[2]
            })
        }

        # Fetch steps
        c.execute("""
            SELECT id, step_title, description, estimated_hours, completed
            FROM user_roadmap_steps 
            WHERE goal_id = ? 
            ORDER BY order_index
        """, (goal_id,))
        steps = []
        for row in c.fetchall():
            steps.append({
                "id": row[0],
                "step_title": row[1],
                "description": row[2],
                "estimated_hours": row[3],
                "completed": bool(row[4])
            })

        conn.close()
        return render_template("learning_roadmap.html", goal=goal, roadmap_steps=steps)

    except sqlite3.Error as e:
        print(f"Database error in learning_roadmap: {e}")
        flash("Error loading roadmap.", "error")
        return redirect(url_for("profile"))

@app.route("/complete_step", methods=["POST"])
def complete_step():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 403

    try:
        data = request.get_json()
        step_id = data.get("step_id")

        if not step_id:
            return jsonify({"error": "Missing step_id"}), 400

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Verify step belongs to user
        c.execute("""
            SELECT g.user_id 
            FROM user_roadmap_steps rs
            JOIN user_goals g ON rs.goal_id = g.id
            WHERE rs.id = ?
        """, (step_id,))
        result = c.fetchone()

        if not result or result[0] != session["user_id"]:
            conn.close()
            return jsonify({"error": "Unauthorized"}), 403

        # Toggle completion
        c.execute("SELECT completed FROM user_roadmap_steps WHERE id = ?", (step_id,))
        current_status = c.fetchone()[0]
        new_status = 1 - current_status  # Toggle 0↔1
        completed_at = "datetime('now')" if new_status else "NULL"

        c.execute(f"""
            UPDATE user_roadmap_steps 
            SET completed = ?, completed_at = {completed_at}
            WHERE id = ?
        """, (new_status, step_id))

        # Get goal_id to check if all steps are done
        c.execute("SELECT goal_id FROM user_roadmap_steps WHERE id = ?", (step_id,))
        goal_id = c.fetchone()[0]

        # Check if all steps in goal are completed
        c.execute("""
            SELECT COUNT(*) as total, SUM(completed) as completed
            FROM user_roadmap_steps WHERE goal_id = ?
        """, (goal_id,))
        counts = c.fetchone()

        if counts[0] > 0 and counts[0] == counts[1]:  # All steps completed
            c.execute("""
                UPDATE user_goals 
                SET status = 'completed', completed_at = datetime('now')
                WHERE id = ?
            """, (goal_id,))

        conn.commit()
        conn.close()

        return jsonify({"status": "saved"})

    except Exception as e:
        print(f"Complete step error: {e}")
        return jsonify({"error": "Server error"}), 500

# ----------------- TRENDS ROUTES -----------------

@app.route("/trends")
def trends_page():
    """Main trends page - shows the search interface"""
    return render_template("trends.html")

@app.route("/trends/<career>")
def show_trends(career):
    """
    Legacy route for career-specific trends - redirects to new dynamic trends page
    This fixes the dashboard.html error when clicking career recommendations
    """
    return render_template("trends.html", initial_query=career)

@app.route("/api/trends", methods=["POST"])
def trends_api():
    """
    API endpoint for fetching trends - combines Google search + Gemini summarization
    """
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"summary": "Please provide a search query.", "results": []})

        # Enhanced query for better career trend results
        enhanced_query = f"{query} trends 2024 2025 skills emerging technologies career"

        # Google Custom Search
        search_results = google_search(enhanced_query)

        # Format results for frontend
        formatted_results = []
        search_text_for_summary = ""

        for item in search_results[:10]:  # Limit to top 10 results
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")

            formatted_results.append({
                "title": title,
                "snippet": snippet,
                "link": link
            })

            # Collect text for summary
            search_text_for_summary += f"{title}. {snippet}\n"

        # Generate summary using Gemini
        if search_text_for_summary.strip():
            summary = summarize_trends(search_text_for_summary)
        else:
            summary = "No relevant information found for this query. Try a different search term."

        return jsonify({
            "summary": summary,
            "results": formatted_results
        })

    except Exception as e:
        print(f"Error in trends_api: {e}")
        return jsonify({
            "summary": "Sorry, there was an error processing your request. Please try again later.",
            "results": []
        }), 500

# ----------------- ERROR HANDLERS -----------------

@app.errorhandler(413)
def too_large(e):
    flash("File too large. Maximum size is 16MB.", "error")
    return redirect(url_for("dashboard"))

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    print(f"Server error: {e}")
    flash("An internal server error occurred. Please try again.", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    init_db()
    print("🚀 Starting Career Guidance Platform...")
    print(f"✅ Database initialized at: {DB_PATH}")
    if clf and mlb:
        print("✅ ML model loaded successfully")
    else:
        print("⚠️  ML model not loaded - using fallback recommendations")
    
    app.run(debug=True)