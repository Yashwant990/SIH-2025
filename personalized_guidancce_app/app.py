from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import sqlite3, os, pickle
import os
from dotenv import load_dotenv
import requests
import numpy as np
import re

# Load .env file
load_dotenv()

# Read keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")


app = Flask(__name__, instance_relative_config=True)
app.secret_key = "supersecretkey"

def fetch_ncvct_data(query):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": f"{query} NSQF qualification site:nielit.gov.in",
        "api_key": SERP_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return {}


# Ensure DB folder
os.makedirs(app.instance_path, exist_ok=True)
DB_PATH = os.path.join(app.instance_path, "db.sqlite3")

# Load ML model
with open("model.pkl", "rb") as f:
    clf, mlb = pickle.load(f)

   

# ----------------- Database -----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Drop old tables for dev (optional)
    # c.execute("DROP TABLE IF EXISTS users")
    # c.execute("DROP TABLE IF EXISTS user_progress")

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
    conn.commit()
    conn.close()

def save_progress(user_id, topic, step):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO user_progress (user_id, topic, step) VALUES (?, ?, ?)", (user_id, topic, step))
    conn.commit()
    conn.close()

def get_progress(user_id, topic):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT step FROM user_progress WHERE user_id=? AND topic=?", (user_id, topic))
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]

def create_user(username, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
        conn.commit()
    except sqlite3.IntegrityError:
        return None
    user_id = c.lastrowid
    conn.close()
    return user_id

def get_user(email, password=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if password:
        c.execute("SELECT id, username FROM users WHERE email=? AND password=?", (email, password))
    else:
        c.execute("SELECT id, username FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    return row if row else None  # returns (id, username)

# ----------------- Skill Extraction -----------------
def extract_skills(text):
    skills_db = ["python", "sql", "java", "c++", "unity", "blender",
                 "flask", "html", "css", "javascript", "machine learning", "c#"]
    text_lower = text.lower()
    return [skill for skill in skills_db if skill in text_lower]

# ----------------- ML Recommendation -----------------
def ml_recommend_career(skills, top_n=3):
    # Transform input skills into binary features
    skills_enc = mlb.transform([skills])

    # Get probabilities for all classes
    probs = clf.predict_proba(skills_enc)[0]

    # Get top N careers
    top_indices = np.argsort(probs)[-top_n:][::-1]
    results = [
        {"career": clf.classes_[i], "score": round(probs[i], 2)}
        for i in top_indices
    ]
    return results
# ----------------- Roadmaps & Trends -----------------
career_roadmaps = {
    "Web Developer": ["HTML", "CSS", "JavaScript", "Python", "SQL"],
    "Data Scientist": ["Python", "SQL", "Machine Learning", "Deep Learning"],
    "Game Developer": ["C++", "Unity", "Blender"],
    "AR/VR Developer": ["Unity", "Blender", "C#"],
    "Backend Developer": ["Python", "Flask", "SQL"],
    "Data Analyst": ["SQL", "Python", "Visualization Tools"]
}

career_trends = {
    "Web Developer": ["Web3", "Progressive Web Apps", "AI Integration"],
    "Data Scientist": ["Generative AI", "LLMs", "MLOps"],
    "Game Developer": ["Metaverse Games", "VR Gaming", "AI NPCs"],
    "AR/VR Developer": ["Immersive AR apps", "VR training tools"],
    "Backend Developer": ["Serverless Computing", "Microservices"],
    "Data Analyst": ["Data Storytelling", "BI Automation"]
}

# ----------------- Routes -----------------
@app.route("/")
def index():
    if "user_id" in session:
        # Instead of showing index.html, redirect straight to dashboard
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        user_id = create_user(username, email, password)
        if user_id:
            session["user_id"] = user_id
            session["username"] = username
            session["email"] = email
            return redirect(url_for("dashboard"))
        return "Username or Email already exists!"
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = get_user(email, password)
        if user:
            session["user_id"] = user[0]
            session["username"] = user[1]  # we store username for display
            session["email"] = email
            return redirect(url_for("dashboard"))
        return "Invalid credentials!"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", skills=[], career=None, aspiration="")

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if "user_id" not in session:
        return redirect(url_for("login"))

    file = request.files["file"]
    text = file.read().decode("utf-8", errors="ignore").strip()

    # Auto-detect if it's a skill list or resume
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]

    if len(lines) > 3 and all(len(line.split()) <= 3 for line in lines):
        # Looks like a skills list (short entries, one per line)
        skills = lines
    else:
        # Looks like a resume (paragraphs, longer text)
        skills = extract_skills(text)

    aspiration = request.form.get("aspiration", "")

    # ML Predictions
    careers = ml_recommend_career(skills, top_n=3)

    
    return render_template("dashboard.html", skills=skills, careers=careers, aspiration=aspiration)

@app.route("/roadmap/<topic>")
def show_roadmap(topic):
    if "user_id" not in session:
        return redirect(url_for("login"))

    roadmap = career_roadmaps.get(topic, [])
    completed = get_progress(session["user_id"], topic)
    data = fetch_ncvct_data(topic)   # use new fetch

    results = []
    if "organic_results" in data:
        for r in data["organic_results"]:
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
        user=session["username"],
        roadmap_data=results
    )

@app.route("/save_progress", methods=["POST"])
def save_progress_route():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 403

    data = request.get_json()
    save_progress(session["user_id"], data.get("topic"), data.get("step"))
    return jsonify({"status": "saved"})

@app.route("/trends/<career>")
def show_trends(career):
    return render_template("trends.html", career=career, trends=career_trends.get(career, ["No trends available"]))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
