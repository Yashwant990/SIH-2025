from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import sqlite3
import os
import pickle
from dotenv import load_dotenv
import requests
import numpy as np
import re
from werkzeug.utils import secure_filename

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
ALLOWED_EXTENSIONS = {'txt'}
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

    # Profile table - using INSERT OR REPLACE for SQLite compatibility
    c.execute("""CREATE TABLE IF NOT EXISTS user_profile (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE,
        interest TEXT,
        time_per_week TEXT,
        current_education TEXT,
        skills_want_to_learn TEXT,
        career_goal TEXT
    )""")

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
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None
    except sqlite3.Error as e:
        print(f"Database error in create_user: {e}")
        return None

def get_user(email, password=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        if password:
            c.execute("SELECT id, username FROM users WHERE email=? AND password=?", (email, password))
        else:
            c.execute("SELECT id, username FROM users WHERE email=?", (email,))
        row = c.fetchone()
        conn.close()
        return row if row else None
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

# ----------------- Routes -----------------
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        if request.method == "POST":
            interest = request.form.get("interest", "").strip()
            time_per_week = request.form.get("time_per_week", "").strip()
            current_education = request.form.get("current_education", "").strip()
            skills_want_to_learn = request.form.get("skills_want_to_learn", "").strip()
            career_goal = request.form.get("career_goal", "").strip()

            # Use INSERT OR REPLACE for SQLite compatibility
            c.execute("""INSERT OR REPLACE INTO user_profile
                (user_id, interest, time_per_week, current_education, skills_want_to_learn, career_goal)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session["user_id"], interest, time_per_week, current_education, skills_want_to_learn, career_goal))
            conn.commit()
            flash("Profile updated successfully!", "success")

        # Fetch profile if exists
        c.execute("SELECT interest, time_per_week, current_education, skills_want_to_learn, career_goal FROM user_profile WHERE user_id=?", (session["user_id"],))
        profile_data = c.fetchone()
        conn.close()

        return render_template("profile.html", profile=profile_data)
    except sqlite3.Error as e:
        print(f"Database error in profile: {e}")
        flash("Database error occurred. Please try again.", "error")
        return render_template("profile.html", profile=None)

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
    return render_template("dashboard.html", skills=[], careers=[], aspiration="")

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if "user_id" not in session:
        return redirect(url_for("login"))

    file = request.files.get("file")
    if not file or file.filename == '':
        flash("No file selected.", "error")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename):
        flash("Only .txt files are allowed.", "error")
        return redirect(url_for("dashboard"))

    try:
        text = file.read().decode("utf-8", errors="ignore").strip()
        if not text:
            flash("File is empty or unreadable.", "error")
            return redirect(url_for("dashboard"))

        # Extract skills + careers
        skills = extract_skills(text)
        careers = ml_recommend_career(skills, top_n=3) if skills else []

        # Save results in session
        session["last_skills"] = skills
        session["last_careers"] = careers

        return redirect(url_for("results"))

    except Exception as e:
        print(f"File processing error: {e}")
        flash("Error processing file. Please try again.", "error")
        return redirect(url_for("dashboard"))


@app.route("/results")
def results():
    if "user_id" not in session:
        return redirect(url_for("login"))

    skills = session.get("last_skills", [])
    careers = session.get("last_careers", [])
    return render_template("results.html", skills=skills, careers=careers)



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