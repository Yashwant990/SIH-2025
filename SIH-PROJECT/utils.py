# utils.py
import sqlite3
import numpy as np
import pickle

# Define the database path. This should match where your Flask app is creating the DB.
# Based on app.py, the DB is created in the instance folder.
DB_PATH = "instance/db.sqlite3" 

# Extract skills
def extract_skills(text):
    """
    Extracts predefined skills found in the input text (e.g., resume content).
    """
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
def ml_recommend_career(skills, clf, mlb, top_n=3):
    """
    Predicts the top career paths based on a list of skills using the 
    provided classifier (clf) and label binarizer (mlb).
    """
    
    if not clf or not mlb:
        # Fallback recommendations if model not loaded (used by app.py and tasks.py)
        return [
            {"career": "Web Developer", "score": 0.75},
            {"career": "Data Analyst", "score": 0.65},
            {"career": "Backend Developer", "score": 0.60}
        ]
    
    try:
        # Transform input skills into binary features
        skills_enc = mlb.transform([skills])
        
        # Get probabilities for all classes
        # [0] selects the probabilities for the single input sample
        probs = clf.predict_proba(skills_enc)[0]
        
        # Get top N careers: argsort gets indices, [-top_n:] gets top N, [::-1] reverses to descending
        top_indices = np.argsort(probs)[-top_n:][::-1]
        results = [
            {"career": clf.classes_[i], "score": round(float(probs[i]), 2)}
            for i in top_indices
        ]
        
        return results
        
    except Exception as e:
        print(f"ML prediction error: {e}")
        # Return a generic fallback on ML error
        return [{"career": "General IT", "score": 0.50}]