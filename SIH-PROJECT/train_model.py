import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

# Training data (demo dataset - can be expanded with real resumes)
X_train = [
    # Data & AI
    ["python", "sql", "machine learning"],
    ["sql", "python", "visualization"],
    ["python", "deep learning", "tensorflow"],
    ["r", "statistics", "data analysis"],

    # Web
    ["html", "css", "javascript"],
    ["html", "css", "react", "nodejs"],
    ["php", "mysql", "javascript"],

    # Backend
    ["python", "flask", "sql"],
    ["java", "spring", "mysql"],
    ["c#", "asp.net", "sqlserver"],

    # Game/AR/VR
    ["c++", "unity", "blender"],
    ["unity", "blender", "c#"],
    ["unreal engine", "c++", "3d modeling"],

    # Mobile
    ["java", "android", "xml"],
    ["kotlin", "android", "firebase"],
    ["swift", "ios", "xcode"],

    # Cloud & DevOps
    ["aws", "docker", "kubernetes"],
    ["azure", "ci/cd", "devops"],
    ["gcp", "terraform", "cloud architecture"],

    # Cybersecurity
    ["networking", "linux", "firewall"],
    ["penetration testing", "python", "cybersecurity"],
    ["ethical hacking", "malware analysis"],

    # UI/UX
    ["figma", "adobe xd", "prototyping"],
    ["sketch", "wireframing", "user research"],

    # Blockchain
    ["solidity", "ethereum", "smart contracts"],
    ["blockchain", "web3", "nfts"],

    # Others
    ["excel", "finance", "accounting"],
    ["marketing", "seo", "content writing"],
    ["salesforce", "crm", "customer support"]
]

y_train = [
    # Data & AI
    "Data Scientist",
    "Data Analyst",
    "AI Engineer",
    "Statistician",

    # Web
    "Web Developer",
    "Full Stack Developer",
    "Backend Developer",

    # Backend
    "Backend Developer",
    "Java Developer",
    "C# Developer",

    # Game/AR/VR
    "Game Developer",
    "AR/VR Developer",
    "Unreal Engine Developer",

    # Mobile
    "Android Developer",
    "Android Developer",
    "iOS Developer",

    # Cloud & DevOps
    "Cloud Engineer",
    "DevOps Engineer",
    "Cloud Architect",

    # Cybersecurity
    "Network Security Engineer",
    "Cybersecurity Analyst",
    "Ethical Hacker",

    # UI/UX
    "UI/UX Designer",
    "UI/UX Designer",

    # Blockchain
    "Blockchain Developer",
    "Blockchain Developer",

    # Others
    "Financial Analyst",
    "Digital Marketing Specialist",
    "CRM Specialist"
]


# Encode skills as binary features
mlb = MultiLabelBinarizer()
X_train_enc = mlb.fit_transform(X_train)

# Train classifier
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train_enc, y_train)

# Save model & encoder
with open("model.pkl", "wb") as f:
    pickle.dump((clf, mlb), f)

print("✅ Model trained and saved as model.pkl")
