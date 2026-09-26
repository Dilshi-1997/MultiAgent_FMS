import os
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Application State (In production, replace this with a database like SQLite)
db_state = {}

# Load AI Model (Downloads automatically on first run)
print("Loading AI Matching Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def load_supervisors():
    supervisors = []
    with open('supervisors.csv', mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            supervisors.append(row)
    return supervisors

def send_html_email(to_email, subject, html_body):
    sender = os.getenv("SENDER_EMAIL")
    password = os.getenv("APP_PASSWORD")
    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(html_body, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print(f"[Email Sent] To: {to_email}")
        return True
    except Exception as e:
        print(f"[Email Failed] To: {to_email}. Error: {e}")
        return False

def dispatch_to_next_supervisor(student_id, base_url):
    state = db_state.get(student_id)
    if not state: return

    index = state['current_index']
    queue = state['queue']
    
    # If we exhausted the list of selected supervisors
    if index >= len(queue):
        fail_body = f"<p>Hello {state['details']['name']},</p><p>Unfortunately, all your selected supervisors declined. Please try again with different matches.</p>"
        send_html_email(state['details']['student_email'], "Supervisor Search Update", fail_body)
        return

    # Get current supervisor from queue
    current_sup_email = queue[index]
    supervisors = load_supervisors()
    supervisor = next((s for s in supervisors if s['Email'] == current_sup_email), None)

    if not supervisor:
        return

    # Build Email with Action Links
    accept_link = f"{base_url}respond/{student_id}/accept"
    decline_link = f"{base_url}respond/{student_id}/decline"
    file_link = f"{base_url}uploads/{state['file_name']}" if state['file_name'] else "No proposal attached."

    subject = f"Supervision Request: {state['details']['name']} - {state['details']['topic']}"
    html_body = f"""
    <h3>Research Supervision Request</h3>
    <p>Dear {supervisor['Name']},</p>
    <p>You have been requested as a supervisor for the following student project:</p>
    <ul>
        <li><b>Student:</b> {state['details']['name']} (SID: {state['details']['sid']})</li>
        <li><b>Topic:</b> {state['details']['topic']}</li>
        <li><b>Methodology:</b> {state['details']['methodology']}</li>
    </ul>
    <p><b>Proposal Document:</b> <a href="{file_link}">Click here to view/download</a></p>
    <br>
    <a href="{accept_link}" style="padding:10px 20px; background-color:green; color:white; text-decoration:none; border-radius:5px;">ACCEPT</a>
    &nbsp;&nbsp;&nbsp;
    <a href="{decline_link}" style="padding:10px 20px; background-color:red; color:white; text-decoration:none; border-radius:5px;">DECLINE</a>
    """
    send_html_email(supervisor['Email'], subject, html_body)

# ==========================================
# 2. HTML TEMPLATES (Inline for simplicity)
# ==========================================
FORM_HTML = """
<!DOCTYPE html>
<html>
<head><title>Find My Supervisor</title></head>
<body style="font-family: Arial; padding: 20px; max-width: 600px; margin: auto;">
    <h2>Find My Supervisor - Student Application</h2>
    <form action="/match" method="POST" enctype="multipart/form-data">
        <label>Name:</label><br><input type="text" name="name" required style="width:100%;"><br><br>
        <label>Student ID (SID):</label><br><input type="text" name="sid" required style="width:100%;"><br><br>
        <label>Email:</label><br><input type="email" name="student_email" required style="width:100%;"><br><br>
        <label>Faculty Name:</label><br><input type="text" name="faculty" required style="width:100%;"><br><br>
        <label>Department Name:</label><br><input type="text" name="department" required style="width:100%;"><br><br>
        <label>Research Topic:</label><br><input type="text" name="topic" required style="width:100%;"><br><br>
        <label>Research Interest (Keywords):</label><br><input type="text" name="interest" required style="width:100%;"><br><br>
        <label>Methodology:</label><br><input type="text" name="methodology" required style="width:100%;"><br><br>
        <label>Upload Proposal (.txt or .pdf):</label><br><input type="file" name="proposal" accept=".txt,.pdf"><br><br>
        <button type="submit" style="padding: 10px 20px; background-color: blue; color: white;">Find Matches</button>
    </form>
</body>
</html>
"""

SELECTION_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Select Supervisors</title>
    <script>
        function checkLimit(checkbox) {
            let checked = document.querySelectorAll('input[type="checkbox"]:checked');
            if(checked.length > 3) {
                checkbox.checked = false;
                alert("You can select a maximum of 3 supervisors.");
            }
        }
    </script>
</head>
<body style="font-family: Arial; padding: 20px; max-width: 800px; margin: auto;">
    <h2>Ranked Matches for: {{ topic }}</h2>
    <p>Select up to 3 supervisors in order of preference.</p>
    <form action="/submit_selection" method="POST">
        <input type="hidden" name="sid" value="{{ sid }}">
        {% for match in matches %}
            <div style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;">
                <input type="checkbox" name="supervisors" value="{{ match.Email }}" onclick="checkLimit(this)">
                <b>{{ match.Name }}</b> - <i>Match: {{ match.score }}%</i><br>
                Areas: {{ match.Research_Areas }} | Dept: {{ match.Department }}
            </div>
        {% endform %}
        <button type="submit" style="padding: 10px 20px; background-color: green; color: white;">Submit Selection</button>
    </form>
</body>
</html>
"""

# ==========================================
# 3. ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template_string(FORM_HTML)

@app.route('/match', methods=['POST'])
def match_supervisors():
    details = request.form.to_dict()
    sid = details['sid']
    
    # Handle File Upload
    file = request.files.get('proposal')
    file_name = None
    if file and file.filename:
        file_name = secure_filename(f"{sid}_{file.filename}")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
        
    db_state[sid] = {
        "details": details,
        "file_name": file_name,
        "queue": [],
        "current_index": 0
    }

    # Run Semantic Matching
    supervisors = load_supervisors()
    student_query = f"{details['topic']} {details['interest']} {details['methodology']}"
    student_embedding = model.encode([student_query])
    
    sup_texts = [s['Research_Areas'] for s in supervisors]
    sup_embeddings = model.encode(sup_texts)
    
    similarities = cosine_similarity(student_embedding, sup_embeddings)[0]
    
    matches = []
    for sup, score in zip(supervisors, similarities):
        sup['score'] = round(float(score) * 100, 2)
        matches.append(sup)
        
    matches.sort(key=lambda x: x['score'], reverse=True)

    return render_template_string(SELECTION_HTML, matches=matches, topic=details['topic'], sid=sid)

@app.route('/submit_selection', methods=['POST'])
def submit_selection():
    sid = request.form.get('sid')
    selected = request.form.getlist('supervisors') # Gets the checked emails
    
    if sid in db_state:
        db_state[sid]['queue'] = selected
        db_state[sid]['current_index'] = 0
        # Trigger email loop. request.host_url gives the base domain (e.g. http://127.0.0.1:5000/)
        dispatch_to_next_supervisor(sid, request.host_url)
        return "<h3>Success!</h3><p>Your request has been sent to your 1st preference.</p>"
    return "Error: Session expired."

@app.route('/respond/<sid>/<action>')
def handle_response(sid, action):
    state = db_state.get(sid)
    if not state:
        return "Invalid or expired request."

    current_email = state['queue'][state['current_index']]
    supervisors = load_supervisors()
    supervisor = next((s for s in supervisors if s['Email'] == current_email), None)

    if action == 'accept':
        success_body = f"""
        <h3>You have found a supervisor!</h3>
        <p><b>Name:</b> {supervisor['Name']}</p>
        <p><b>Email:</b> {supervisor['Email']}</p>
        <p><b>Research Areas:</b> {supervisor['Research_Areas']}</p>
        <p>Please reach out to them to begin your project.</p>
        """
        send_html_email(state['details']['student_email'], "Supervisor Confirmed!", success_body)
        return f"Thank you {supervisor['Name']}. You have accepted this student."

    elif action == 'decline':
        state['current_index'] += 1
        # request.host_url passes the domain so the next email has valid links
        dispatch_to_next_supervisor(sid, request.host_url)
        return "You have declined. The system is contacting the student's next preference."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded proposal file to the supervisor"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)