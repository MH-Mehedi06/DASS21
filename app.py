from flask import Flask, render_template, request
import pandas as pd
import os
import sys

# Ensure src can be imported
sys.path.append(os.getcwd())

from src.features import calculate_scores
from src.clustering import preprocess_for_clustering, run_kmeans
from src.data import load_and_clean_data

app = Flask(__name__)

# Load model data on startup
DATA_PATH = "DASS 21 Dataset.xlsx" # Or .csv

def get_kmeans_model():
    if os.path.exists(DATA_PATH):
        df = load_and_clean_data(DATA_PATH)
        df = calculate_scores(df)
        X_scaled = preprocess_for_clustering(df)
        kmeans, _, _ = run_kmeans(X_scaled, k=3)
        
        # We need the scaler too
        from sklearn.preprocessing import StandardScaler
        question_cols = [f'Q{i}' for i in range(1, 22)]
        scaler = StandardScaler()
        scaler.fit(df[question_cols])
        
        return kmeans, scaler
    return None, None

KMEANS_MODEL, SCALER = get_kmeans_model()

QUESTIONS = {
    "Q1": "I found it hard to wind down",
    "Q2": "I was aware of dryness of my mouth",
    "Q3": "I couldn't seem to experience any positive feeling at all",
    "Q4": "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)",
    "Q5": "I found it difficult to work up the initiative to do things",
    "Q6": "I tended to over-react to situations",
    "Q7": "I experienced trembling (e.g. in the hands)",
    "Q8": "I felt that I was using a lot of nervous energy",
    "Q9": "I was worried about situations in which I might panic and make a fool of myself",
    "Q10": "I felt that I had nothing to look forward to",
    "Q11": "I found myself getting agitated",
    "Q12": "I found it difficult to relax",
    "Q13": "I felt down-hearted and blue",
    "Q14": "I was intolerant of anything that kept me from getting on with what I was doing",
    "Q15": "I felt I was close to panic",
    "Q16": "I was unable to become enthusiastic about anything",
    "Q17": "I felt I wasn't worth much as a person",
    "Q18": "I felt that I was rather touchy",
    "Q19": "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)",
    "Q20": "I felt scared without any good reason",
    "Q21": "I felt that life was meaningless"
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', questions=QUESTIONS)

@app.route('/calculate', methods=['POST'])
def calculate():
    # 1. Collect User Responses
    user_responses = {}
    for q_id in QUESTIONS.keys():
        val = request.form.get(q_id)
        user_responses[q_id] = int(val) if val else 0
        
    # 2. Calculate Scores
    user_df = pd.DataFrame([user_responses])
    user_df = calculate_scores(user_df)
    
    # 3. Predict Cluster
    cluster_group = "Unknown"
    if KMEANS_MODEL and SCALER:
        question_cols = [f'Q{i}' for i in range(1, 22)]
        user_X_scaled = SCALER.transform(user_df[question_cols])
        cluster_group = KMEANS_MODEL.predict(user_X_scaled)[0]

    # 4. Prepare Result Data (Filtered as requested)
    full_results = {
        "Depression": {
            "Score": user_df['Depression_Score'][0],
            "Level": user_df['Depression_Level'][0]
        },
        "Anxiety": {
            "Score": user_df['Anxiety_Score'][0],
            "Level": user_df['Anxiety_Level'][0]
        },
        "Stress": {
            "Score": user_df['Stress_Score'][0],
            "Level": user_df['Stress_Level'][0]
        }
    }
    
    # Filter: Only keep categories that are NOT 'Normal'
    filtered_results = {k: v for k, v in full_results.items() if v['Level'] != 'Normal'}
    
    # If filtered_results is empty (meaning all are Normal), we might want to show that explicitly.
    # We will pass both full and filtered, but the template will use filtered primarily.
    # Or cleaner: Just pass filtered_results. If empty, template handles "All Normal".
    
    return render_template('result.html', result=filtered_results, cluster=cluster_group, all_normal=len(filtered_results)==0)

if __name__ == '__main__':
    app.run(debug=True)
