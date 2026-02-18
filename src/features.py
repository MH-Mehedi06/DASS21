import pandas as pd

def get_depression_label(score):
    if score <= 9: return 'Normal'
    elif score <= 13: return 'Mild'
    elif score <= 20: return 'Moderate'
    elif score <= 27: return 'Severe'
    else: return 'Extremely Severe'

def get_anxiety_label(score):
    if score <= 7: return 'Normal'
    elif score <= 9: return 'Mild'
    elif score <= 14: return 'Moderate'
    elif score <= 19: return 'Severe'
    else: return 'Extremely Severe'

def get_stress_label(score):
    if score <= 14: return 'Normal'
    elif score <= 18: return 'Mild'
    elif score <= 25: return 'Moderate'
    elif score <= 33: return 'Severe'
    else: return 'Extremely Severe'

def calculate_scores(df):
    """
    Calculates Depression, Anxiety, and Stress scores based on DASS-21 keys.
    Adds Score and Level columns.
    """
    df = df.copy()
    
    # Standard DASS-21 mapping
    depression_cols = ['Q3', 'Q5', 'Q10', 'Q13', 'Q16', 'Q17', 'Q21']
    anxiety_cols    = ['Q2', 'Q4', 'Q7', 'Q9', 'Q15', 'Q19', 'Q20']
    stress_cols     = ['Q1', 'Q6', 'Q8', 'Q11', 'Q12', 'Q14', 'Q18']

    # We sum the answers (0-3) and multiply by 2 as per DASS-21 protocol
    df['Depression_Score'] = df[depression_cols].sum(axis=1) * 2
    df['Anxiety_Score']    = df[anxiety_cols].sum(axis=1) * 2
    df['Stress_Score']     = df[stress_cols].sum(axis=1) * 2

    # Apply labels
    df['Depression_Level'] = df['Depression_Score'].apply(get_depression_label)
    df['Anxiety_Level']    = df['Anxiety_Score'].apply(get_anxiety_label)
    df['Stress_Level']     = df['Stress_Score'].apply(get_stress_label)
    
    return df
