import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_matrix(df, save_path=None):
    """
    Plots correlation matrix of Questions.
    Returns matplotlib figure.
    """
    question_cols = [f'Q{i}' for i in range(1, 22)]
    corr = df[question_cols].corr()
    
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix of DASS-21 Questions")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Correlation plot saved to {save_path}")
    
    return fig

def plot_severity_distribution(df, save_path=None):
    """
    Plots the count of severity levels for D, A, S.
    Returns matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.countplot(x='Depression_Level', data=df, ax=axes[0], order=['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])
    axes[0].set_title('Depression Severity')
    axes[0].tick_params(axis='x', rotation=45)

    sns.countplot(x='Anxiety_Level', data=df, ax=axes[1], order=['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])
    axes[1].set_title('Anxiety Severity')
    axes[1].tick_params(axis='x', rotation=45)

    sns.countplot(x='Stress_Level', data=df, ax=axes[2], order=['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])
    axes[2].set_title('Stress Severity')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Severity plot saved to {save_path}")
    
    return fig
