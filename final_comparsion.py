import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
INPUT_CSV = "final_features_with_scaling.csv"

def run_comparison():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
        
    print(f"Loading Dataset: {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Cleaning
    cols_to_drop = [c for c in df.columns if c.startswith('diagnostics_')]
    df = df.drop(columns=cols_to_drop)
    
    # Filter for CN vs AD
    df = df[df['Label'].isin(['CN', 'AD'])]

    X = df.drop(columns=['Subject', 'Label'], errors='ignore')
    X = X.select_dtypes(include=[np.number]) 
    y = df['Label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    print(f"Comparison Dataset: {len(df)} Patients (CN vs AD)")
    print("-" * 50)

    # --- DEFINE MODELS ---
    # 1. Logistic Regression (Baseline)
    clf_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 2. Decision Tree (Simple but unstable)
    clf_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    # 3. SVM (The Strong Individual)
    clf_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, probability=True, class_weight='balanced', random_state=42))
    ])
    
    # 4. Random Forest (The Robust One)
    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, class_weight='balanced')
    
    # 5. Gradient Boosting (The Learner)
    clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    # 6. YOUR ENSEMBLE (The Champion)
    # We use the EXACT same settings that gave 70.65%
    ensemble_model = VotingClassifier(
        estimators=[('rf', clf_rf), ('svm', clf_svm), ('gb', clf_gb)],
        voting='soft'
    )

    models = {
        "Logistic Reg.": clf_lr,
        "Decision Tree": clf_dt,
        "SVM (RBF)": clf_svm,
        "Random Forest": clf_rf,
        "Grad. Boosting": clf_gb,
        "Voting Ensemble": ensemble_model
    }

    # --- TRAIN & COMPARE ---
    results = []
    names = []
    
    print(f"{'Model Name':<20} | {'Accuracy (Mean)'}")
    print("-" * 40)
    
    # We use Stratified K-Fold to be scientific
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y_enc, cv=cv, scoring='accuracy')
        acc_mean = scores.mean()
        
        results.append(acc_mean * 100)
        names.append(name)
        
        print(f"{name:<20} | {acc_mean*100:.2f}%")

    # --- PLOT COMPARISON ---
    plt.figure(figsize=(10, 6))
    
    # Create colors: Grey for others, Blue/Green for Ensemble
    colors = ['grey' if 'Ensemble' not in n else '#2ca02c' for n in names]
    
    bars = plt.bar(names, results, color=colors, alpha=0.8, edgecolor='black')
    
    # Add numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    plt.title('Model Performance Comparison (5-Fold CV)')
    plt.ylabel('Accuracy (%)')
    plt.ylim(40, 90) # Focus the y-axis
    plt.axhline(y=max(results), color='red', linestyle='--', alpha=0.5, label='Best Score')
    plt.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    save_path = 'Model_Comparison_Chart.png'
    plt.savefig(save_path, dpi=300)
    print("-" * 50)
    print(f"Comparison Chart saved to: {save_path}")
    

if __name__ == "__main__":
    run_comparison()