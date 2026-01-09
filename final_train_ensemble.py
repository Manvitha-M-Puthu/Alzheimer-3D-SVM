import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import shap

# --- CONFIGURATION ---
INPUT_CSV = "final_features_with_scaling.csv" 

def run_ensemble_training():
    if not os.path.exists(INPUT_CSV):
        print("❌ Error: Feature file not found.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Filter: Binary Only (CN vs AD)
    df = df[df['Label'] != 'MCI'].copy()
    
    # 2. Fix Scaling Factor
    if 'scaling_factor' in df.columns:
        df['scaling_factor'] = df['scaling_factor'].abs()
        # Filter extreme outliers only
        df = df[(df['scaling_factor'] > 0.5) & (df['scaling_factor'] < 4.0)]

    # 3. Drop Diagnostics
    cols_to_drop = [c for c in df.columns if c.startswith('diagnostics_')]
    df = df.drop(columns=cols_to_drop)

    # Prepare Data
    X = df.drop(columns=['Subject', 'Label'], errors='ignore')
    X = X.select_dtypes(include=[np.number]) 
    y = df['Label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    print(f"Classes: {classes}")

   
    clf1 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    
    
    clf2 = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, C=10))
    ])
    
    
    clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    
    voting_clf = VotingClassifier(
        estimators=[('rf', clf1), ('svm', clf2), ('gb', clf3)],
        voting='soft'
    )

    print("\n🧠 Training Ensemble Model (RF + SVM + GB)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    
    y_pred = cross_val_predict(voting_clf, X, y_enc, cv=cv)
    y_prob = cross_val_predict(voting_clf, X, y_enc, cv=cv, method='predict_proba')

    # --- RESULTS ---
    acc = accuracy_score(y_enc, y_pred)
    print(f"\n🏆 FINAL ENSEMBLE ACCURACY: {acc*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_enc, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes)
    plt.title(f'Ensemble Accuracy: {acc*100:.1f}%')
    plt.savefig('ensemble_confusion_matrix.png')
    print("  Saved: ensemble_confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_enc, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='purple', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve (Ensemble)')
    plt.legend(loc="lower right")
    plt.savefig('ensemble_roc_curve.png')
    print("  Saved: ensemble_roc_curve.png")

    # --- SAFE SHAP (Using only Random Forest to avoid Crash) ---
    print("\n📊 Generating Feature Importance...")
    
    clf1.fit(X, y_enc)
    
    explainer = shap.TreeExplainer(clf1)
    shap_values = explainer.shap_values(X)

    # Safe Shape Handling
    if isinstance(shap_values, list):
        shap_data = shap_values[1]
    elif len(np.array(shap_values).shape) == 3:
        shap_data = shap_values[:, :, 1]
    else:
        shap_data = shap_values

    plt.figure()
    shap.summary_plot(shap_data, X, show=False)
    plt.title("Key Features (Derived from RF Expert)")
    plt.tight_layout()
    plt.savefig('ensemble_shap_summary.png')
    print("  Saved: ensemble_shap_summary.png")
    
    # Print Top Features
    if len(shap_data.shape) > 2: shap_data = shap_data[:, :, 0]
    vals = np.abs(shap_data).mean(0)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': vals
    }).sort_values(by='importance', ascending=False)
    
    print("\n📝 TOP PREDICTORS:")
    print(feature_importance.head(5))

if __name__ == "__main__":
    run_ensemble_training()