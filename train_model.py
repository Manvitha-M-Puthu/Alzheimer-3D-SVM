import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# --- CONFIGURATION ---
INPUT_CSV = "dementia_features.csv"

def run_pipeline():
    # 1. Load Data
    if not os.path.exists(INPUT_CSV):
        print("❌ Error: Feature file not found. Run Step 1 first!")
        return
        
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} subjects.")
    
    # Prepare X (Features) and y (Labels)
    # First, drop the known non-feature columns
    X = df.drop(columns=['Subject', 'Label'], errors='ignore')
    y = df['Label']
    
    # --- FIX: FORCE ONLY NUMERIC COLUMNS ---
    # This automatically removes 'Hash' strings or metadata that crashed the script
    X = X.select_dtypes(include=[np.number])
    print(f"Features used for training: {X.shape[1]} (Non-numeric columns dropped)")

    # Encode Labels: CN=0, MCI=1, AD=2
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    print(f"Classes detected: {classes}")
    
    # 2. DISCRETIZATION (Optimization Step)
    print("⚙️ Applying Discretization (KBins)...")
    try:
        # Quantile strategy ensures bins have equal number of samples
        est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        X_disc = est.fit_transform(X)
    except Exception as e:
        print(f"  Warning: Discretization issue ({e}), using raw values.")
        X_disc = X

    # 3. PCA (Feature Selection)
    print("⚙️ Applying PCA (13 Components)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_disc)
    
    # Safety check: Ensure we don't ask for more components than we have samples/features
    n_components = min(13, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. CLASSIFICATION (SVM)
    # [cite_start]SVM with RBF kernel [cite: 16]
    svm = SVC(kernel='rbf', probability=True, C=1.0)
    
    # [cite_start]5-Fold Cross Validation [cite: 16]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n🧠 Training SVM Model with 5-Fold CV...")
    
    # Get predictions for all folds
    y_pred = cross_val_predict(svm, X_pca, y_enc, cv=cv)
    # Get probabilities for ROC Curve
    y_prob = cross_val_predict(svm, X_pca, y_enc, cv=cv, method='predict_proba')

    # --- REPORTING RESULTS ---
    acc = accuracy_score(y_enc, y_pred)
    print(f"\n🏆 Overall Accuracy: {acc*100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(y_enc, y_pred, target_names=classes))
    
    # --- PLOT 1: Confusion Matrix ---
    cm = confusion_matrix(y_enc, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'SVM Confusion Matrix\nAccuracy: {acc*100:.1f}%')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("  Saved: confusion_matrix.png")
    
    # --- PLOT 2: ROC Curve ---
    y_bin = label_binarize(y_enc, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    
    plt.figure(figsize=(8,6))
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{classes[i]} (AUC = {roc_auc:.2f})')
        except:
            pass 
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Replicating Paper Results)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png')
    print("  Saved: roc_curve.png")
    plt.show()

if __name__ == "__main__":
    run_pipeline()