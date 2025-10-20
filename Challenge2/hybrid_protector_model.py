"""
Hybrid "Protector" Model Architecture
4-Stage Advanced ML System for Jailbreak Detection
Target: 0.99+ ROC-AUC Score
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HYBRID PROTECTOR MODEL - 4-STAGE ARCHITECTURE")
print("="*80)

# ============================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================

def extract_advanced_features(text_series):
    """Enhanced feature extraction with more sophisticated patterns"""
    df = pd.DataFrame()
    
    # Basic text statistics
    df['char_length'] = text_series.str.len()
    df['num_words'] = text_series.str.split().apply(len)
    df['avg_word_length'] = text_series.apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)
    
    # Capitalization patterns
    df['num_upper'] = text_series.str.findall(r'[A-Z]').apply(len)
    df['upper_ratio'] = df['num_upper'] / (df['char_length'] + 1)
    df['num_title_words'] = text_series.apply(lambda x: sum(1 for w in str(x).split() if w.istitle()))
    
    # Special characters and punctuation
    df['num_special'] = text_series.str.findall(r'[!@#$%^&*()_+=\[\]{}|\\:;"\'<>,.?/~`-]').apply(len)
    df['num_exclamation'] = text_series.str.count(r'!')
    df['num_question'] = text_series.str.count(r'\?')
    df['num_quotes'] = text_series.str.count(r'["\'`]')
    
    # Jailbreak-specific patterns (expanded)
    jailbreak_keywords = [
        'ignore', 'bypass', 'system', 'filter', 'unfiltered', 'jailbreak', 'prompt',
        'override', 'admin', 'sudo', 'root', 'privilege', 'unrestricted', 'disable',
        'disregard', 'forget', 'pretend', 'roleplay', 'act as', 'simulate', 'emulate',
        'hypothetical', 'imagine', 'fictional', 'alternative', 'opposite', 'reverse'
    ]
    df['num_jailbreak_terms'] = text_series.apply(
        lambda x: sum(keyword in str(x).lower() for keyword in jailbreak_keywords)
    )
    
    # Command-like patterns
    df['has_code_block'] = text_series.str.contains(r'```|~~~', regex=True).astype(int)
    df['has_instruction'] = text_series.str.contains(r'\b(do|execute|run|perform|tell me|show me)\b', regex=True, case=False).astype(int)
    df['has_negation'] = text_series.str.contains(r'\b(not|no|don\'t|never|without)\b', regex=True, case=False).astype(int)
    
    # Sentence structure
    df['num_sentences'] = text_series.str.count(r'[.!?]+') + 1
    df['avg_sentence_length'] = df['num_words'] / df['num_sentences']
    
    # Repetition patterns
    df['has_repetition'] = text_series.str.contains(r'(\b\w+\b)(?:\s+\1\b)+', regex=True).astype(int)
    
    return df

# ============================================
# STAGE 0: DATA LOADING AND PREPARATION
# ============================================

print("\n[STAGE 0] Loading and Preparing Data...")
train = pd.read_csv('Dataset/train.csv')
test = pd.read_csv('Dataset/test.csv')

# Encode labels
label_mapping = {'benign': 0, 'jailbreak': 1}
train['label'] = train['label'].map(label_mapping)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    train['text'], train['label'], test_size=0.2, random_state=42, stratify=train['label']
)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
print(f"Class distribution - Val: {y_val.value_counts().to_dict()}")

# ============================================
# STAGE 1: CLASSICAL PIPELINE (TF-IDF + Meta + LR + LGBM)
# ============================================

print("\n" + "="*80)
print("[STAGE 1] Classical Pipeline - TF-IDF + Meta Features + LR + LGBM")
print("="*80)

# Extract advanced meta features
print("Extracting advanced meta features...")
X_train_meta = extract_advanced_features(X_train)
X_val_meta = extract_advanced_features(X_val)
X_test_meta = extract_advanced_features(test['text'])

# Scale meta features
scaler = StandardScaler()
X_train_meta_scaled = scaler.fit_transform(X_train_meta)
X_val_meta_scaled = scaler.transform(X_val_meta)
X_test_meta_scaled = scaler.transform(X_test_meta)

# TF-IDF features with optimized parameters
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 4),  # up to 4-grams for better context
    stop_words='english',
    min_df=2,
    max_df=0.95,
    sublinear_tf=True  # use log-scaled term frequency
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(test['text'])

# Combine features
X_train_classical = hstack([X_train_tfidf, X_train_meta_scaled])
X_val_classical = hstack([X_val_tfidf, X_val_meta_scaled])
X_test_classical = hstack([X_test_tfidf, X_test_meta_scaled])

# Train Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(C=10, solver='liblinear', max_iter=1000, random_state=42)
lr_model.fit(X_train_classical, y_train)
lr_val_preds = lr_model.predict_proba(X_val_classical)[:, 1]
lr_score = roc_auc_score(y_val, lr_val_preds)
print(f"✓ Logistic Regression ROC-AUC: {lr_score:.6f}")

# Train LightGBM on classical features
print("\nTraining LightGBM (Classical)...")
lgbm_classical_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42,
    'force_col_wise': True
}

lgb_train_classical = lgb.Dataset(X_train_classical, label=y_train)
lgb_val_classical = lgb.Dataset(X_val_classical, label=y_val, reference=lgb_train_classical)

lgbm_classical = lgb.train(
    lgbm_classical_params,
    lgb_train_classical,
    num_boost_round=2000,
    valid_sets=[lgb_val_classical],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(200)]
)

lgbm_classical_val_preds = lgbm_classical.predict(X_val_classical, num_iteration=lgbm_classical.best_iteration)
lgbm_classical_score = roc_auc_score(y_val, lgbm_classical_val_preds)
print(f"✓ LightGBM (Classical) ROC-AUC: {lgbm_classical_score:.6f}")

# ============================================
# STAGE 2: SEMANTIC EMBEDDING MODEL
# ============================================

print("\n" + "="*80)
print("[STAGE 2] Semantic Embedding Model - Sentence Transformers")
print("="*80)

print("Loading sentence transformer model (this may take a moment)...")
# Using a fast and effective model for semantic understanding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and effective

print("Generating semantic embeddings for training data...")
X_train_embeddings = embedding_model.encode(
    X_train.tolist(),
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)

print("Generating semantic embeddings for validation data...")
X_val_embeddings = embedding_model.encode(
    X_val.tolist(),
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)

print("Generating semantic embeddings for test data...")
X_test_embeddings = embedding_model.encode(
    test['text'].tolist(),
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)

print(f"Embedding dimension: {X_train_embeddings.shape[1]}")

# ============================================
# STAGE 3: LGBM ON SEMANTIC EMBEDDINGS
# ============================================

print("\n" + "="*80)
print("[STAGE 3] LightGBM on Semantic Embeddings")
print("="*80)

print("Training LightGBM on semantic embeddings...")
lgbm_semantic_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'max_depth': 6,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42
}

lgb_train_semantic = lgb.Dataset(X_train_embeddings, label=y_train)
lgb_val_semantic = lgb.Dataset(X_val_embeddings, label=y_val, reference=lgb_train_semantic)

lgbm_semantic = lgb.train(
    lgbm_semantic_params,
    lgb_train_semantic,
    num_boost_round=2000,
    valid_sets=[lgb_val_semantic],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(200)]
)

lgbm_semantic_val_preds = lgbm_semantic.predict(X_val_embeddings, num_iteration=lgbm_semantic.best_iteration)
lgbm_semantic_score = roc_auc_score(y_val, lgbm_semantic_val_preds)
print(f"✓ LightGBM (Semantic) ROC-AUC: {lgbm_semantic_score:.6f}")

# ============================================
# STAGE 4: HYBRID ENSEMBLE - BLEND ALL MODELS
# ============================================

print("\n" + "="*80)
print("[STAGE 4] Hybrid Ensemble - Blending All Models")
print("="*80)

# Combine classical + semantic embeddings for a super model
print("\nCreating hybrid features (Classical + Semantic)...")
X_train_hybrid = np.hstack([X_train_classical.toarray(), X_train_embeddings, X_train_meta_scaled])
X_val_hybrid = np.hstack([X_val_classical.toarray(), X_val_embeddings, X_val_meta_scaled])

print(f"Hybrid feature dimension: {X_train_hybrid.shape[1]}")

print("\nTraining LightGBM on hybrid features...")
lgbm_hybrid_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_child_samples': 20,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'verbose': -1,
    'random_state': 42
}

lgb_train_hybrid = lgb.Dataset(X_train_hybrid, label=y_train)
lgb_val_hybrid = lgb.Dataset(X_val_hybrid, label=y_val, reference=lgb_train_hybrid)

lgbm_hybrid = lgb.train(
    lgbm_hybrid_params,
    lgb_train_hybrid,
    num_boost_round=3000,
    valid_sets=[lgb_val_hybrid],
    callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(200)]
)

lgbm_hybrid_val_preds = lgbm_hybrid.predict(X_val_hybrid, num_iteration=lgbm_hybrid.best_iteration)
lgbm_hybrid_score = roc_auc_score(y_val, lgbm_hybrid_val_preds)
print(f"✓ LightGBM (Hybrid) ROC-AUC: {lgbm_hybrid_score:.6f}")

# ============================================
# STAGE 5: FINAL WEIGHTED ENSEMBLE
# ============================================

print("\n" + "="*80)
print("[STAGE 5] Final Weighted Ensemble Optimization")
print("="*80)

# Test different ensemble combinations
ensemble_configs = [
    # (LR, LGBM_Classical, LGBM_Semantic, LGBM_Hybrid, Description)
    (0.15, 0.25, 0.25, 0.35, "Hybrid Dominant"),
    (0.10, 0.20, 0.30, 0.40, "Hybrid Heavy"),
    (0.20, 0.20, 0.20, 0.40, "Balanced with Hybrid Focus"),
    (0.15, 0.15, 0.30, 0.40, "Semantic + Hybrid"),
    (0.10, 0.30, 0.20, 0.40, "Classical + Hybrid"),
    (0.05, 0.25, 0.25, 0.45, "Maximum Hybrid"),
    (0.12, 0.22, 0.28, 0.38, "Fine-tuned Balance"),
]

best_ensemble_score = 0
best_ensemble_weights = None
best_ensemble_preds = None

print("\nTesting ensemble combinations:")
print("-" * 80)

for w_lr, w_lgbm_c, w_lgbm_s, w_lgbm_h, desc in ensemble_configs:
    ensemble_preds = (
        w_lr * lr_val_preds +
        w_lgbm_c * lgbm_classical_val_preds +
        w_lgbm_s * lgbm_semantic_val_preds +
        w_lgbm_h * lgbm_hybrid_val_preds
    )
    score = roc_auc_score(y_val, ensemble_preds)
    
    print(f"{desc:30s} | LR:{w_lr:.2f} C:{w_lgbm_c:.2f} S:{w_lgbm_s:.2f} H:{w_lgbm_h:.2f} | AUC: {score:.6f}")
    
    if score > best_ensemble_score:
        best_ensemble_score = score
        best_ensemble_weights = (w_lr, w_lgbm_c, w_lgbm_s, w_lgbm_h)
        best_ensemble_preds = ensemble_preds

print("-" * 80)
print(f"\n🏆 BEST ENSEMBLE CONFIGURATION:")
print(f"   Weights: LR={best_ensemble_weights[0]:.2f}, "
      f"LGBM_Classical={best_ensemble_weights[1]:.2f}, "
      f"LGBM_Semantic={best_ensemble_weights[2]:.2f}, "
      f"LGBM_Hybrid={best_ensemble_weights[3]:.2f}")
print(f"   Validation ROC-AUC: {best_ensemble_score:.6f}")

# ============================================
# GENERATE TEST PREDICTIONS
# ============================================

print("\n" + "="*80)
print("GENERATING FINAL TEST PREDICTIONS")
print("="*80)

# Generate all predictions for test set
print("\nGenerating predictions from all models...")

lr_test_preds = lr_model.predict_proba(X_test_classical)[:, 1]
lgbm_classical_test_preds = lgbm_classical.predict(X_test_classical, num_iteration=lgbm_classical.best_iteration)
lgbm_semantic_test_preds = lgbm_semantic.predict(X_test_embeddings, num_iteration=lgbm_semantic.best_iteration)

X_test_hybrid = np.hstack([X_test_classical.toarray(), X_test_embeddings, X_test_meta_scaled])
lgbm_hybrid_test_preds = lgbm_hybrid.predict(X_test_hybrid, num_iteration=lgbm_hybrid.best_iteration)

# Final ensemble
final_test_preds = (
    best_ensemble_weights[0] * lr_test_preds +
    best_ensemble_weights[1] * lgbm_classical_test_preds +
    best_ensemble_weights[2] * lgbm_semantic_test_preds +
    best_ensemble_weights[3] * lgbm_hybrid_test_preds
)

# Create submission
submission = pd.DataFrame({
    'Id': test['Id'],
    'TARGET': final_test_preds
})
submission.to_csv('submission.csv', index=False)

print("\n✓ submission.csv saved!")
print(f"✓ Predictions range: [{final_test_preds.min():.6f}, {final_test_preds.max():.6f}]")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
print(f"Logistic Regression:           {lr_score:.6f}")
print(f"LightGBM (Classical):          {lgbm_classical_score:.6f}")
print(f"LightGBM (Semantic Embeddings): {lgbm_semantic_score:.6f}")
print(f"LightGBM (Hybrid Features):    {lgbm_hybrid_score:.6f}")
print(f"{'─' * 80}")
print(f"FINAL ENSEMBLE:                {best_ensemble_score:.6f} 🎯")
print("="*80)

if best_ensemble_score >= 0.99:
    print("\n🎉 SUCCESS! Achieved 0.99+ ROC-AUC target!")
else:
    print(f"\n📊 Current score: {best_ensemble_score:.6f}")
    print(f"   Gap to target: {0.99 - best_ensemble_score:.6f}")
    print("   Consider: More training data, hyperparameter tuning, or additional features")

print("\n" + "="*80)
print("HYBRID PROTECTOR MODEL - COMPLETE")
print("="*80)
