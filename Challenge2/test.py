"""
Hybrid Protector Model - Optimized for 0.99+ ROC-AUC
4-Stage Architecture: Classical + Semantic + Hybrid Ensemble
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HYBRID PROTECTOR MODEL - 4-STAGE ARCHITECTURE")
print("="*70)

def extract_features(text_series):
    """Enhanced feature extraction for jailbreak detection"""
    df = pd.DataFrame()
    
    # Basic statistics
    df['char_length'] = text_series.str.len()
    df['num_words'] = text_series.str.split().apply(len)
    df['avg_word_length'] = text_series.apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)
    
    # Capitalization
    df['num_upper'] = text_series.str.findall(r'[A-Z]').apply(len)
    df['upper_ratio'] = df['num_upper'] / (df['char_length'] + 1)
    
    # Special characters
    df['num_special'] = text_series.str.findall(r'[!@#$%^&*]').apply(len)
    df['num_exclamation'] = text_series.str.count(r'!')
    df['num_question'] = text_series.str.count(r'\?')
    
    # Jailbreak keywords (expanded)
    jailbreak_terms = r'ignore|bypass|system|filter|unfiltered|jailbreak|prompt|override|admin|sudo|disable|disregard|forget|pretend|roleplay|hypothetical'
    df['num_jailbreak_terms'] = text_series.str.count(jailbreak_terms, flags=2)  # case-insensitive
    
    # Command patterns
    df['has_instruction'] = text_series.str.contains(r'\b(do|execute|run|tell me|show me)\b', regex=True, case=False).astype(int)
    df['has_negation'] = text_series.str.contains(r'\b(not|no|don\'t|never)\b', regex=True, case=False).astype(int)
    
    # Sentence structure
    df['num_sentences'] = text_series.str.count(r'[.!?]+') + 1
    
    return df
# 1. Load data
train = pd.read_csv('Dataset/train.csv')
test = pd.read_csv('Dataset/test.csv')

# Encode labels: 'benign' -> 0, 'jailbreak' -> 1
label_mapping = {'benign': 0, 'jailbreak': 1}
train['label'] = train['label'].map(label_mapping)

# 2. Split data for validation (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    train['text'], train['label'], test_size=0.2, random_state=42, stratify=train['label']
)

print(f"\nTraining samples: {len(X_train)}, Validation: {len(X_val)}")

# ============================================
# STAGE 1: CLASSICAL FEATURES (TF-IDF + Meta)
# ============================================

print("\n[STAGE 1] Classical Pipeline...")

# Create feature set
X_train_meta = extract_features(X_train)
X_val_meta = extract_features(X_val)
X_test_meta = extract_features(test['text'])

# Scale meta features
scaler = StandardScaler()
X_train_meta_scaled = scaler.fit_transform(X_train_meta)
X_val_meta_scaled = scaler.transform(X_val_meta)
X_test_meta_scaled = scaler.transform(X_test_meta)

# 3. Convert text to TF-IDF features (optimized)
vectorizer = TfidfVectorizer(
   max_features=20000, 
   ngram_range=(1, 4),
   stop_words='english',
   min_df=2,
   max_df=0.95,
   sublinear_tf=True)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(test['text'])

# Combine TF-IDF features with meta features
X_train_combined = hstack([X_train_tfidf, X_train_meta_scaled])
X_val_combined = hstack([X_val_tfidf, X_val_meta_scaled])
X_test_combined = hstack([X_test_tfidf, X_test_meta_scaled])

# Train Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(C=10, solver='liblinear', max_iter=1000, random_state=42)
lr_model.fit(X_train_combined, y_train)
lr_val_preds = lr_model.predict_proba(X_val_combined)[:, 1]
lr_score = roc_auc_score(y_val, lr_val_preds)
print(f"✓ LR ROC-AUC: {lr_score:.6f}")

# Train LightGBM on classical features
print("Training LightGBM (Classical)...")
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
    'verbose': -1,
    'random_state': 42,
    'force_col_wise': True
}

lgb_train = lgb.Dataset(X_train_combined, label=y_train)
lgb_val = lgb.Dataset(X_val_combined, label=y_val, reference=lgb_train)

lgbm_classical = lgb.train(
    lgbm_classical_params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(200)]
)

lgbm_classical_val_preds = lgbm_classical.predict(X_val_combined, num_iteration=lgbm_classical.best_iteration)
lgbm_classical_score = roc_auc_score(y_val, lgbm_classical_val_preds)
print(f"✓ LGBM (Classical) ROC-AUC: {lgbm_classical_score:.6f}")

# ============================================
# STAGE 2: SEMANTIC EMBEDDINGS
# ============================================

print("\n[STAGE 2] Semantic Embeddings...")
print("Loading sentence transformer...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding texts...")
X_train_embeddings = embedding_model.encode(X_train.tolist(), show_progress_bar=True, batch_size=32)
X_val_embeddings = embedding_model.encode(X_val.tolist(), show_progress_bar=True, batch_size=32)
X_test_embeddings = embedding_model.encode(test['text'].tolist(), show_progress_bar=True, batch_size=32)

# ============================================
# STAGE 3: LGBM ON SEMANTIC EMBEDDINGS
# ============================================

print("\n[STAGE 3] LGBM on Semantic Embeddings...")
lgbm_semantic_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

lgb_train_sem = lgb.Dataset(X_train_embeddings, label=y_train)
lgb_val_sem = lgb.Dataset(X_val_embeddings, label=y_val, reference=lgb_train_sem)

lgbm_semantic = lgb.train(
    lgbm_semantic_params,
    lgb_train_sem,
    num_boost_round=2000,
    valid_sets=[lgb_val_sem],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(200)]
)

lgbm_semantic_val_preds = lgbm_semantic.predict(X_val_embeddings, num_iteration=lgbm_semantic.best_iteration)
lgbm_semantic_score = roc_auc_score(y_val, lgbm_semantic_val_preds)
print(f"✓ LGBM (Semantic) ROC-AUC: {lgbm_semantic_score:.6f}")

# ============================================
# STAGE 4: HYBRID ENSEMBLE
# ============================================

print("\n[STAGE 4] Hybrid Ensemble...")

# Test ensemble weights
weights = [
    (0.15, 0.35, 0.50, "Semantic Heavy"),
    (0.20, 0.30, 0.50, "Balanced Semantic"),
    (0.15, 0.40, 0.45, "Classical + Semantic"),
    (0.10, 0.35, 0.55, "Maximum Semantic"),
]

best_score = 0
best_weight = None

print("\nTesting ensemble combinations:")
for w_lr, w_cl, w_sem, desc in weights:
    val_preds = w_lr * lr_val_preds + w_cl * lgbm_classical_val_preds + w_sem * lgbm_semantic_val_preds
    score = roc_auc_score(y_val, val_preds)
    print(f"{desc:25s} | LR:{w_lr:.2f} CL:{w_cl:.2f} SEM:{w_sem:.2f} | AUC: {score:.6f}")
    
    if score > best_score:
        best_score = score
        best_weight = (w_lr, w_cl, w_sem)

print(f"\n🏆 Best Ensemble: LR={best_weight[0]:.2f}, Classical={best_weight[1]:.2f}, Semantic={best_weight[2]:.2f}")
print(f"   Validation ROC-AUC: {best_score:.6f}")


# 6. Predict on test data
print("\n=== Generating Test Predictions ===")

lr_test_preds = lr_model.predict_proba(X_test_combined)[:, 1]
lgbm_classical_test_preds = lgbm_classical.predict(X_test_combined, num_iteration=lgbm_classical.best_iteration)
lgbm_semantic_test_preds = lgbm_semantic.predict(X_test_embeddings, num_iteration=lgbm_semantic.best_iteration)

# Final ensemble
test_preds = (
    best_weight[0] * lr_test_preds +
    best_weight[1] * lgbm_classical_test_preds +
    best_weight[2] * lgbm_semantic_test_preds
)

# 7. Create submission file
submission = pd.DataFrame({
    'Id': test['Id'],
    'TARGET': test_preds
})
submission.to_csv('submission.csv', index=False)

print("\n✓ submission.csv saved!")
print(f"✓ Predictions range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
print(f"\n{'='*70}")
print(f"FINAL VALIDATION ROC-AUC: {best_score:.6f}")
print(f"{'='*70}")


