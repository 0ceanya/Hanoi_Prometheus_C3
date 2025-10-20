# main.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. Load data
# train = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
# test = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')

train = pd.read_csv('Dataset/train.csv')
test = pd.read_csv('Dataset/test.csv')

# 2. Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train['text'], train['label'], test_size=0.2, random_state=42
)

# 3. Convert text to TF-IDF features
vectorizer = TfidfVectorizer(
   max_features=15000, 
   ngram_range=(1, 3),     #unigrams + bigrams + trigrams
   stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# 4. Train logistic regression model
model = LogisticRegression(max_iter=300)
model.fit(X_train_tfidf, y_train)

# 5. Evaluate on validation set
val_preds = model.predict_proba(X_val_tfidf)[:, 1]
print("Validation ROC-AUC:", roc_auc_score(y_val, val_preds))

# 6. Predict on test data
X_test_tfidf = vectorizer.transform(test['text'])
test_preds = model.predict_proba(X_test_tfidf)[:, 1]

# 7. Create submission file
submission = pd.DataFrame({
    'Id': test['Id'],
    'TARGET': test_preds
})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved!")
