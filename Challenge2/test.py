# main.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
# model = LogisticRegression(max_iter=300)
# model.fit(X_train_tfidf, y_train)

#4.1 Tune Logistic Regression hyperparameters
params = {
    'C': [0.1, 1, 5],
    'solver': ['liblinear', 'saga'],
}
grid = GridSearchCV(LogisticRegression(max_iter=500), params, scoring='roc_auc', cv=3, n_jobs=-1)
grid.fit(X_train_tfidf, y_train)
print("Best params:", grid.best_params_)

# 5. Evaluate on validation set
val_preds = grid.predict_proba(X_val_tfidf)[:, 1]
print("Validation ROC-AUC:", roc_auc_score(y_val, val_preds))

# 6. Predict on test data
X_test_tfidf = vectorizer.transform(test['text'])
test_preds = grid.predict_proba(X_test_tfidf)[:, 1]

# 7. Create submission file
submission = pd.DataFrame({
    'Id': test['Id'],
    'TARGET': test_preds
})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved!")
