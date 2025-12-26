import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import GridSearchCV
import joblib

df = pd.read_csv('health_lifestyle_encoded.csv')

X = df.drop(columns=['disease_risk'])
y = df['disease_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = [
    'age','bmi','daily_steps','sleep_hours','water_intake_l',
    'calories_consumed','resting_hr',
    'systolic_bp','diastolic_bp','cholesterol'
]
#Chuan hoa du lieu
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

#Khoi tao model
log_reg = LogisticRegression(
    C=1,
    solver='lbfgs',
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

##Danh gia mo hinh
#Classification report
print(classification_report(y_test, y_pred))
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#Cac chi so
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

print("CV 10folds:\n")
#10 fold cross validation
pipeline = Pipeline([ ('scaler', StandardScaler()),
                      ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
                      ])

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(
    pipeline,
    X,
    y,
    cv=10,
    scoring=scoring
)

for metric in scoring.keys():
    mean_score = np.mean(cv_results[f'test_{metric}'])
    print(f"{metric.upper():10}: {mean_score:.4f}")

#Find Hyperparameter
param_grid = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs']
}
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring='f1',
    n_jobs=-1
)

grid.fit(X, y)
print("Best parameters:", grid.best_params_)
print("Best F1-score :", grid.best_score_)

#Lưu model
best_model = grid.best_estimator_
joblib.dump(best_model, 'health_risk_pipeline.pkl')
print("Đã lưu model và pipeline thành công vào file 'health_risk_pipeline.pkl'")
# Lưu danh sách các cột để đảm bảo Demo nhập đúng thứ tự
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')