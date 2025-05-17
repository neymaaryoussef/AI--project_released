import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import time

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Loading and preprocessing data...")

# Load the dataset
try:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Warning: Dataset files not found. Using dummy data for demonstration.")
    # Create dummy data for demonstration
    np.random.seed(42)
    
    # Create features similar to the airline dataset
    n_samples = 10000
    
    # Generate dummy data with similar structure to the airline dataset
    train = pd.DataFrame({
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Customer Type': np.random.choice(['Loyal Customer', 'disloyal Customer'], n_samples),
        'Age': np.random.randint(10, 80, n_samples),
        'Type of Travel': np.random.choice(['Personal Travel', 'Business travel'], n_samples),
        'Class': np.random.choice(['Eco', 'Eco Plus', 'Business'], n_samples),
        'Flight Distance': np.random.randint(100, 4000, n_samples),
        'Inflight wifi service': np.random.randint(1, 6, n_samples),
        'Departure/Arrival time convenient': np.random.randint(1, 6, n_samples),
        'Ease of Online booking': np.random.randint(1, 6, n_samples),
        'Gate location': np.random.randint(1, 6, n_samples),
        'Food and drink': np.random.randint(1, 6, n_samples),
        'Online boarding': np.random.randint(1, 6, n_samples),
        'Seat comfort': np.random.randint(1, 6, n_samples),
        'Inflight entertainment': np.random.randint(1, 6, n_samples),
        'On-board service': np.random.randint(1, 6, n_samples),
        'Leg room service': np.random.randint(1, 6, n_samples),
        'Baggage handling': np.random.randint(1, 6, n_samples),
        'Checkin service': np.random.randint(1, 6, n_samples),
        'Inflight service': np.random.randint(1, 6, n_samples),
        'Cleanliness': np.random.randint(1, 6, n_samples),
        'Departure Delay in Minutes': np.random.randint(0, 90, n_samples),
        'Arrival Delay in Minutes': np.random.randint(0, 90, n_samples),
        'satisfaction': np.random.choice(['satisfied', 'neutral or dissatisfied'], n_samples)
    })
    
    # Create a smaller test set
    test = train.sample(n=int(n_samples * 0.2))
    train = train.drop(test.index)
    
    print("Created dummy data for demonstration")

# Preprocessing steps
print("Preprocessing data...")

# Handle missing values if any
train['Arrival Delay in Minutes'].fillna(train['Arrival Delay in Minutes'].median(), inplace=True)
test['Arrival Delay in Minutes'].fillna(test['Arrival Delay in Minutes'].median(), inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()

# Encode categorical columns
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_cols:
    train[col] = label_encoder.fit_transform(train[col])
    test[col] = label_encoder.fit_transform(test[col])

# Drop unnecessary columns if they exist
if 'Unnamed: 0' in train.columns:
    train.drop(['Unnamed: 0'], axis=1, inplace=True)
if 'id' in train.columns:
    train.drop(['id'], axis=1, inplace=True)
    
if 'Unnamed: 0' in test.columns:
    test.drop(['Unnamed: 0'], axis=1, inplace=True)
if 'id' in test.columns:
    test.drop(['id'], axis=1, inplace=True)

# Split data
X_train = train.drop(columns=['satisfaction'])
y_train = train['satisfaction']

X_test = test.drop(columns=['satisfaction'])
y_test = test['satisfaction']

print("Data preprocessing complete!")

# Save feature names for reference in the app
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.pkl')
print(f"Saved {len(feature_names)} feature names")

# Train and save models
print("Training models...")

# 1. Logistic Regression
print("Training Logistic Regression...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
log_reg_accuracy = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.4f}")

# 2. Decision Tree
print("Training Decision Tree...")
dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

# 3. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# 4. XGBoost with RandomizedSearchCV
print("\nTraining XGBoost with RandomizedSearchCV...")
start_time = time.time()

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0.1, 0.5, 1, 5]
}

# Create XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,  # Number of parameter settings sampled
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit RandomizedSearchCV
random_search.fit(X_train_scaled, y_train)

# Get the best model
xgb_best = random_search.best_estimator_

# Evaluate on test set
y_pred_xgb = xgb_best.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

# Print results
print(f"\nXGBoost with RandomizedSearchCV completed in {time.time() - start_time:.2f} seconds")
print(f"Best parameters: {random_search.best_params_}")
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

# Save models
print("Saving models...")
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(log_reg, 'models/logistic_regression.pkl')
joblib.dump(dt, 'models/decision_tree.pkl')
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(xgb_best, 'models/xgboost.pkl')  # Save the best XGBoost model

# Save label encoder for categorical variables
joblib.dump(label_encoder, 'models/label_encoder.pkl')

# Save model accuracies for display in the app
model_accuracies = {
    'logistic_regression': log_reg_accuracy,
    'decision_tree': dt_accuracy,
    'random_forest': rf_accuracy,
    'xgboost': xgb_accuracy
}
joblib.dump(model_accuracies, 'models/model_accuracies.pkl')

print("Model extraction and saving complete!")
print("You can now run the Streamlit app with: streamlit run app.py")