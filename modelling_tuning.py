import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Proyek_Akhir_Skilled")

# 2. Load Data
df = pd.read_csv('dataset_preprocessing.csv')
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training & Tuning dengan Manual Logging
with mlflow.start_run() as run:
    print("Memulai proses training dan tuning...")
    
    # Setup Hyperparameter
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Evaluasi model terbaik
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # --- MANUAL LOGGING KE MLFLOW ---
    # Log Parameter Terbaik
    mlflow.log_params(grid_search.best_params_)
    # Log Metrik
    mlflow.log_metric("accuracy", acc)
    # Log Artefak Model
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    print(f"Training selesai. Akurasi terbaik: {acc}")
    print(f"Run ID: {run.info.run_id}")