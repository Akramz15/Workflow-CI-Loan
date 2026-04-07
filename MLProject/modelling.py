import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Membaca dataset...
df = pd.read_csv('dataset_preprocessing.csv')

X = df.iloc[:, :-1] 
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- INI BARIS TAMBAHANNYA ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# -----------------------------

mlflow.set_experiment("Proyek_Akhir_Basic")
mlflow.autolog()

with mlflow.start_run(run_name="Basic_RandomForest"):
    print("Memulai training model dasar...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Training berhasil! Model dasar telah tersimpan di MLflow.")