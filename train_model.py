import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1️⃣ Load CSV
df = pd.read_csv("embedded_system_network_security_dataset.csv")

# 2️⃣ Drop columns not needed for training
df = df.drop(columns=["id", "label"], errors="ignore")

# 3️⃣ Encode categorical columns
for col in df.select_dtypes(include="object"):
    df[col] = df[col].astype("category").cat.codes

# 4️⃣ Scale features
scaler = StandardScaler()
scaled = scaler.fit_transform(df)  # fit_transform → learns mean/std and scales

# 5️⃣ Train Isolation Forest for anomaly detection
model = IsolationForest()
model.fit(scaled)

# 6️⃣ Save trained model and scaler
joblib.dump(model, "anomaly_model.pkl")
joblib.dump(scaler, "scaler.pkl")