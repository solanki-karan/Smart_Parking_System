import re
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# ----------- Time conversion function -----------

def time_to_seconds(hhmmss):
    h = int(hhmmss[:2])
    m = int(hhmmss[2:4])
    s = int(hhmmss[4:6])
    return h * 3600 + m * 60 + s

# ----------- Function to load data from multiple files -----------

def load_data_from_files(file_list):
    X, y = [], []
    for file_path in file_list:
        with open(file_path, 'r') as f:
            for line in f:
                match = re.match(r'(\d{6}):\s*(\d+)\s+vehicles', line)
                if match:
                    hhmmss = match.group(1)
                    vehicles = int(match.group(2))
                    seconds = time_to_seconds(hhmmss)
                    X.append(seconds)
                    y.append(vehicles)
    return np.array(X).reshape(-1, 1), np.array(y)

# ----------- List of all txt files -----------

txt_files = [
    "pred_model/moments/vehicle_count_log.txt",
    # Add more file paths here
]

# ----------- Load data and train model -----------

X, y = load_data_from_files(txt_files)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "parking_predictor_model.joblib")
print("✅ RandomForest model saved as 'parking_predictor_model.joblib'")

# ----------- Prediction function -----------

def predict_probability(hhmmss, max_capacity=35):
    seconds = time_to_seconds(hhmmss)
    predicted_vehicles = model.predict(np.array([[seconds]]))[0]
    prob = max(0.0, min(1.0, 1 - predicted_vehicles / max_capacity))
    return prob, predicted_vehicles

# ----------- Example usage -----------

# query_time = input()  # HHMMSS format
# prob, count = predict_probability(query_time)

# print(f"At {query_time}:")
# print(f"  Estimated vehicles: {count:.2f}")
# print(f"  Probability of finding a spot: {prob:.2%}")
