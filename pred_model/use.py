import joblib
import numpy as np

# Load saved model
model = joblib.load("parking_predictor_model.joblib")

def time_to_seconds(hhmmss):
    h = int(hhmmss[:2])
    m = int(hhmmss[2:4])
    s = int(hhmmss[4:6])
    return h * 3600 + m * 60 + s

def predict_probability(hhmmss, max_capacity=35):
    seconds = time_to_seconds(hhmmss)
    predicted_vehicles = model.predict(np.array([[seconds]]))[0]
    prob = max(0.0, min(1.0, 1 - predicted_vehicles / max_capacity))
    return prob, predicted_vehicles

# Example use
hhmmss = input("Enter time: ")
prob, count = predict_probability(hhmmss)
print(f"Expected vehicles at {hhmmss}: {count:.2f}, Probability of finding a spot: {prob:.2%}")
