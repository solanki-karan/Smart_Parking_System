import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import joblib

# ----------- Utility Functions -----------

def hhmm_to_seconds(h, m):
    return h * 3600 + m * 60

def seconds_to_hhmm(seconds):
    return datetime.strptime(str(int(seconds // 3600)).zfill(2) + ':' + str(int((seconds % 3600) // 60)).zfill(2), "%H:%M")

# ----------- Load Model -----------

model = joblib.load("parking_predictor_model.joblib")
print("✅ Loaded model: parking_predictor_model.joblib")

# ----------- Generate Predictions for Entire Day -----------

interval_minutes = 5  # change to 1 for every minute
start_time = 0           # 00:00
end_time = 24 * 3600     # 24:00

times = []
predicted_counts = []

for seconds in range(start_time, end_time, interval_minutes * 60):
    times.append(seconds_to_hhmm(seconds))
    predicted = model.predict(np.array([[seconds]]))[0]
    predicted_counts.append(predicted)

# ----------- Plot -----------

plt.figure(figsize=(12, 6))
plt.plot(times, predicted_counts, color='green', marker='', linestyle='-', label='Predicted Vehicle Count')

plt.title("Predicted Vehicle Count vs Time of Day")
plt.xlabel("Time (HH:MM)")
plt.ylabel("Predicted Number of Vehicles")
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.tight_layout()
plt.show()
