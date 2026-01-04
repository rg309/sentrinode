#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime

# 1. Create some "Normal" training data
# We tell the AI: lets-go (1) at Low load (0) should be ~20ms
# We tell the AI: okey-dokey (0) at Low load (0) should be ~20ms
data = {
    'service_num': [0, 0, 0, 0, 1, 1, 1, 1],
    'load_num': [0, 0, 0, 0, 0, 0, 0, 0],
    'actual_ms': [19.5, 20.1, 20.5, 19.8, 20.2, 19.9, 21.0, 18.5]
}
df = pd.DataFrame(data)

# 2. Train the model
X = df[['service_num', 'load_num']]
y = df['actual_ms']
model = RandomForestRegressor(n_estimators=100).fit(X, y)

# 3. Save it with the pattern the app expects
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"latency_model_{timestamp}.pkl"
joblib.dump(model, filename)

print(f"Created {filename}. Restart your dashboard now!")
