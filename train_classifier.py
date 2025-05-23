import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load data.pickle: {e}")

raw_data = data_dict.get('data', [])
labels = np.asarray(data_dict.get('labels', []))

# Check for empty data
if len(raw_data) == 0 or len(labels) == 0:
    raise ValueError("No data found in data.pickle! Ensure data was saved correctly.")

# Filter out inconsistent data lengths (e.g., expect 63 features for 21 hand landmarks with x, y, z)
expected_length = len(raw_data[0])
filtered_data = []
filtered_labels = []

for d, l in zip(raw_data, labels):
    if len(d) == expected_length:
        filtered_data.append(d)
        filtered_labels.append(l)

if len(filtered_data) == 0:
    raise ValueError("No valid data samples after filtering inconsistent shapes.")

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Final size check
if len(data) != len(labels):
    raise ValueError(f"Mismatch after filtering: {len(data)} samples, {len(labels)} labels.")

# Split and train
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)