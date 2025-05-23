import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except Exception as e:
    raise RuntimeError(f"Failed to load data.pickle: {e}")

data = np.asarray(data_dict.get('data', []))
labels = np.asarray(data_dict.get('labels', []))

# Check if data is loaded correctly
if len(data) == 0 or len(labels) == 0:
    raise ValueError("No data found in data.pickle! Ensure data was saved correctly.")

# Check data-label size match
if len(data) != len(labels):
    raise ValueError(f"Mismatch between number of data samples ({len(data)}) and labels ({len(labels)}).")

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