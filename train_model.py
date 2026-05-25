import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("housing.csv")

# List of categorical columns
cat_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']

# Encode categorical columns and save encoders
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    with open(f"{col}_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

# Features and target
X = df.drop(columns=['price'])
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and encoders saved successfully!")
