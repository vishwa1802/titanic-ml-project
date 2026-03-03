import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Titanic project running successfully")

# Load data
data = pd.read_csv("titanic.csv")

# 🔍 DEBUG INFO
print("Shape:", data.shape)
print("Unique Survived:", data["Survived"].unique())

# Clean column names
data.columns = data.columns.str.strip()

# Convert categorical
if "Sex" in data.columns:
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

if "Embarked" in data.columns:
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Fill missing values
for col in ["Age", "Fare"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())

if "Embarked" in data.columns:
    data["Embarked"] = data["Embarked"].fillna(0)

# Feature selection
possible_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
features = [f for f in possible_features if f in data.columns]

X = data[features]
y = data["Survived"]

# Safety numeric conversion
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# ✅ IMPROVED SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\nModel trained successfully ✅")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ---- TEST.CSV PART ----
print("\n--- Running on test.csv ---")

test_data = pd.read_csv("test.csv")
test_data.columns = test_data.columns.str.strip()

if "Sex" in test_data.columns:
    test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})

if "Embarked" in test_data.columns:
    test_data["Embarked"] = test_data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

for col in ["Age", "Fare"]:
    if col in test_data.columns:
        test_data[col] = pd.to_numeric(test_data[col], errors="coerce")
        test_data[col] = test_data[col].fillna(test_data[col].median())

if "Embarked" in test_data.columns:
    test_data["Embarked"] = test_data["Embarked"].fillna(0)

X_test_real = test_data[features]
X_test_real = X_test_real.apply(pd.to_numeric, errors="coerce").fillna(0)

test_predictions = model.predict(X_test_real)

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv("submission.csv", index=False)

print("✅ submission.csv created!")