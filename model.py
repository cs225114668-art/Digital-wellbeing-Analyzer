import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("digital_wellbeing.csv")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode target
le = LabelEncoder()
df["WellBeing_Status"] = le.fit_transform(df["WellBeing_Status"])

# Features and target
X = df.drop("WellBeing_Status", axis=1)
y = df["WellBeing_Status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results[name] = acc

    print(f"{name} : {acc:.4f}")

# Best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model :", best_model_name)
print("Accuracy :", results[best_model_name])

# Save model
pickle.dump(best_model, open("best_model.pkl", "wb"))

# Save label encoder
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("\nModel Saved Successfully")