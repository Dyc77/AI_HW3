import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Generate simple 1D data
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)  # Single feature
y = (X > 2.5).astype(int).ravel()  # Binary target based on a threshold

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Accuracy
acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"Logistic Regression Accuracy: {acc_log_reg}")
print(f"SVM Accuracy: {acc_svm}")

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="black", zorder=20)
X_plot = np.linspace(0, 5, 300).reshape(-1, 1)

# Plot logistic regression
plt.plot(X_plot, log_reg.predict_proba(X_plot)[:, 1], color="blue", linewidth=2, label="Logistic Regression")
# Plot SVM decision boundary
plt.plot(X_plot, svm_model.decision_function(X_plot), color="red", linestyle='--', linewidth=2, label="SVM")

plt.xlabel("Feature")
plt.ylabel("Decision Boundary / Probability")
plt.legend()
plt.title("1D Comparison: Logistic Regression vs SVM")
plt.show()
