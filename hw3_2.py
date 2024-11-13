import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

# Generate circular data distribution
np.random.seed(0)
num_samples = 100
radius = np.random.rand(num_samples) * 2
angle = 2 * np.pi * np.random.rand(num_samples)
X = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
y = (radius > 1).astype(int)  # Label: inside circle (0) or outside (1)

# Train SVM model with RBF kernel for circular decision boundary
svm_model = SVC(kernel='rbf', C=1, gamma='auto')
svm_model.fit(X, y)

# Create a 3D plot for decision boundary
def plot_3d_svm(X, y, model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of original data
    ax.scatter(X[y == 0, 0], X[y == 0, 1], 0, color='blue', label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], 0, color='red', label="Class 1")
    
    # Generate grid for decision boundary plot
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X1, X2 = np.meshgrid(x, y)
    Z = model.decision_function(np.c_[X1.ravel(), X2.ravel()])
    Z = Z.reshape(X1.shape)
    
    # Plot decision surface
    ax.plot_surface(X1, X2, Z, color='green', alpha=0.3, edgecolor='none')
    
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Decision Boundary")
    ax.set_title("2D SVM with Circular Data Distribution")
    ax.legend()
    
    return fig

# Streamlit deployment
st.title("2D SVM Decision Boundary (Circular Data)")
st.pyplot(plot_3d_svm(X, y, svm_model))
