import matplotlib as plt
import numpy as np
def create_linreg_Scatter(y_test, y_pred):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7)

    # Fit regression line
    m, b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, m*y_test + b, color="red", linewidth=2)

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted with Regression Line")
    plt.savefig("vis/regressionline.png")
    plt.close()