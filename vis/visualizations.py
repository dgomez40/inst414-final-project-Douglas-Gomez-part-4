import matplotlib.pyplot as plt
import numpy as np
def create_linreg_Scatter(y_test, y_pred):
    """creates linear regression scatter plot

    Args:
        y_test (series): True target values
        y_pred (ndarray): Predicted target values
    """
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7)

    # Fit regression line
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)
    m, b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, m*y_test + b, color="red", linewidth=2)

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted with Regression Line")
    plt.savefig("vis/regressionline.png")
    plt.close()
    
def create_line(y_test,y_pred):
    """creates line plot to show how well the model did vs actual values

    Args:
        y_test (series): True target values
        y_pred (ndarray): Predicted target values
    """
    plt.figure(figsize=(8,6))
    plt.plot(y_test.values, label="Actual AQI")
    plt.plot(y_pred, label="Predicted AQI")
    plt.legend()
    plt.title("Actual vs Predicted AQI Over Time")
    plt.savefig("vis/lineplot.png")
    plt.close()

