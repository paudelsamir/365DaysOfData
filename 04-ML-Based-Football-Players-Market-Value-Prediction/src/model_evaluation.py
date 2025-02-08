# evaluates model performance using metrics and visualizations
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    # plot actual vs predicted
    return mae

if __name__ == "__main__":
    # load saved model, evaluate, visualize
