# loads engineered data, trains multiple models, and saves the best one
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # load processed dataset
    # split into train/test
    # train model, save best model
