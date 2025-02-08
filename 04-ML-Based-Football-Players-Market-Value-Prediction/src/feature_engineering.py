# transforms cleaned data into model-ready features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_feature_pipeline():
    # define numerical and categorical transformations
    return feature_pipeline

if __name__ == "__main__":
    # apply transformations to cleaned data and save
