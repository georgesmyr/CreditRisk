import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader():
    
    def __init__(self, path, feature_names, target_names):
        self.loan_data = None
        self.features = None
        self.loan_status = None
        self.features_train, self.features_test = None, None
        self.loan_status_train, self.loan_status_test = None, None
        
        self.load_data(path)
        self.set_features(feature_names)
        self.set_target(target_names)
        
    def load_data(self, path):
        """Loads the loan data from a CSV file."""
        self.loan_data = pd.read_csv(path)

    def set_features(self, feature_names):
        """Preprocesses the features (feature_names)."""
        features_num = self.loan_data[feature_names].select_dtypes(exclude='object')
        features_obj = self.loan_data[feature_names].select_dtypes(include='object')
        
        if not features_obj.empty:
            # One-hot encoding
            features_obj = pd.get_dummies(features_obj)
        
        dfs_x = [df_x for df_x in [features_num, features_obj] if not df_x.empty]
        self.features = pd.concat(dfs_x, axis=1)

    def set_target(self, target_names):
        """Preprocesses the target (target_names)."""
        target_num = self.loan_data[target_names].select_dtypes(exclude='object')
        target_obj = self.loan_data[target_names].select_dtypes(include='object')
        
        if not target_obj.empty:
            # One-hot encoding
            target_obj = pd.get_dummies(target_obj)
        
        dfs_y = [df_y for df_y in [target_num, target_obj] if not df_y.empty]
        self.loan_status = pd.concat(dfs_y, axis=1)

    def train_test_split(self, test_size=0.4, random_state=126):
        """Splits the loan dataset into train and test datasets."""
        self.features_train, self.features_test, self.loan_status_train, self.loan_status_test = (
            train_test_split(
                self.features, self.loan_status,
                test_size=test_size,
                random_state=random_state))
        
        
