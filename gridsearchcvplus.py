from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from itertools import product


class ModelCalculations():
    """
    Class that performs custom calculations from a model's attributes.
    For example, calculates the number of leaf nodes of a DecisionTreeClassifier
    """
    
    def __init__(self):
        """Initialize ModelCalculations instance"""
        self.estimator = None
              
            
    def set_estimator(self, estimator):
        """Sets the estimator to do the calculations for"""
        self.estimator = estimator
            
    
    def get_leaves_count(self):
        """
        Returns the number of leaves in the tree
        The child_left[i] method returns the number of the node the is the left child of node i.
        If the node i is a leaf node, child_left[i] = -1. Therefore
        """
        t = self.estimator.tree_
        n = t.node_count
        leaves = len([i for i in range(n) if t.children_left[i]== -1])
        return {"leaves_count": leaves}
    
    
    def calculate(self):
        """ Entry point to do all the calculations"""
        
        return {**self.get_leaves_count()}


    
    
class GridSearchCVPlus():
    """
    Performs a grid search cross validation. It can perform custom calculations for a model from its
    attributes, by passing a ModelCalculation instance in model_calculations
    """
    
    def __init__(self, estimator, param_grid, scoring='roc_auc', cv=3, shuffle=True,
                 refit=True, return_train_score=True, model_calculations=None):
        """Initialisation of grid search cross validation configuration, and parameters"""
        
        # GridSeachCV configuration
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_splits = cv
        self.shuffle = shuffle
        self.refit = refit
        self.return_train_score = return_train_score
        self.random_state = self.estimator.random_state
        
        # Custom calculations
        self.model_calculations = model_calculations
        
        # Cross-validation results
        self.cv_results_ = {}
        
        # Best parameters, estimator and score
        self.best_params_ = None
        self.best_calculations_ = None
        self.best_estimator_ = None
        self.best_score_ = None
        
    def fit(self, features, target):        
        """ Fit the grid search cross validation instance to the dataset """
        
        # Initialise cross validation results dictionary
        self.cv_results_["params"] = []      
        for split in range(self.n_splits):
            self.cv_results_[f"split{split}_train_score"] = []
        for split in range(self.n_splits):
            self.cv_results_[f"split{split}_test_score"] = []
        if self.model_calculations is not None:
            self.cv_results_["model_calculations"] = []
            
        # Perform cross validation
        self._grid_search_cross_validate(features, target)
        
        # Look for best test/validation score
        test_scores = np.vstack([self.cv_results_[f"split{i}_test_score"] for i in range(self.n_splits)])
        mean_test_scores = np.mean(test_scores, axis=0)
        best_score_arg = np.argmax(mean_test_scores)
        
        # Save best validation score, parameters, and calculations
        self.best_score_ = mean_test_scores[best_score_arg]
        self.best_params_ = self.cv_results_["params"][best_score_arg]
        self.best_calculations_ = self.cv_results_["model_calculations"][best_score_arg]
        
        # Fit best estimator to the dataset
        self.best_estimator_ = self.estimator
        self.best_estimator_.set_params(**self.best_params_)
        features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                                    test_size=0.4,
                                                                                    random_state=self.random_state)
        self.best_estimator_.fit(features_train, target_train)
        
        
                
    def _grid_search_cross_validate(self, features, target):
        """ 
        Cross validation, save parameters, training and testing scores for each split,
        and custom calculations 
        """        
        # Find all combinations of parameters
        param_combinations = list(product(*self.param_grid.values()))
        for params in param_combinations:
            
            # Set the parameters of the estimator each combination of parameters, and store them
            params_dict = dict(zip(self.param_grid.keys(), params))
            self.cv_results_["params"].append(params_dict)
            self.estimator.set_params(**params_dict)

            # Cross-validate
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            train_scores, val_scores, calculations = [], [], []
            split_number = 0
            for train_idx, test_idx in kf.split(features):
                
                # Fit the tree to the training data
                features_train, target_train,  = features.iloc[train_idx], target.iloc[train_idx]
                features_test, target_test = features.iloc[test_idx], target.iloc[test_idx]
                self.estimator.fit(X=features_train, y=target_train)
                
                # Score the train dataset and testing dataset (validation)
                train_pred = self.estimator.predict_proba(X=features_train)[:, 1]
                train_score = roc_auc_score(y_score=train_pred, y_true=target_train)
                train_scores.append(train_score)
                test_pred = self.estimator.predict_proba(X=features_test)[:, 1]
                val_score = roc_auc_score(y_score=test_pred, y_true=target_test)
                val_scores.append(val_score)
                self.cv_results_[f"split{split_number}_train_score"].append(train_score)
                self.cv_results_[f"split{split_number}_test_score"].append(val_score)
                
                # Calculate the custom model quantities
                if self.model_calculations is not None:
                    self.model_calculations.set_estimator(self.estimator)   # Set the current tree for tree calculations
                    calcs = self.model_calculations.calculate()             # Get tree calculations
                    self.cv_results_["model_calculations"].append(calcs)    # Aggregate calculations
            
                split_number += 1

            

