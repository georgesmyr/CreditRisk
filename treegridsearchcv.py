from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold
from itertools import product


class TreeCalculation():
    
    def __init__(self):
        self.tree = None
        self.calcs_names = []
        
        self._get_calcs_names()
        
    def set_tree(self, tree):
        self.tree = tree
        
    def _get_calcs_names(self):
        if not self.calcs_names:
            self.calculate()
    
    def get_leaves_count(self):
        """
        Returns the number of leaves in the tree
        The child_left[i] method returns the number of the node the is the left child of node i.
        If the node i is a leaf node, child_left[i] = -1. Therefore
        """
        if self.tree is None:
            self.calcs_names.append("leaves_count")
        else:
            t = self.tree.tree_
            n = t.node_count
            leaves = len([i for i in range(n) if t.children_left[i]== -1])
            return leaves
    
    def calculate(self):
        return [self.get_leaves_count()]
     

class TreeGridSearchCV():
    
    def __init__(self, estimator, param_grid, scoring='roc_auc', cv=3, shuffle=True,
                 refit=True, return_train_score=True, tree_calculation=None):
        self.model = estimator
        self.param_grid = param_grid
        
        self.n_splits = cv
        self.shuffle = shuffle
        
        self.refit = refit
        self.return_train_score = return_train_score
        
        self.tree_calculation = tree_calculation
        
        self.best_params_ = None
        self.best_estimator_ = None
        self.best_score_ = None
        
                
    def fit(self, features, target):
        
        params_scores_calcs = []
        
        # Find all combinations of parameters
        param_combinations = list(product(*self.param_grid.values()))
        for params in param_combinations:
            
            # Define a tree with each combination of parameters
            params_dict = dict(zip(self.param_grid.keys(), params))
            tree = DecisionTreeClassifier(random_state=42)
            tree.set_params(**params_dict)

            # Cross-validate
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=42)
            train_scores, val_scores, quantities = [], [], []
            for train_idx, test_idx in kf.split(features):
                # Fit the tree to the training data
                features_train, target_train,  = features.iloc[train_idx], target.iloc[train_idx]
                features_test, target_test = features.iloc[test_idx], target.iloc[test_idx]
                tree.fit(X=features_train, y=target_train)
                
                # Score the train dataset and testing dataset (validation)
                train_pred = tree.predict_proba(X=features_train)[:, 1]
                train_score = roc_auc_score(y_score=train_pred, y_true=target_train)
                train_scores.append(train_score)
                
                test_pred = tree.predict_proba(X=features_test)[:, 1]
                val_score = roc_auc_score(y_score=test_pred, y_true=target_test)
                val_scores.append(val_score)
                
                
#                 # Calculate the extra quantities
#                 if self.tree_calculation is not None:
#                     self.tree_calculation.set_tree(tree)
#                     self.tree_calculation.calculate()
#                     quantities.append()
            
            # Keep calculations for median validation score
            cv_scores = np.transpose(np.array([train_scores, val_scores]))
            median_val_row = cv_scores[np.where(cv_scores[:, 1] == np.median(cv_scores[:, 1]))][0]
            median_val_row = np.concatenate((list(params_dict.values()), median_val_row))
            params_scores_calcs.append(median_val_row)
            
        # Get columns
        hyperparams_names = list(self.param_grid.keys())
        cols = hyperparams_names + ['train_score', 'val_score']
        if self.tree_calculation is not None:
            cols += self.tree_calculation.calcs_names
        
        # Stack the tree features for each hyperparameter combination vertically
        params_scores_calcs = np.vstack(params_scores_calcs)
        print(params_scores_calcs[:,-1])
        # Find the best hyperparameters that lead to best validation score
        self.best_score_ = np.max(params_scores_calcs[:,-1])
        best_params_scores_calcs = params_scores_calcs[np.where(params_scores_calcs[:,-1] == self.best_score_)]
        self.best_params_ = dict(zip(hyperparams_names, best_params_scores_calcs[0,: len(hyperparams_names)]))
        # Define and train the estimator with the best hyperparameters
        self.best_estimator = None
        
        # Return a DataFrame
        params_scores_calcs = dict(zip(cols, np.transpose(params_scores_calcs)))
        return pd.DataFrame(params_scores_calcs)

            

