import numpy as np

class Preprocessor:
    def __init__(self):
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        
    def fit(self, X_train, y_train):
        """
        Compute statistics on the training set.
        """
        # Feature Scaling
        self.mean_x = np.mean(X_train, axis=0)
        self.std_x = np.std(X_train, axis=0) + 1e-6
        
        # Target Centering
        self.mean_y = np.mean(y_train)
        
    def transform(self, X, y=None):
        """
        Apply statistics to X (and optionally y).
        """
        if self.mean_x is None:
            raise RuntimeError("Preprocessor must be fit before calling transform.")
            
        X_scaled = (X - self.mean_x) / self.std_x
        
        y_scaled = None
        if y is not None:
            y_scaled = y - self.mean_y
            
        return X_scaled, y_scaled

    def inverse_transform_y(self, y_centered):
        """
        Add the mean back to predictions.
        """
        return y_centered + self.mean_y