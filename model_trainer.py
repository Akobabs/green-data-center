import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_curve, auc
import joblib
import plotly.express as px
import plotly.graph_objects as go

class ModelTrainer:
    def __init__(self, data_path, model_path_reg, model_path_clf):
        self.data_path = data_path
        self.model_path_reg = model_path_reg
        self.model_path_clf = model_path_clf
        self.features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                         'Overall_Height', 'Glazing_Area']
    
    def load_data(self):
        """Load preprocessed data."""
        df = pd.read_csv(self.data_path)
        X = df[self.features]
        y_reg = df['Cooling_Load']
        y_clf = (df['PUE'] > df['PUE'].median()).astype(int)
        return X, y_reg, y_clf
    
    def train_regression(self, X, y):
        """Train Random Forest Regressor."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Regression Metrics: MSE: {mse:.2f}, R²: {r2:.2f}')
        
        # Plot actual vs predicted
        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Cooling Load', 'y': 'Predicted Cooling Load'}, 
                         title='Regression Performance')
        fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
        fig.write('../figures/regression_performance.jpg')
        
        return model, mse, r2
    
    def train_classification(self, X, y):
        """Train Random Forest Classifier."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        print(f'Classification Metrics: Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {roc_auc:.2f}')
        
        # Plot ROC curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC: {roc_auc:.2f})'))
        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        fig.write('../figures/roc_curve.png')
        
        return model, precision, recall, f1, roc_auc
    
    def train(self):
        """Execute training pipeline."""
        X, y_reg, y_clf = self.load_data()
        reg_model, mse, r2 = self.train_regression(X, y_reg)
        clf_model, precision, recall, f1, roc_auc = self.train_classification(X, y_clf)
        joblib.dump(reg_model, self.model_path_reg)
        joblib.dump(clf_model, self.model_path_clf)
        print(f'Models saved to {self.model_path_reg} and {self.model_path_clf}')
        return {'MSE': mse, 'R2': r2, 'Precision': precision, 'Recall': recall, 'F1': f1, 'AUC': roc_auc}

if __name__ == '__main__':
    trainer = ModelTrainer('../data/processed/preprocessed_energy_data.csv', 
                          'random_forest_regressor.pkl', 
                          'random_forest_classifier.pkl')
    metrics = trainer.train()