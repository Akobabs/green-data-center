import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

class DataPreprocessor:
    def __init__(self, input_path, output_path):
        base_dir = Path(__file__).resolve().parent.parent
        self.input_path = base_dir / input_path
        self.output_path = base_dir / output_path
        self.features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                         'Overall_Height', 'Glazing_Area']
    
    def load_data(self):
        """Load and rename dataset."""
        df = pd.read_excel(self.input_path)
        df.columns = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 
                      'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Distribution', 
                      'Heating_Load', 'Cooling_Load']
        return df
    
    def handle_missing(self, df):
        """Impute missing values with median."""
        df.fillna(df.median(), inplace=True)
        return df
    
    def remove_outliers(self, df, column='Cooling_Load'):
        """Remove outliers using IQR method."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
        return df
    
    def normalize_features(self, df):
        """Normalize specified features."""
        scaler = MinMaxScaler()
        df[self.features] = scaler.fit_transform(df[self.features])
        return df
    
    def engineer_features(self, df):
        """Create synthetic PUE feature."""
        df['PUE'] = df['Cooling_Load'] / df['Cooling_Load'].mean() + 1.0
        return df
    
    def preprocess(self):
        """Execute full preprocessing pipeline."""
        df = self.load_data()
        df = self.handle_missing(df)
        df = self.remove_outliers(df)
        df = self.normalize_features(df)
        df = self.engineer_features(df)
        df.to_csv(self.output_path, index=False)
        print(f'Preprocessed data saved to {self.output_path}')
        return df

if __name__ == '__main__':
    
    preprocessor = DataPreprocessor('data/raw/ENB2012_data.xlsx', 
                                  'data/processed/preprocessed_energy_data.csv')
    preprocessor.preprocess()