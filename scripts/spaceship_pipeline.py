"""
Spaceship Titanic - Binary Classification Pipeline
Advanced classification solution for Kaggle Spaceship Titanic competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

class SpaceshipPredictor:
    """Advanced spaceship passenger transport prediction pipeline"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load training and test datasets"""
        print("=== Loading Spaceship Titanic Data ===")
        
        # Load datasets
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"
        
        self.train_df = pd.read_csv(train_path) if train_path.exists() else self.create_sample_data('train')
        self.test_df = pd.read_csv(test_path) if test_path.exists() else self.create_sample_data('test')
        
        print(f"Training data: {self.train_df.shape}")
        print(f"Test data: {self.test_df.shape}")
        
        if 'Transported' in self.train_df.columns:
            transport_rate = self.train_df['Transported'].mean()
            print(f"Transport success rate: {transport_rate:.2%}")
        
        return self.train_df, self.test_df
    
    def create_sample_data(self, data_type='train'):
        """Create sample space travel data for demonstration"""
        print(f"Creating sample {data_type} data for demonstration...")
        
        n_samples = 8000 if data_type == 'train' else 4000
        np.random.seed(42)
        
        # Generate realistic spaceship passenger data
        data = {
            'PassengerId': [f"{np.random.randint(1000, 9999):04d}_{np.random.randint(1, 9):02d}" for _ in range(n_samples)],
            'HomePlanet': np.random.choice(['Earth', 'Europa', 'Mars'], n_samples, p=[0.5, 0.3, 0.2]),
            'CryoSleep': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
            'Cabin': [f"{np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])}/{np.random.randint(0, 2000)}/{np.random.choice(['P', 'S'])}" for _ in range(n_samples)],
            'Destination': np.random.choice(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], n_samples, p=[0.7, 0.2, 0.1]),
            'Age': np.random.randint(0, 80, n_samples),
            'VIP': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
            'RoomService': np.random.exponential(200, n_samples),
            'FoodCourt': np.random.exponential(150, n_samples),
            'ShoppingMall': np.random.exponential(100, n_samples),
            'Spa': np.random.exponential(300, n_samples),
            'VRDeck': np.random.exponential(250, n_samples),
            'Name': [f"Space Traveler {i}" for i in range(1, n_samples + 1)]
        }
        
        df = pd.DataFrame(data)
        
        if data_type == 'train':
            # Create realistic transportation success based on features
            transport_prob = (
                0.5 +  # Base probability
                (df['CryoSleep'] * 0.2) +  # CryoSleep increases success
                (df['VIP'] * 0.1) +  # VIP status helps
                ((df['Age'] > 60) * -0.15) +  # Elderly have lower success
                ((df['Age'] < 18) * -0.1) +  # Children have lower success
                (df['HomePlanet'] == 'Earth') * 0.05 +  # Earth origin slight advantage
                np.random.normal(0, 0.15, n_samples)  # Random noise
            )
            df['Transported'] = (transport_prob > 0.5).astype(bool)
        
        return df
    
    def feature_engineering(self):
        """Advanced feature engineering for space travel data"""
        print("=== Space-Age Feature Engineering ===")
        
        # Combine datasets for consistent preprocessing
        train_size = len(self.train_df)
        
        if 'Transported' in self.train_df.columns:
            df_combined = pd.concat([
                self.train_df.drop(['Transported'], axis=1),
                self.test_df
            ], ignore_index=True)
        else:
            df_combined = pd.concat([self.train_df, self.test_df], ignore_index=True)
        
        # 1. Parse PassengerId
        if 'PassengerId' in df_combined.columns:
            df_combined[['GroupId', 'PersonId']] = df_combined['PassengerId'].str.split('_', expand=True)
            df_combined['GroupId'] = df_combined['GroupId'].astype(int)
            df_combined['PersonId'] = df_combined['PersonId'].astype(int)
            df_combined['GroupSize'] = df_combined.groupby('GroupId')['GroupId'].transform('count')
            df_combined['IsAlone'] = (df_combined['GroupSize'] == 1).astype(int)
        
        # 2. Parse Cabin information
        if 'Cabin' in df_combined.columns:
            cabin_parts = df_combined['Cabin'].str.split('/', expand=True)
            df_combined['Deck'] = cabin_parts[0]
            df_combined['CabinNum'] = cabin_parts[1].astype(float)
            df_combined['Side'] = cabin_parts[2]
            
            # Cabin features
            df_combined['DeckLevel'] = df_combined['Deck'].map({
                'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8
            }).fillna(0)
            df_combined['SidePort'] = (df_combined['Side'] == 'P').astype(int)
            df_combined['SideStarboard'] = (df_combined['Side'] == 'S').astype(int)
        
        # 3. Spending features
        spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        
        # Fill missing spending with 0 (likely didn't use service)
        for col in spending_cols:
            if col in df_combined.columns:
                df_combined[col].fillna(0, inplace=True)
        
        # Create spending aggregations
        if all(col in df_combined.columns for col in spending_cols):
            df_combined['TotalSpent'] = df_combined[spending_cols].sum(axis=1)
            df_combined['AvgSpending'] = df_combined[spending_cols].mean(axis=1)
            df_combined['MaxSpending'] = df_combined[spending_cols].max(axis=1)
            df_combined['SpentNothing'] = (df_combined['TotalSpent'] == 0).astype(int)
            df_combined['HighSpender'] = (df_combined['TotalSpent'] > df_combined['TotalSpent'].quantile(0.8)).astype(int)
            
            # Spending ratios
            df_combined['LuxuryRatio'] = (df_combined['Spa'] + df_combined['VRDeck']) / (df_combined['TotalSpent'] + 1)
            df_combined['BasicRatio'] = (df_combined['RoomService'] + df_combined['FoodCourt']) / (df_combined['TotalSpent'] + 1)
        
        # 4. Age features
        if 'Age' in df_combined.columns:
            df_combined['Age'].fillna(df_combined['Age'].median(), inplace=True)
            df_combined['IsChild'] = (df_combined['Age'] < 18).astype(int)
            df_combined['IsAdult'] = ((df_combined['Age'] >= 18) & (df_combined['Age'] < 60)).astype(int)
            df_combined['IsSenior'] = (df_combined['Age'] >= 60).astype(int)
            df_combined['AgeGroup'] = pd.cut(df_combined['Age'], bins=[0, 12, 18, 30, 50, 100], 
                                           labels=['Child', 'Teen', 'Young', 'Middle', 'Senior'])
        
        # 5. Planet and destination features
        if 'HomePlanet' in df_combined.columns:
            df_combined['FromEarth'] = (df_combined['HomePlanet'] == 'Earth').astype(int)
            df_combined['FromEuropa'] = (df_combined['HomePlanet'] == 'Europa').astype(int)
            df_combined['FromMars'] = (df_combined['HomePlanet'] == 'Mars').astype(int)
        
        if 'Destination' in df_combined.columns:
            df_combined['ToTrappist'] = (df_combined['Destination'] == 'TRAPPIST-1e').astype(int)
            df_combined['ToPSO'] = (df_combined['Destination'] == 'PSO J318.5-22').astype(int)
            df_combined['ToCancri'] = (df_combined['Destination'] == '55 Cancri e').astype(int)
        
        # 6. Service combination features
        if 'CryoSleep' in df_combined.columns:
            df_combined['CryoSleep'] = df_combined['CryoSleep'].fillna(False).astype(int)
            
        if 'VIP' in df_combined.columns:
            df_combined['VIP'] = df_combined['VIP'].fillna(False).astype(int)
        
        # Handle remaining missing values
        numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_combined[col].fillna(df_combined[col].median(), inplace=True)
        
        categorical_cols = df_combined.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in ['PassengerId', 'Name', 'Cabin']:
                le = LabelEncoder()
                df_combined[col] = le.fit_transform(df_combined[col].astype(str))
        
        # Also encode any remaining categorical features
        for col in df_combined.columns:
            if df_combined[col].dtype == 'object' or df_combined[col].dtype.name == 'category':
                if col not in ['PassengerId', 'Name', 'Cabin']:
                    le = LabelEncoder()
                    df_combined[col] = le.fit_transform(df_combined[col].astype(str))
        
        # Select features for modeling
        feature_cols = [col for col in df_combined.columns if col not in [
            'PassengerId', 'Name', 'Cabin', 'Transported'
        ]]
        
        # Split back to train and test
        self.X_train = df_combined[:train_size][feature_cols]
        self.X_test = df_combined[train_size:][feature_cols]
        
        if 'Transported' in self.train_df.columns:
            self.y_train = self.train_df['Transported'].astype(int)
        else:
            # Create sample target for demonstration
            self.y_train = np.random.choice([0, 1], size=train_size, p=[0.5, 0.5])
        
        print(f"Training features: {self.X_train.shape}")
        print(f"Test features: {self.X_test.shape}")
        print(f"Feature columns: {len(feature_cols)}")
        
        return self.X_train, self.X_test, self.y_train
    
    def train_models(self):
        """Train ensemble of classification models"""
        print("=== Training Space Classification Models ===")
        
        models = {
            'logistic': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.01, random_state=42, max_iter=500)
        }
        
        # Cross-validation with stratified folds
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=cv, scoring='accuracy')
                
                # Fit model
                model.fit(self.X_train, self.y_train)
                
                # Train predictions
                train_pred = model.predict(self.X_train)
                train_accuracy = accuracy_score(self.y_train, train_pred)
                
                # Store results
                self.models[name] = model
                self.results[name] = {
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'train_accuracy': train_accuracy
                }
                
                print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                print(f"  Train Accuracy: {train_accuracy:.4f}")
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                continue
        
        # Create voting ensemble
        if len(self.models) >= 3:
            voting_models = [(name, model) for name, model in self.models.items()]
            voting_clf = VotingClassifier(estimators=voting_models, voting='soft')
            
            cv_scores = cross_val_score(voting_clf, self.X_train, self.y_train, 
                                      cv=cv, scoring='accuracy')
            voting_clf.fit(self.X_train, self.y_train)
            
            self.models['voting_ensemble'] = voting_clf
            self.results['voting_ensemble'] = {
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'train_accuracy': accuracy_score(self.y_train, voting_clf.predict(self.X_train))
            }
            
            print(f"\\nVoting Ensemble:")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Rank models by CV performance
        if self.results:
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['cv_accuracy_mean'])
            print(f"\\nBest model: {best_model_name}")
            print(f"Best CV Accuracy: {self.results[best_model_name]['cv_accuracy_mean']:.4f}")
        
        return self.models, self.results
    
    def make_predictions(self, model_name=None):
        """Generate predictions"""
        print("=== Making Spaceship Predictions ===")
        
        if not self.results:
            print("No trained models available")
            return None
        
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x]['cv_accuracy_mean'])
        
        model = self.models[model_name]
        predictions = model.predict(self.X_test)
        
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'],
            'Transported': predictions.astype(bool)
        })
        
        print(f"Model used: {model_name}")
        print(f"CV Accuracy: {self.results[model_name]['cv_accuracy_mean']:.4f}")
        print(f"Transport success rate: {predictions.mean():.2%}")
        
        return submission
    
    def save_submission(self, submission, filename='spaceship_submission.csv'):
        """Save submission file"""
        if submission is not None:
            output_path = self.data_dir.parent / 'submissions' / filename
            submission.to_csv(output_path, index=False)
            print(f"Submission saved: {output_path}")
            return output_path
        return None

def main():
    """Main execution pipeline"""
    print("=" * 50)
    print("SPACESHIP TITANIC CLASSIFICATION PIPELINE")
    print("=" * 50)
    
    # Initialize predictor
    predictor = SpaceshipPredictor()
    
    # Load data
    predictor.load_data()
    
    # Feature engineering
    predictor.feature_engineering()
    
    # Train models
    predictor.train_models()
    
    # Make predictions
    submission = predictor.make_predictions()
    
    # Save results
    predictor.save_submission(submission)
    
    print("\\n" + "=" * 50)
    print("SPACESHIP PIPELINE COMPLETE!")
    print("=" * 50)
    
    return predictor, submission

if __name__ == "__main__":
    predictor, submission = main()