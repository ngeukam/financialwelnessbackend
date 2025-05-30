import xgboost as xgb
import numpy as np
import joblib
from pathlib import Path

class RiskPredictor:
    def __init__(self):
        self.model = self._create_model()
        
    def _create_model(self):
        # Create a model with hardcoded "knowledge" instead of loading from file
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # Mock "training" with some reasonable weights
        # These weights represent our risk assessment logic
        model._Booster = xgb.Booster()
        model._Booster.feature_names = [
            'gdp_growth', 'inflation', 'interest_rate', 'market_volatility'
        ]
        
        # Set mock weights that would produce reasonable predictions
        # In a real app, these would come from actual training
        mock_weights = {
            'gdp_growth': -0.5,  # Higher GDP growth reduces risk
            'inflation': 0.8,     # Higher inflation increases risk
            'interest_rate': 0.6, # Higher interest rates increase risk
            'market_volatility': 1.2  # Higher volatility increases risk
        }
        
        # This is a simplified approach - real models would be more complex
        def mock_predict(features):
            risk_score = (
                features[0] * mock_weights['gdp_growth'] +
                features[1] * mock_weights['inflation'] +
                features[2] * mock_weights['interest_rate'] +
                features[3] * mock_weights['market_volatility']
            )
            
            if risk_score < -1: return 0, [0.8, 0.15, 0.05]  # Low risk
            elif risk_score < 1: return 1, [0.2, 0.6, 0.2]    # Medium risk
            else: return 2, [0.05, 0.25, 0.7]                 # High risk
        
        # Override predict methods to use our mock logic
        model.predict = lambda X: np.array([mock_predict(x)[0] for x in X])
        model.predict_proba = lambda X: np.array([mock_predict(x)[1] for x in X])
        
        return model
    
    def predict_risk(self, input_data):
        # Prepare input
        input_values = [
            input_data['gdp_growth'],
            input_data['inflation'],
            input_data['interest_rate'],
            input_data['market_volatility']
        ]
        
        # Make prediction
        prediction = self.model.predict([input_values])[0]
        probabilities = self.model.predict_proba([input_values])[0]
        confidence = probabilities.max()
        
        return {
            'risk_level': int(prediction),
            'confidence': float(confidence),
            'probabilities': probabilities.tolist(),
            'model_type': 'XGBoost (Mock)'
        }