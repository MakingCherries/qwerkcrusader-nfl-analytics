"""
NFL Analytics Data Accuracy & Validation Framework
Ensures analytics are accurate, reliable, and actionable for professional betting
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging for data quality monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAccuracyFramework:
    """
    Comprehensive framework for ensuring NFL analytics accuracy and actionability
    """
    
    def __init__(self):
        self.data_sources = {
            'primary': 'ESPN API',
            'secondary': 'Pro Football Reference',
            'odds': 'The Odds API',
            'weather': 'OpenWeatherMap API'
        }
        
        self.quality_thresholds = {
            'data_freshness_hours': 2,  # Data must be within 2 hours
            'minimum_sample_size': 5,   # Minimum games for predictions
            'confidence_threshold': 0.7, # Minimum model confidence
            'odds_variance_limit': 0.15  # Max variance between sportsbooks
        }
        
        self.validation_results = {}
    
    def validate_data_accuracy(self, data: pd.DataFrame, data_type: str) -> Dict:
        """
        Comprehensive data validation for accuracy and completeness
        
        Args:
            data: DataFrame to validate
            data_type: Type of data ('games', 'odds', 'stats', etc.)
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_report = {
            'data_type': data_type,
            'timestamp': datetime.now(),
            'total_records': len(data),
            'issues': [],
            'quality_score': 0,
            'actionable': False,
            'recommendations': []
        }
        
        # 1. Data Completeness Check
        completeness_score = self._check_completeness(data, validation_report)
        
        # 2. Data Freshness Check
        freshness_score = self._check_freshness(data, validation_report)
        
        # 3. Data Consistency Check
        consistency_score = self._check_consistency(data, validation_report)
        
        # 4. Statistical Validity Check
        statistical_score = self._check_statistical_validity(data, validation_report)
        
        # 5. Cross-Reference Validation
        cross_ref_score = self._cross_reference_validation(data, validation_report)
        
        # Calculate overall quality score
        validation_report['quality_score'] = np.mean([
            completeness_score, freshness_score, consistency_score,
            statistical_score, cross_ref_score
        ])
        
        # Determine if data is actionable
        validation_report['actionable'] = validation_report['quality_score'] >= 0.8
        
        # Generate recommendations
        self._generate_recommendations(validation_report)
        
        return validation_report
    
    def _check_completeness(self, data: pd.DataFrame, report: Dict) -> float:
        """Check for missing data and completeness"""
        if data.empty:
            report['issues'].append("Dataset is empty")
            return 0.0
        
        # Check for missing values
        missing_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        if missing_percentage > 0.1:  # More than 10% missing
            report['issues'].append(f"High missing data: {missing_percentage:.1%}")
            return max(0, 1 - missing_percentage)
        
        return 1.0
    
    def _check_freshness(self, data: pd.DataFrame, report: Dict) -> float:
        """Check if data is recent enough for actionable insights"""
        if 'timestamp' not in data.columns and 'date' not in data.columns:
            report['issues'].append("No timestamp column found for freshness check")
            return 0.5  # Neutral score if we can't check
        
        # Find the most recent timestamp
        time_col = 'timestamp' if 'timestamp' in data.columns else 'date'
        try:
            latest_time = pd.to_datetime(data[time_col]).max()
            hours_old = (datetime.now() - latest_time).total_seconds() / 3600
            
            if hours_old > self.quality_thresholds['data_freshness_hours']:
                report['issues'].append(f"Data is {hours_old:.1f} hours old")
                return max(0, 1 - (hours_old / 24))  # Decay over 24 hours
            
            return 1.0
        except:
            report['issues'].append("Could not parse timestamp data")
            return 0.5
    
    def _check_consistency(self, data: pd.DataFrame, report: Dict) -> float:
        """Check for data consistency and logical errors"""
        consistency_issues = 0
        total_checks = 0
        
        # Check for negative values where they shouldn't exist
        if 'score' in data.columns:
            total_checks += 1
            if (data['score'] < 0).any():
                consistency_issues += 1
                report['issues'].append("Found negative scores")
        
        # Check for impossible odds values
        odds_columns = [col for col in data.columns if 'odds' in col.lower() or 'ml' in col.lower()]
        for col in odds_columns:
            total_checks += 1
            if data[col].abs().max() > 10000:  # Impossible odds
                consistency_issues += 1
                report['issues'].append(f"Impossible odds values in {col}")
        
        # Check for future dates in historical data
        if 'date' in data.columns:
            total_checks += 1
            future_dates = pd.to_datetime(data['date']) > datetime.now()
            if future_dates.any():
                consistency_issues += 1
                report['issues'].append("Found future dates in historical data")
        
        if total_checks == 0:
            return 1.0
        
        return 1 - (consistency_issues / total_checks)
    
    def _check_statistical_validity(self, data: pd.DataFrame, report: Dict) -> float:
        """Check statistical validity of the data"""
        if len(data) < self.quality_thresholds['minimum_sample_size']:
            report['issues'].append(f"Sample size too small: {len(data)} records")
            return 0.3
        
        # Check for outliers in numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        outlier_percentage = 0
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | 
                       (data[col] > (Q3 + 1.5 * IQR))).sum()
            
            outlier_percentage += outliers / len(data)
        
        if len(numerical_cols) > 0:
            outlier_percentage /= len(numerical_cols)
            
            if outlier_percentage > 0.05:  # More than 5% outliers
                report['issues'].append(f"High outlier percentage: {outlier_percentage:.1%}")
                return max(0.5, 1 - outlier_percentage)
        
        return 1.0
    
    def _cross_reference_validation(self, data: pd.DataFrame, report: Dict) -> float:
        """Cross-reference data with multiple sources for validation"""
        # This would implement actual cross-referencing with multiple APIs
        # For now, return a baseline score
        return 0.8
    
    def _generate_recommendations(self, report: Dict):
        """Generate actionable recommendations based on validation results"""
        if report['quality_score'] < 0.5:
            report['recommendations'].append("❌ Data quality too poor for betting decisions")
            report['recommendations'].append("🔄 Refresh data sources and try again")
        
        elif report['quality_score'] < 0.8:
            report['recommendations'].append("⚠️ Use caution - data quality is moderate")
            report['recommendations'].append("📊 Consider additional data sources for confirmation")
        
        else:
            report['recommendations'].append("✅ Data quality is excellent for analysis")
            report['recommendations'].append("🎯 Proceed with confidence in betting decisions")
        
        # Specific recommendations based on issues
        for issue in report['issues']:
            if "missing data" in issue.lower():
                report['recommendations'].append("🔍 Investigate missing data sources")
            elif "old" in issue.lower():
                report['recommendations'].append("⏰ Set up automated data refresh")
            elif "outlier" in issue.lower():
                report['recommendations'].append("📈 Review outliers for data entry errors")

class ProfessionalValidationSuite:
    """
    Professional-grade validation suite for NFL betting analytics
    """
    
    def __init__(self):
        self.accuracy_framework = DataAccuracyFramework()
        self.model_validators = {}
        
    def validate_prediction_accuracy(self, predictions: Dict, actual_results: Dict) -> Dict:
        """
        Validate prediction accuracy against actual results
        
        Args:
            predictions: Dictionary of game predictions
            actual_results: Dictionary of actual game results
            
        Returns:
            Accuracy metrics and performance analysis
        """
        accuracy_report = {
            'total_predictions': len(predictions),
            'correct_predictions': 0,
            'accuracy_percentage': 0,
            'spread_accuracy': 0,
            'total_accuracy': 0,
            'moneyline_accuracy': 0,
            'average_error': 0,
            'confidence_calibration': {},
            'recommendations': []
        }
        
        if not predictions or not actual_results:
            accuracy_report['recommendations'].append("❌ Insufficient data for accuracy validation")
            return accuracy_report
        
        # Calculate accuracy metrics
        correct_spreads = 0
        correct_totals = 0
        correct_moneylines = 0
        total_error = 0
        
        for game_id in predictions:
            if game_id in actual_results:
                pred = predictions[game_id]
                actual = actual_results[game_id]
                
                # Spread accuracy
                if abs(pred.get('spread', 0) - actual.get('final_margin', 0)) <= 3:
                    correct_spreads += 1
                
                # Total accuracy
                predicted_total = pred.get('total', 0)
                actual_total = actual.get('total_points', 0)
                if abs(predicted_total - actual_total) <= 3:
                    correct_totals += 1
                
                # Moneyline accuracy
                pred_winner = pred.get('predicted_winner', '')
                actual_winner = actual.get('winner', '')
                if pred_winner == actual_winner:
                    correct_moneylines += 1
                
                # Calculate error
                total_error += abs(predicted_total - actual_total)
        
        total_games = len([g for g in predictions if g in actual_results])
        
        if total_games > 0:
            accuracy_report['spread_accuracy'] = correct_spreads / total_games
            accuracy_report['total_accuracy'] = correct_totals / total_games
            accuracy_report['moneyline_accuracy'] = correct_moneylines / total_games
            accuracy_report['average_error'] = total_error / total_games
            
            overall_accuracy = (correct_spreads + correct_totals + correct_moneylines) / (total_games * 3)
            accuracy_report['accuracy_percentage'] = overall_accuracy
        
        # Generate recommendations
        self._generate_accuracy_recommendations(accuracy_report)
        
        return accuracy_report
    
    def _generate_accuracy_recommendations(self, report: Dict):
        """Generate recommendations based on accuracy analysis"""
        accuracy = report['accuracy_percentage']
        
        if accuracy >= 0.6:
            report['recommendations'].append("🎯 Excellent model performance - continue current strategy")
        elif accuracy >= 0.53:
            report['recommendations'].append("📈 Good performance - minor adjustments may improve results")
        else:
            report['recommendations'].append("⚠️ Model needs significant improvement")
            report['recommendations'].append("🔄 Consider retraining with more data or different features")
        
        # Specific recommendations
        if report['spread_accuracy'] < 0.5:
            report['recommendations'].append("📊 Focus on improving spread predictions")
        
        if report['total_accuracy'] < 0.5:
            report['recommendations'].append("🎯 Enhance total points prediction models")
        
        if report['average_error'] > 5:
            report['recommendations'].append("🎲 High prediction error - review model parameters")

class ActionableInsightsGenerator:
    """
    Generate actionable insights from validated NFL analytics data
    """
    
    def __init__(self):
        self.insight_templates = {
            'high_confidence': "🔥 HIGH CONFIDENCE: {recommendation} (Confidence: {confidence:.1%})",
            'moderate_confidence': "⚡ MODERATE: {recommendation} (Confidence: {confidence:.1%})",
            'low_confidence': "💡 CONSIDER: {recommendation} (Confidence: {confidence:.1%})",
            'avoid': "❌ AVOID: {recommendation} (Risk: High)"
        }
    
    def generate_betting_insights(self, validated_data: Dict, predictions: Dict) -> List[str]:
        """
        Generate actionable betting insights from validated data
        
        Args:
            validated_data: Data that has passed validation
            predictions: Model predictions with confidence scores
            
        Returns:
            List of actionable insights for betting decisions
        """
        insights = []
        
        if not validated_data.get('actionable', False):
            insights.append("❌ Data quality insufficient for reliable betting insights")
            return insights
        
        # Generate insights based on predictions and confidence
        for game_id, prediction in predictions.items():
            confidence = prediction.get('confidence', 0)
            
            if confidence >= 0.8:
                template = self.insight_templates['high_confidence']
            elif confidence >= 0.65:
                template = self.insight_templates['moderate_confidence']
            elif confidence >= 0.55:
                template = self.insight_templates['low_confidence']
            else:
                template = self.insight_templates['avoid']
            
            # Generate specific recommendations
            if prediction.get('spread_edge', 0) > 2:
                recommendation = f"Bet {prediction.get('recommended_side', 'TBD')} spread"
                insights.append(template.format(recommendation=recommendation, confidence=confidence))
            
            if prediction.get('total_edge', 0) > 1.5:
                total_rec = "OVER" if prediction.get('total_prediction', 0) > prediction.get('line_total', 0) else "UNDER"
                recommendation = f"Bet {total_rec} {prediction.get('line_total', 0)}"
                insights.append(template.format(recommendation=recommendation, confidence=confidence))
        
        return insights

# Usage example and testing
if __name__ == "__main__":
    # Example usage
    framework = DataAccuracyFramework()
    validator = ProfessionalValidationSuite()
    insights_gen = ActionableInsightsGenerator()
    
    # Sample data validation
    sample_data = pd.DataFrame({
        'game_id': ['game1', 'game2', 'game3'],
        'home_team': ['Chiefs', 'Bills', 'Cowboys'],
        'away_team': ['Broncos', 'Dolphins', 'Giants'],
        'spread': [-7.0, -3.5, -2.5],
        'total': [44.0, 47.5, 51.5],
        'timestamp': [datetime.now() - timedelta(hours=1)] * 3
    })
    
    validation_result = framework.validate_data_accuracy(sample_data, 'games')
    print("Data Validation Results:")
    print(f"Quality Score: {validation_result['quality_score']:.2f}")
    print(f"Actionable: {validation_result['actionable']}")
    print("Recommendations:")
    for rec in validation_result['recommendations']:
        print(f"  {rec}")
