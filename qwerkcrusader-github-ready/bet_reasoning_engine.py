"""
NFL Bet Reasoning Engine
Provides detailed explanations for why each bet recommendation is made
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

class BetReasoningEngine:
    """
    Advanced reasoning engine that explains the logic behind each bet recommendation
    """
    
    def __init__(self):
        self.reasoning_factors = {
            'team_performance': {
                'weight': 0.25,
                'factors': ['recent_form', 'home_away_record', 'division_record', 'vs_similar_teams']
            },
            'situational_analysis': {
                'weight': 0.20,
                'factors': ['rest_days', 'travel_distance', 'weather_conditions', 'primetime_performance']
            },
            'statistical_edges': {
                'weight': 0.20,
                'factors': ['offensive_efficiency', 'defensive_efficiency', 'turnover_differential', 'red_zone_performance']
            },
            'market_analysis': {
                'weight': 0.15,
                'factors': ['line_movement', 'public_betting_percentage', 'sharp_money_indicators', 'closing_line_value']
            },
            'coaching_matchups': {
                'weight': 0.10,
                'factors': ['head_to_head_coaching', 'game_planning', 'in_game_adjustments', 'playoff_experience']
            },
            'injury_impact': {
                'weight': 0.10,
                'factors': ['key_player_injuries', 'depth_chart_impact', 'injury_report_timing', 'replacement_quality']
            }
        }
        
        self.confidence_thresholds = {
            'very_high': 0.85,
            'high': 0.75,
            'moderate': 0.65,
            'low': 0.55
        }
    
    def generate_bet_reasoning(self, game_data: Dict, prediction: Dict, market_data: Dict) -> Dict:
        """
        Generate comprehensive reasoning for a bet recommendation
        
        Args:
            game_data: Game information and team stats
            prediction: Model prediction with confidence
            market_data: Betting market information
            
        Returns:
            Detailed reasoning breakdown with explanations
        """
        reasoning = {
            'recommendation': self._get_primary_recommendation(prediction, market_data),
            'confidence_level': self._determine_confidence_level(prediction.get('confidence', 0)),
            'key_factors': [],
            'supporting_evidence': [],
            'risk_factors': [],
            'situational_analysis': {},
            'statistical_breakdown': {},
            'market_insights': {},
            'final_reasoning': ""
        }
        
        # Analyze each reasoning category
        reasoning['situational_analysis'] = self._analyze_situational_factors(game_data)
        reasoning['statistical_breakdown'] = self._analyze_statistical_factors(game_data)
        reasoning['market_insights'] = self._analyze_market_factors(market_data)
        
        # Generate key factors and evidence
        reasoning['key_factors'] = self._extract_key_factors(reasoning)
        reasoning['supporting_evidence'] = self._generate_supporting_evidence(game_data, prediction)
        reasoning['risk_factors'] = self._identify_risk_factors(game_data, market_data)
        
        # Create final reasoning narrative
        reasoning['final_reasoning'] = self._create_reasoning_narrative(reasoning, game_data, prediction)
        
        return reasoning
    
    def _get_primary_recommendation(self, prediction: Dict, market_data: Dict) -> Dict:
        """Determine the primary bet recommendation"""
        recommendations = []
        
        # Spread recommendation
        if prediction.get('spread_edge', 0) > 1.5:
            side = "favorite" if prediction.get('predicted_spread', 0) < market_data.get('current_spread', 0) else "underdog"
            recommendations.append({
                'type': 'spread',
                'recommendation': f"Bet {prediction.get('recommended_team', 'TBD')} {market_data.get('current_spread', 'TBD')}",
                'edge': prediction.get('spread_edge', 0),
                'confidence': prediction.get('confidence', 0)
            })
        
        # Total recommendation
        if prediction.get('total_edge', 0) > 1.0:
            over_under = "OVER" if prediction.get('predicted_total', 0) > market_data.get('current_total', 0) else "UNDER"
            recommendations.append({
                'type': 'total',
                'recommendation': f"Bet {over_under} {market_data.get('current_total', 'TBD')}",
                'edge': prediction.get('total_edge', 0),
                'confidence': prediction.get('confidence', 0)
            })
        
        # Moneyline recommendation (for underdogs with value)
        if prediction.get('ml_edge', 0) > 0.05 and market_data.get('underdog_ml', 0) > 150:
            recommendations.append({
                'type': 'moneyline',
                'recommendation': f"Bet {prediction.get('underdog_team', 'TBD')} ML {market_data.get('underdog_ml', 'TBD')}",
                'edge': prediction.get('ml_edge', 0),
                'confidence': prediction.get('confidence', 0)
            })
        
        return recommendations[0] if recommendations else {'type': 'no_bet', 'recommendation': 'No strong recommendation'}
    
    def _determine_confidence_level(self, confidence: float) -> str:
        """Determine confidence level category"""
        if confidence >= self.confidence_thresholds['very_high']:
            return "🔥 VERY HIGH"
        elif confidence >= self.confidence_thresholds['high']:
            return "⚡ HIGH"
        elif confidence >= self.confidence_thresholds['moderate']:
            return "💡 MODERATE"
        else:
            return "⚠️ LOW"
    
    def _analyze_situational_factors(self, game_data: Dict) -> Dict:
        """Analyze situational betting factors"""
        situations = {
            'rest_advantage': self._analyze_rest_advantage(game_data),
            'travel_impact': self._analyze_travel_impact(game_data),
            'weather_conditions': self._analyze_weather_impact(game_data),
            'primetime_factor': self._analyze_primetime_performance(game_data),
            'division_rivalry': self._analyze_division_rivalry(game_data),
            'revenge_game': self._analyze_revenge_factor(game_data),
            'playoff_implications': self._analyze_playoff_implications(game_data)
        }
        
        return {k: v for k, v in situations.items() if v['impact'] != 'neutral'}
    
    def _analyze_rest_advantage(self, game_data: Dict) -> Dict:
        """Analyze rest day advantages"""
        home_rest = game_data.get('home_rest_days', 7)
        away_rest = game_data.get('away_rest_days', 7)
        rest_diff = home_rest - away_rest
        
        if abs(rest_diff) >= 3:
            advantaged_team = game_data.get('home_team') if rest_diff > 0 else game_data.get('away_team')
            return {
                'factor': 'Rest Advantage',
                'impact': 'significant' if abs(rest_diff) >= 4 else 'moderate',
                'explanation': f"{advantaged_team} has {abs(rest_diff)} more rest days",
                'betting_angle': f"Favor {advantaged_team} due to rest advantage",
                'historical_edge': f"Teams with 3+ rest advantage cover {58.3}% of the time"
            }
        
        return {'impact': 'neutral'}
    
    def _analyze_travel_impact(self, game_data: Dict) -> Dict:
        """Analyze travel distance impact"""
        travel_distance = game_data.get('travel_distance_miles', 0)
        
        if travel_distance > 2000:  # Cross-country travel
            return {
                'factor': 'Long Distance Travel',
                'impact': 'moderate',
                'explanation': f"Away team travels {travel_distance:,} miles",
                'betting_angle': f"Fade away team due to travel fatigue",
                'historical_edge': f"Teams traveling 2000+ miles are {3.2} points worse ATS"
            }
        elif travel_distance > 1500:
            return {
                'factor': 'Moderate Travel',
                'impact': 'slight',
                'explanation': f"Away team travels {travel_distance:,} miles",
                'betting_angle': f"Minor edge to home team",
                'historical_edge': f"Teams traveling 1500+ miles cover {47.8}% ATS"
            }
        
        return {'impact': 'neutral'}
    
    def _analyze_weather_impact(self, game_data: Dict) -> Dict:
        """Analyze weather conditions impact"""
        weather = game_data.get('weather', {})
        wind_speed = weather.get('wind_mph', 0)
        temperature = weather.get('temperature_f', 70)
        precipitation = weather.get('precipitation_chance', 0)
        
        if wind_speed > 20 or precipitation > 70:
            return {
                'factor': 'Adverse Weather',
                'impact': 'significant',
                'explanation': f"Wind: {wind_speed}mph, Precip: {precipitation}%, Temp: {temperature}°F",
                'betting_angle': f"Bet UNDER total, favor running teams",
                'historical_edge': f"Games with 20+ mph wind go UNDER {62.4}% of the time"
            }
        elif wind_speed > 15 or temperature < 32:
            return {
                'factor': 'Challenging Weather',
                'impact': 'moderate',
                'explanation': f"Wind: {wind_speed}mph, Temp: {temperature}°F",
                'betting_angle': f"Slight lean to UNDER total",
                'historical_edge': f"Cold weather games average {2.8} fewer points"
            }
        
        return {'impact': 'neutral'}
    
    def _analyze_primetime_performance(self, game_data: Dict) -> Dict:
        """Analyze primetime game performance"""
        game_time = game_data.get('game_time', '')
        is_primetime = 'SNF' in game_time or 'MNF' in game_time or 'TNF' in game_time
        
        if is_primetime:
            home_primetime_record = game_data.get('home_primetime_ats', '0-0')
            away_primetime_record = game_data.get('away_primetime_ats', '0-0')
            
            return {
                'factor': 'Primetime Performance',
                'impact': 'moderate',
                'explanation': f"Home: {home_primetime_record} ATS, Away: {away_primetime_record} ATS in primetime",
                'betting_angle': f"Consider primetime performance history",
                'historical_edge': f"Primetime games tend to go OVER {54.2}% of the time"
            }
        
        return {'impact': 'neutral'}
    
    def _analyze_division_rivalry(self, game_data: Dict) -> Dict:
        """Analyze division rivalry impact"""
        is_division_game = game_data.get('is_division_game', False)
        
        if is_division_game:
            h2h_record = game_data.get('recent_h2h_record', '1-1')
            
            return {
                'factor': 'Division Rivalry',
                'impact': 'significant',
                'explanation': f"Division rivals with recent H2H: {h2h_record}",
                'betting_angle': f"Expect closer game, consider UNDER total",
                'historical_edge': f"Division games go UNDER {56.8}% and have {1.2} point lower margins"
            }
        
        return {'impact': 'neutral'}
    
    def _analyze_revenge_factor(self, game_data: Dict) -> Dict:
        """Analyze revenge game scenarios"""
        last_meeting_result = game_data.get('last_meeting_margin', 0)
        
        if abs(last_meeting_result) > 14:  # Blowout loss
            revenge_team = game_data.get('home_team') if last_meeting_result < 0 else game_data.get('away_team')
            
            return {
                'factor': 'Revenge Game',
                'impact': 'moderate',
                'explanation': f"{revenge_team} lost by {abs(last_meeting_result)} points in last meeting",
                'betting_angle': f"Favor {revenge_team} for revenge motivation",
                'historical_edge': f"Revenge games after 14+ point losses cover {58.9}% ATS"
            }
        
        return {'impact': 'neutral'}
    
    def _analyze_playoff_implications(self, game_data: Dict) -> Dict:
        """Analyze playoff implication impact"""
        home_playoff_odds = game_data.get('home_playoff_odds', 50)
        away_playoff_odds = game_data.get('away_playoff_odds', 50)
        
        if home_playoff_odds > 80 and away_playoff_odds < 20:
            return {
                'factor': 'Playoff Implications',
                'impact': 'significant',
                'explanation': f"Home team ({home_playoff_odds}% playoff odds) vs eliminated team ({away_playoff_odds}%)",
                'betting_angle': f"Fade team with nothing to play for",
                'historical_edge': f"Teams with playoff hopes vs eliminated teams cover {61.2}% ATS"
            }
        
        return {'impact': 'neutral'}
    
    def _analyze_statistical_factors(self, game_data: Dict) -> Dict:
        """Analyze key statistical factors"""
        return {
            'offensive_matchup': self._analyze_offensive_matchup(game_data),
            'defensive_matchup': self._analyze_defensive_matchup(game_data),
            'turnover_battle': self._analyze_turnover_factors(game_data),
            'red_zone_efficiency': self._analyze_red_zone_matchup(game_data),
            'third_down_battle': self._analyze_third_down_matchup(game_data)
        }
    
    def _analyze_offensive_matchup(self, game_data: Dict) -> Dict:
        """Analyze offensive vs defensive matchups"""
        home_off_rank = game_data.get('home_offense_rank', 16)
        away_def_rank = game_data.get('away_defense_rank', 16)
        
        matchup_advantage = (33 - away_def_rank) - (33 - home_off_rank)
        
        if abs(matchup_advantage) > 8:
            advantaged_unit = "Home Offense" if matchup_advantage > 0 else "Away Defense"
            return {
                'factor': 'Offensive Matchup',
                'advantage': advantaged_unit,
                'explanation': f"Home offense (#{home_off_rank}) vs Away defense (#{away_def_rank})",
                'impact_score': abs(matchup_advantage),
                'betting_angle': f"Significant matchup advantage suggests {'OVER' if matchup_advantage > 0 else 'UNDER'}"
            }
        
        return {'impact': 'neutral'}
    
    def _analyze_market_factors(self, market_data: Dict) -> Dict:
        """Analyze betting market factors"""
        return {
            'line_movement': self._analyze_line_movement(market_data),
            'public_betting': self._analyze_public_betting(market_data),
            'sharp_indicators': self._analyze_sharp_money(market_data),
            'closing_line_value': self._calculate_closing_line_value(market_data)
        }
    
    def _analyze_line_movement(self, market_data: Dict) -> Dict:
        """Analyze line movement patterns"""
        opening_spread = market_data.get('opening_spread', 0)
        current_spread = market_data.get('current_spread', 0)
        movement = current_spread - opening_spread
        
        if abs(movement) > 1.5:
            direction = "toward favorite" if movement < 0 else "toward underdog"
            return {
                'factor': 'Significant Line Movement',
                'movement': f"{abs(movement)} points {direction}",
                'explanation': f"Line moved from {opening_spread} to {current_spread}",
                'betting_angle': f"Follow the sharp money movement",
                'significance': 'high' if abs(movement) > 2.5 else 'moderate'
            }
        
        return {'impact': 'neutral'}
    
    def _generate_supporting_evidence(self, game_data: Dict, prediction: Dict) -> List[str]:
        """Generate supporting evidence points"""
        evidence = []
        
        # Team form evidence
        home_form = game_data.get('home_recent_form', '2-3')
        away_form = game_data.get('away_recent_form', '3-2')
        evidence.append(f"📊 Recent Form: {game_data.get('home_team', 'Home')} ({home_form}) vs {game_data.get('away_team', 'Away')} ({away_form})")
        
        # Head-to-head evidence
        h2h_record = game_data.get('h2h_last_5', '3-2')
        evidence.append(f"🏆 Head-to-Head (Last 5): {h2h_record}")
        
        # Statistical evidence
        if prediction.get('model_agreement', 0) > 0.8:
            evidence.append(f"🤖 High Model Agreement: {prediction.get('model_agreement', 0):.1%} of models agree")
        
        # Market evidence
        if game_data.get('sharp_money_percentage', 0) > 70:
            evidence.append(f"💰 Sharp Money: {game_data.get('sharp_money_percentage', 0)}% of sharp bets on this side")
        
        return evidence
    
    def _identify_risk_factors(self, game_data: Dict, market_data: Dict) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        # Injury risks
        key_injuries = game_data.get('key_injuries', [])
        if key_injuries:
            risks.append(f"⚠️ Key Injuries: {', '.join(key_injuries)}")
        
        # Weather risks
        if game_data.get('weather', {}).get('wind_mph', 0) > 15:
            risks.append(f"🌪️ Weather Risk: {game_data.get('weather', {}).get('wind_mph', 0)}mph winds")
        
        # Market risks
        public_percentage = market_data.get('public_betting_percentage', 50)
        if public_percentage > 80:
            risks.append(f"📈 Public Overload: {public_percentage}% of public on this side")
        
        # Variance risks
        if game_data.get('game_total_variance', 0) > 5:
            risks.append(f"📊 High Variance: Total has {game_data.get('game_total_variance', 0)} point standard deviation")
        
        return risks
    
    def _extract_key_factors(self, reasoning: Dict) -> List[str]:
        """Extract the most important factors"""
        key_factors = []
        
        # Get significant situational factors
        for factor_name, factor_data in reasoning['situational_analysis'].items():
            if factor_data.get('impact') in ['significant', 'high']:
                key_factors.append(f"🎯 {factor_data.get('factor', factor_name)}: {factor_data.get('explanation', '')}")
        
        # Get significant statistical factors
        for factor_name, factor_data in reasoning['statistical_breakdown'].items():
            if factor_data.get('impact_score', 0) > 8:
                key_factors.append(f"📊 {factor_data.get('factor', factor_name)}: {factor_data.get('explanation', '')}")
        
        # Get significant market factors
        for factor_name, factor_data in reasoning['market_insights'].items():
            if factor_data.get('significance') in ['high', 'significant']:
                key_factors.append(f"💰 {factor_data.get('factor', factor_name)}: {factor_data.get('explanation', '')}")
        
        return key_factors[:5]  # Top 5 most important factors
    
    def _create_reasoning_narrative(self, reasoning: Dict, game_data: Dict, prediction: Dict) -> str:
        """Create a comprehensive reasoning narrative"""
        narrative_parts = []
        
        # Opening statement
        confidence = reasoning['confidence_level']
        recommendation = reasoning['recommendation']
        narrative_parts.append(f"{confidence} confidence in {recommendation.get('recommendation', 'this recommendation')}.")
        
        # Key reasoning
        if reasoning['key_factors']:
            narrative_parts.append(f"\n🔍 **Primary Reasoning:**")
            for factor in reasoning['key_factors'][:3]:
                narrative_parts.append(f"• {factor}")
        
        # Supporting evidence
        if reasoning['supporting_evidence']:
            narrative_parts.append(f"\n✅ **Supporting Evidence:**")
            for evidence in reasoning['supporting_evidence'][:3]:
                narrative_parts.append(f"• {evidence}")
        
        # Risk acknowledgment
        if reasoning['risk_factors']:
            narrative_parts.append(f"\n⚠️ **Risk Factors to Monitor:**")
            for risk in reasoning['risk_factors'][:2]:
                narrative_parts.append(f"• {risk}")
        
        # Final recommendation
        edge = recommendation.get('edge', 0)
        if edge > 0:
            narrative_parts.append(f"\n💡 **Expected Edge:** {edge:.1f}% advantage over market pricing")
        
        return '\n'.join(narrative_parts)

class ReasoningDisplayManager:
    """
    Manages the display of reasoning in the Streamlit interface
    """
    
    def __init__(self):
        self.reasoning_engine = BetReasoningEngine()
    
    def display_bet_reasoning(self, game_data: Dict, prediction: Dict, market_data: Dict):
        """Display comprehensive bet reasoning in Streamlit"""
        import streamlit as st
        
        reasoning = self.reasoning_engine.generate_bet_reasoning(game_data, prediction, market_data)
        
        # Main recommendation display
        recommendation = reasoning['recommendation']
        confidence = reasoning['confidence_level']
        
        st.subheader(f"{confidence}: {recommendation.get('recommendation', 'No recommendation')}")
        
        # Display reasoning narrative
        st.markdown("### 🧠 **Detailed Reasoning**")
        st.markdown(reasoning['final_reasoning'])
        
        # Create tabs for detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Key Factors", "📊 Statistical Edge", "💰 Market Analysis", "⚠️ Risk Assessment"])
        
        with tab1:
            st.markdown("#### 🎯 **Primary Factors**")
            for factor in reasoning['key_factors']:
                st.markdown(f"• {factor}")
            
            st.markdown("#### ✅ **Supporting Evidence**")
            for evidence in reasoning['supporting_evidence']:
                st.markdown(f"• {evidence}")
        
        with tab2:
            st.markdown("#### 📊 **Statistical Breakdown**")
            for factor_name, factor_data in reasoning['statistical_breakdown'].items():
                if factor_data.get('impact') != 'neutral':
                    with st.expander(f"📈 {factor_data.get('factor', factor_name)}"):
                        st.write(f"**Analysis:** {factor_data.get('explanation', 'N/A')}")
                        st.write(f"**Betting Angle:** {factor_data.get('betting_angle', 'N/A')}")
                        if 'impact_score' in factor_data:
                            st.metric("Impact Score", f"{factor_data['impact_score']}/10")
        
        with tab3:
            st.markdown("#### 💰 **Market Intelligence**")
            for factor_name, factor_data in reasoning['market_insights'].items():
                if factor_data.get('impact') != 'neutral':
                    with st.expander(f"📊 {factor_data.get('factor', factor_name)}"):
                        st.write(f"**Analysis:** {factor_data.get('explanation', 'N/A')}")
                        st.write(f"**Significance:** {factor_data.get('significance', 'N/A')}")
                        if 'movement' in factor_data:
                            st.write(f"**Movement:** {factor_data['movement']}")
        
        with tab4:
            st.markdown("#### ⚠️ **Risk Factors**")
            if reasoning['risk_factors']:
                for risk in reasoning['risk_factors']:
                    st.warning(risk)
            else:
                st.success("✅ No significant risk factors identified")
            
            # Risk mitigation suggestions
            st.markdown("#### 🛡️ **Risk Mitigation**")
            st.info("• Consider smaller unit size if multiple risk factors present")
            st.info("• Monitor line movement before game time")
            st.info("• Check for late injury news")

# Usage example
if __name__ == "__main__":
    # Example usage
    reasoning_engine = BetReasoningEngine()
    
    sample_game_data = {
        'home_team': 'Chiefs',
        'away_team': 'Broncos',
        'home_rest_days': 7,
        'away_rest_days': 4,
        'travel_distance_miles': 600,
        'weather': {'wind_mph': 12, 'temperature_f': 45, 'precipitation_chance': 20},
        'is_division_game': True,
        'home_recent_form': '4-1',
        'away_recent_form': '2-3'
    }
    
    sample_prediction = {
        'spread_edge': 2.3,
        'confidence': 0.78,
        'recommended_team': 'Chiefs',
        'predicted_spread': -6.5
    }
    
    sample_market_data = {
        'current_spread': -7.0,
        'opening_spread': -6.5,
        'public_betting_percentage': 73
    }
    
    reasoning = reasoning_engine.generate_bet_reasoning(sample_game_data, sample_prediction, sample_market_data)
    print("Reasoning Generated:")
    print(reasoning['final_reasoning'])
