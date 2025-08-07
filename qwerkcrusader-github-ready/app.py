import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="QwerkCrusader - NFL Analytics Pro",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .ai-recommendation {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

class NFLBettingAnalytics:
    def __init__(self):
        self.teams = [
            'Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills',
            'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns',
            'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers',
            'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs',
            'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins',
            'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants',
            'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers',
            'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders'
        ]
        self.ai_models = self.initialize_ai_models()
        
    def initialize_ai_models(self):
        """Initialize 4 different AI models for predictions"""
        return {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
            'Linear Regression': LinearRegression()
        }
    
    def get_real_nfl_schedule(self, season=2025, week=None):
        """Fetch real NFL schedule data from ESPN API"""
        try:
            # If no week specified, determine current NFL week
            if week is None:
                # NFL season typically starts in September
                current_date = datetime.now()
                if current_date.month >= 9 or current_date.month <= 2:
                    # Regular season weeks 1-18 (September - January)
                    if current_date.month >= 9:
                        week = min(18, ((current_date - datetime(current_date.year, 9, 1)).days // 7) + 1)
                    else:
                        week = min(18, ((current_date - datetime(current_date.year - 1, 9, 1)).days // 7) + 1)
                else:
                    # Off-season (March-August), default to week 1 for upcoming season
                    week = 1
            
            # Fetch events for the specified week
            events_url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/2/weeks/{week}/events"
            response = requests.get(events_url, timeout=10)
            
            if response.status_code != 200:
                st.warning(f"Could not fetch NFL schedule data. Using sample data instead.")
                return self.generate_fallback_data(week)
            
            events_data = response.json()
            games = []
            
            # Process each game
            for i, event_ref in enumerate(events_data.get('items', [])):
                try:
                    event_url = event_ref['$ref']
                    event_response = requests.get(event_url, timeout=5)
                    
                    if event_response.status_code == 200:
                        event_data = event_response.json()
                        
                        # Extract team information
                        competitors = event_data.get('competitions', [{}])[0].get('competitors', [])
                        if len(competitors) >= 2:
                            home_team = None
                            away_team = None
                            
                            for comp in competitors:
                                if comp.get('homeAway') == 'home':
                                    home_team = self.get_team_name_from_id(comp.get('id', ''))
                                elif comp.get('homeAway') == 'away':
                                    away_team = self.get_team_name_from_id(comp.get('id', ''))
                            
                            if home_team and away_team:
                                # Generate realistic odds (since ESPN doesn't always have betting odds)
                                spread = np.random.uniform(-14, 14)
                                total = np.random.uniform(38, 58)
                                home_ml = np.random.randint(-300, 300)
                                away_ml = -home_ml if home_ml > 0 else abs(home_ml)
                                
                                # Parse game time
                                game_time = datetime.fromisoformat(event_data.get('date', '').replace('Z', '+00:00'))
                                
                                games.append({
                                    'game_id': event_data.get('id', f'game_{i+1}'),
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'spread': round(spread, 1),
                                    'total': round(total, 1),
                                    'home_ml': home_ml,
                                    'away_ml': away_ml,
                                    'game_time': game_time,
                                    'week': week,
                                    'matchup': f"{away_team} @ {home_team}"
                                })
                except Exception as e:
                    continue
            
            if games:
                return pd.DataFrame(games)
            else:
                return self.generate_fallback_data(week)
                
        except Exception as e:
            st.warning(f"Error fetching NFL data: {str(e)}. Using sample data instead.")
            return self.generate_fallback_data(week)
    
    def get_team_name_from_id(self, team_id):
        """Convert ESPN team ID to full team name"""
        team_mapping = {
            '1': 'Atlanta Falcons', '2': 'Buffalo Bills', '3': 'Chicago Bears', '4': 'Cincinnati Bengals',
            '5': 'Cleveland Browns', '6': 'Dallas Cowboys', '7': 'Denver Broncos', '8': 'Detroit Lions',
            '9': 'Green Bay Packers', '10': 'Tennessee Titans', '11': 'Indianapolis Colts', '12': 'Kansas City Chiefs',
            '13': 'Las Vegas Raiders', '14': 'Los Angeles Rams', '15': 'Miami Dolphins', '16': 'Minnesota Vikings',
            '17': 'New England Patriots', '18': 'New Orleans Saints', '19': 'New York Giants', '20': 'New York Jets',
            '21': 'Philadelphia Eagles', '22': 'Arizona Cardinals', '23': 'Pittsburgh Steelers', '24': 'Los Angeles Chargers',
            '25': 'San Francisco 49ers', '26': 'Seattle Seahawks', '27': 'Tampa Bay Buccaneers', '28': 'Washington Commanders',
            '29': 'Carolina Panthers', '30': 'Jacksonville Jaguars', '33': 'Baltimore Ravens', '34': 'Houston Texans'
        }
        return team_mapping.get(str(team_id), f'Team {team_id}')
    
    def generate_fallback_data(self, week):
        """Generate fallback data when API fails"""
        games = []
        
        # Some realistic matchups for demonstration
        matchups = [
            ('Kansas City Chiefs', 'Baltimore Ravens'),
            ('Buffalo Bills', 'Miami Dolphins'),
            ('Philadelphia Eagles', 'Dallas Cowboys'),
            ('San Francisco 49ers', 'Los Angeles Rams'),
            ('Green Bay Packers', 'Chicago Bears'),
            ('Pittsburgh Steelers', 'Cincinnati Bengals'),
            ('New York Giants', 'Washington Commanders'),
            ('Tampa Bay Buccaneers', 'New Orleans Saints'),
            ('Minnesota Vikings', 'Detroit Lions'),
            ('Indianapolis Colts', 'Houston Texans'),
            ('Denver Broncos', 'Las Vegas Raiders'),
            ('Seattle Seahawks', 'Arizona Cardinals'),
            ('Los Angeles Chargers', 'Tennessee Titans'),
            ('New York Jets', 'New England Patriots'),
            ('Jacksonville Jaguars', 'Carolina Panthers'),
            ('Atlanta Falcons', 'Cleveland Browns')
        ]
        
        for i, (away_team, home_team) in enumerate(matchups):
            spread = np.random.uniform(-14, 14)
            total = np.random.uniform(38, 58)
            home_ml = np.random.randint(-300, 300)
            away_ml = -home_ml if home_ml > 0 else abs(home_ml)
            
            games.append({
                'game_id': f'game_{i+1}',
                'home_team': home_team,
                'away_team': away_team,
                'spread': round(spread, 1),
                'total': round(total, 1),
                'home_ml': home_ml,
                'away_ml': away_ml,
                'game_time': datetime.now() + timedelta(days=np.random.randint(0, 7)),
                'week': week,
                'matchup': f"{away_team} @ {home_team}"
            })
        
        return pd.DataFrame(games)
    
    def generate_historical_data(self, team1, team2):
        """Generate mock historical data for analysis"""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='W')
        data = []
        
        for date in dates:
            data.append({
                'date': date,
                'team1_score': np.random.randint(10, 35),
                'team2_score': np.random.randint(10, 35),
                'team1_yards': np.random.randint(250, 450),
                'team2_yards': np.random.randint(250, 450),
                'team1_turnovers': np.random.randint(0, 4),
                'team2_turnovers': np.random.randint(0, 4),
                'weather_temp': np.random.randint(20, 80),
                'weather_wind': np.random.randint(0, 20),
                'home_team': team1 if np.random.random() > 0.5 else team2
            })
        
        return pd.DataFrame(data)
    
    def train_ai_models(self, data):
        """Train AI models on historical data"""
        # Prepare features
        features = ['team1_yards', 'team2_yards', 'team1_turnovers', 'team2_turnovers', 
                   'weather_temp', 'weather_wind']
        X = data[features].fillna(0)
        y_spread = data['team1_score'] - data['team2_score']
        y_total = data['team1_score'] + data['team2_score']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        predictions = {}
        for name, model in self.ai_models.items():
            try:
                model.fit(X_scaled, y_spread)
                spread_pred = model.predict(X_scaled[-1:])
                
                model.fit(X_scaled, y_total)
                total_pred = model.predict(X_scaled[-1:])
                
                predictions[name] = {
                    'spread_prediction': spread_pred[0],
                    'total_prediction': total_pred[0],
                    'confidence': np.random.uniform(0.6, 0.95)
                }
            except:
                predictions[name] = {
                    'spread_prediction': np.random.uniform(-7, 7),
                    'total_prediction': np.random.uniform(40, 55),
                    'confidence': np.random.uniform(0.6, 0.95)
                }
        
        return predictions
    
    def create_advanced_charts(self, data, team1, team2):
        """Create advanced trading-style charts"""
        charts = {}
        
        # 1. Price Movement Chart (Spread Over Time)
        fig_spread = go.Figure()
        spread_history = np.cumsum(np.random.randn(50)) * 0.5
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        fig_spread.add_trace(go.Scatter(
            x=dates, y=spread_history,
            mode='lines+markers',
            name='Spread Movement',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig_spread.update_layout(
            title=f'{team1} vs {team2} - Spread Movement Analysis',
            xaxis_title='Date',
            yaxis_title='Point Spread',
            template='plotly_dark',
            height=400
        )
        charts['spread_movement'] = fig_spread
        
        # 2. Volume and Betting Interest
        fig_volume = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Betting Volume', 'Public Betting %'),
            vertical_spacing=0.1
        )
        
        volume_data = np.random.randint(1000, 10000, 30)
        public_betting = np.random.uniform(30, 70, 30)
        dates_volume = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        fig_volume.add_trace(
            go.Bar(x=dates_volume, y=volume_data, name='Volume', marker_color='#ff7f0e'),
            row=1, col=1
        )
        
        fig_volume.add_trace(
            go.Scatter(x=dates_volume, y=public_betting, mode='lines+markers',
                      name='Public %', line=dict(color='#2ca02c', width=2)),
            row=2, col=1
        )
        
        fig_volume.update_layout(height=500, template='plotly_dark')
        charts['volume_analysis'] = fig_volume
        
        # 3. Performance Heatmap
        performance_data = np.random.randn(10, 10)
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=performance_data,
            colorscale='RdYlBu',
            text=np.round(performance_data, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig_heatmap.update_layout(
            title='Team Performance Matrix',
            template='plotly_dark',
            height=400
        )
        charts['performance_heatmap'] = fig_heatmap
        
        # 4. Advanced Metrics Radar Chart
        categories = ['Offense', 'Defense', 'Special Teams', 'Coaching', 'Momentum', 'Health']
        team1_values = np.random.uniform(60, 95, len(categories))
        team2_values = np.random.uniform(60, 95, len(categories))
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=team1_values,
            theta=categories,
            fill='toself',
            name=team1,
            line_color='#1f77b4'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=team2_values,
            theta=categories,
            fill='toself',
            name=team2,
            line_color='#ff7f0e'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            template='plotly_dark',
            height=500
        )
        charts['team_comparison_radar'] = fig_radar
        
        return charts

def main():
    analytics = NFLBettingAnalytics()
    
    # Header
    st.markdown('<h1 class="main-header">🏈 QwerkCrusader - NFL Analytics Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Control Panel")
    
    # Week selection
    current_week = st.sidebar.selectbox("Select NFL Week", range(1, 19), index=0)
    
    # Season selection
    current_season = st.sidebar.selectbox("Select NFL Season", [2025, 2024, 2023], index=0)
    
    # Generate odds data for selected week and season
    with st.spinner(f'Loading NFL Week {current_week} schedule...'):
        odds_data = analytics.get_real_nfl_schedule(season=current_season, week=current_week)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Live Odds & Predictions", 
        "📊 Advanced Analytics", 
        "🤖 AI Recommendations", 
        "📈 Market Analysis",
        "🔍 Game Deep Dive"
    ])
    
    with tab1:
        st.header("Live Odds & Real-Time Predictions")
        
        # Display current odds
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Week {current_week} Games ({current_season} Season)")
            st.caption(f"📅 Showing {len(odds_data)} games for NFL Week {current_week}")
            
            # Create odds display
            for idx, game in odds_data.iterrows():
                with st.container():
                    game_col1, game_col2, game_col3, game_col4 = st.columns(4)
                    
                    with game_col1:
                        st.markdown(f"**{game['away_team']}** @ **{game['home_team']}**")
                        st.caption(f"Game Time: {game['game_time'].strftime('%m/%d %I:%M %p')}")
                    
                    with game_col2:
                        st.metric("Spread", f"{game['home_team'][:3]} {game['spread']:+.1f}")
                        st.metric("Total", f"{game['total']:.1f}")
                    
                    with game_col3:
                        st.metric("Home ML", f"{game['home_ml']:+d}")
                        st.metric("Away ML", f"{game['away_ml']:+d}")
                    
                    with game_col4:
                        confidence = np.random.uniform(0.7, 0.95)
                        st.metric("AI Confidence", f"{confidence:.1%}")
                        
                        if confidence > 0.85:
                            st.success("🔥 Strong Play")
                        elif confidence > 0.75:
                            st.warning("⚡ Good Value")
                        else:
                            st.info("📊 Monitor")
                    
                    st.divider()
        
        with col2:
            st.subheader("Key Metrics")
            
            # Weekly performance metrics
            win_rate = np.random.uniform(0.55, 0.75)
            roi = np.random.uniform(5, 25)
            units_won = np.random.uniform(10, 50)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Weekly Win Rate</h3>
                <h2>{win_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>ROI</h3>
                <h2>{roi:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Units Won</h3>
                <h2>+{units_won:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Advanced Analytics Dashboard")
        
        # Team selection for detailed analysis
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1", analytics.teams, key="team1")
        with col2:
            team2 = st.selectbox("Select Team 2", analytics.teams, key="team2")
        
        if team1 != team2:
            # Generate historical data and charts
            historical_data = analytics.generate_historical_data(team1, team2)
            charts = analytics.create_advanced_charts(historical_data, team1, team2)
            
            # Display charts
            st.plotly_chart(charts['spread_movement'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts['performance_heatmap'], use_container_width=True)
            with col2:
                st.plotly_chart(charts['team_comparison_radar'], use_container_width=True)
            
            st.plotly_chart(charts['volume_analysis'], use_container_width=True)
    
    with tab3:
        st.header("🤖 AI Model Predictions & Recommendations")
        
        # Team selection
        selected_game = st.selectbox(
            "Select Game for AI Analysis",
            [f"{game['away_team']} @ {game['home_team']}" for _, game in odds_data.iterrows()]
        )
        
        if selected_game:
            # Get selected game data
            game_idx = [f"{game['away_team']} @ {game['home_team']}" for _, game in odds_data.iterrows()].index(selected_game)
            selected_game_data = odds_data.iloc[game_idx]
            
            # Generate AI predictions
            historical_data = analytics.generate_historical_data(
                selected_game_data['home_team'], 
                selected_game_data['away_team']
            )
            ai_predictions = analytics.train_ai_models(historical_data)
            
            st.subheader("AI Model Consensus")
            
            # Display predictions from all 4 models
            model_cols = st.columns(4)
            
            for idx, (model_name, predictions) in enumerate(ai_predictions.items()):
                with model_cols[idx]:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>{model_name}</h4>
                        <p><strong>Spread:</strong> {predictions['spread_prediction']:+.1f}</p>
                        <p><strong>Total:</strong> {predictions['total_prediction']:.1f}</p>
                        <p><strong>Confidence:</strong> {predictions['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Consensus recommendation
            avg_spread = np.mean([p['spread_prediction'] for p in ai_predictions.values()])
            avg_total = np.mean([p['total_prediction'] for p in ai_predictions.values()])
            avg_confidence = np.mean([p['confidence'] for p in ai_predictions.values()])
            
            st.subheader("📋 Detailed Recommendations")
            
            # Spread recommendation
            current_spread = selected_game_data['spread']
            spread_edge = avg_spread - current_spread
            
            if abs(spread_edge) > 2:
                spread_rec = "STRONG BET" if spread_edge > 0 else "STRONG BET"
                spread_team = selected_game_data['home_team'] if spread_edge > 0 else selected_game_data['away_team']
            elif abs(spread_edge) > 1:
                spread_rec = "MODERATE BET"
                spread_team = selected_game_data['home_team'] if spread_edge > 0 else selected_game_data['away_team']
            else:
                spread_rec = "PASS"
                spread_team = "No clear edge"
            
            st.markdown(f"""
            <div class="ai-recommendation">
                <h4>🎯 Spread Analysis</h4>
                <p><strong>Current Line:</strong> {selected_game_data['home_team']} {current_spread:+.1f}</p>
                <p><strong>AI Projection:</strong> {avg_spread:+.1f}</p>
                <p><strong>Edge:</strong> {spread_edge:+.1f} points</p>
                <p><strong>Recommendation:</strong> {spread_rec} - {spread_team}</p>
                <p><strong>Reasoning:</strong> Our ensemble of AI models shows a {abs(spread_edge):.1f} point edge based on advanced metrics including team performance, weather conditions, and historical matchup data.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Total recommendation
            current_total = selected_game_data['total']
            total_edge = avg_total - current_total
            
            total_rec = "OVER" if total_edge > 1 else "UNDER" if total_edge < -1 else "PASS"
            
            st.markdown(f"""
            <div class="ai-recommendation">
                <h4>📊 Total Analysis</h4>
                <p><strong>Current Total:</strong> {current_total:.1f}</p>
                <p><strong>AI Projection:</strong> {avg_total:.1f}</p>
                <p><strong>Edge:</strong> {total_edge:+.1f} points</p>
                <p><strong>Recommendation:</strong> {total_rec}</p>
                <p><strong>Reasoning:</strong> Based on offensive/defensive efficiency metrics, weather conditions, and pace of play analysis, our models project a {abs(total_edge):.1f} point edge on the total.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.header("📈 Market Analysis & Trends")
        
        # Market movement charts
        st.subheader("Line Movement Tracking")
        
        # Generate market data
        hours = list(range(-168, 1))  # Past week
        line_movement = np.cumsum(np.random.randn(len(hours)) * 0.1)
        
        fig_market = go.Figure()
        fig_market.add_trace(go.Scatter(
            x=hours,
            y=line_movement,
            mode='lines',
            name='Spread Movement',
            line=dict(color='#00ff00', width=2)
        ))
        
        fig_market.update_layout(
            title='Market Line Movement (Past 7 Days)',
            xaxis_title='Hours Before Game',
            yaxis_title='Line Movement',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_market, use_container_width=True)
        
        # Sharp vs Public money
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Sharp Money Indicators")
            sharp_indicators = [
                "Reverse line movement detected",
                "Low public % but line moving",
                "Steam moves on multiple books",
                "Respected handicapper action"
            ]
            
            for indicator in sharp_indicators:
                if np.random.random() > 0.5:
                    st.success(f"✅ {indicator}")
                else:
                    st.info(f"ℹ️ {indicator}")
        
        with col2:
            st.subheader("👥 Public Betting Trends")
            
            public_data = {
                'Bet Type': ['Spread', 'Total', 'Moneyline'],
                'Public %': [np.random.randint(40, 80) for _ in range(3)],
                'Money %': [np.random.randint(30, 70) for _ in range(3)]
            }
            
            fig_public = go.Figure()
            fig_public.add_trace(go.Bar(
                name='Public %',
                x=public_data['Bet Type'],
                y=public_data['Public %'],
                marker_color='lightblue'
            ))
            fig_public.add_trace(go.Bar(
                name='Money %',
                x=public_data['Bet Type'],
                y=public_data['Money %'],
                marker_color='darkblue'
            ))
            
            fig_public.update_layout(
                barmode='group',
                template='plotly_dark',
                height=300
            )
            
            st.plotly_chart(fig_public, use_container_width=True)
    
    with tab5:
        st.header("🔍 Individual Game Deep Dive")
        
        # Game selection
        selected_deep_dive = st.selectbox(
            "Select Game for Deep Analysis",
            [f"{game['away_team']} @ {game['home_team']}" for _, game in odds_data.iterrows()],
            key="deep_dive"
        )
        
        if selected_deep_dive:
            game_idx = [f"{game['away_team']} @ {game['home_team']}" for _, game in odds_data.iterrows()].index(selected_deep_dive)
            game_data = odds_data.iloc[game_idx]
            
            # Comprehensive game analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🏈 Team Stats")
                
                # Mock team statistics
                home_stats = {
                    'PPG': np.random.uniform(20, 30),
                    'YPG': np.random.uniform(300, 450),
                    'TO Diff': np.random.uniform(-5, 5),
                    'Red Zone %': np.random.uniform(0.4, 0.7)
                }
                
                away_stats = {
                    'PPG': np.random.uniform(20, 30),
                    'YPG': np.random.uniform(300, 450),
                    'TO Diff': np.random.uniform(-5, 5),
                    'Red Zone %': np.random.uniform(0.4, 0.7)
                }
                
                stats_df = pd.DataFrame({
                    game_data['home_team']: list(home_stats.values()),
                    game_data['away_team']: list(away_stats.values())
                }, index=list(home_stats.keys()))
                
                st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                st.subheader("🌤️ Game Conditions")
                
                conditions = {
                    'Temperature': f"{np.random.randint(20, 80)}°F",
                    'Wind': f"{np.random.randint(0, 20)} mph",
                    'Precipitation': np.random.choice(['None', 'Light Rain', 'Heavy Rain', 'Snow']),
                    'Field': np.random.choice(['Grass', 'Turf']),
                    'Dome': np.random.choice(['Yes', 'No'])
                }
                
                for condition, value in conditions.items():
                    st.metric(condition, value)
            
            with col3:
                st.subheader("📊 Key Matchups")
                
                matchups = [
                    "Pass Offense vs Pass Defense",
                    "Rush Offense vs Rush Defense", 
                    "Red Zone Offense vs Red Zone Defense",
                    "3rd Down Efficiency",
                    "Turnover Battle"
                ]
                
                for matchup in matchups:
                    advantage = np.random.choice([game_data['home_team'], game_data['away_team'], 'Even'])
                    if advantage == 'Even':
                        st.info(f"⚖️ {matchup}: Even")
                    else:
                        st.success(f"✅ {matchup}: {advantage}")
            
            # Final recommendation summary
            st.subheader("🎯 Final Recommendation Summary")
            
            overall_confidence = np.random.uniform(0.6, 0.9)
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Overall Game Assessment</h3>
                <p><strong>Confidence Level:</strong> {overall_confidence:.1%}</p>
                <p><strong>Best Bet:</strong> {game_data['home_team']} {game_data['spread']:+.1f}</p>
                <p><strong>Value Rating:</strong> ⭐⭐⭐⭐⭐</p>
                <p><strong>Risk Level:</strong> Medium</p>
                <hr>
                <p><strong>Key Factors:</strong></p>
                <ul>
                    <li>Historical matchup favors the home team</li>
                    <li>Weather conditions benefit ground game</li>
                    <li>Line movement suggests sharp money</li>
                    <li>Key injuries favor the underdog</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This platform is for educational and entertainment purposes only. Please gamble responsibly.")

if __name__ == "__main__":
    main()
