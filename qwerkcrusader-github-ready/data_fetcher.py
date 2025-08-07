import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from bs4 import BeautifulSoup

class NFLDataFetcher:
    """
    Comprehensive NFL data fetcher for real-time odds, team stats, and historical data
    """
    
    def __init__(self):
        self.base_urls = {
            'odds_api': 'https://api.the-odds-api.com/v4',
            'espn_api': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl',
            'pro_football_ref': 'https://www.pro-football-reference.com'
        }
        
        # You'll need to get a free API key from https://the-odds-api.com/
        self.odds_api_key = "YOUR_ODDS_API_KEY_HERE"
        
    def fetch_live_odds(self, sport='americanfootball_nfl'):
        """
        Fetch live NFL odds from multiple sportsbooks
        """
        if self.odds_api_key == "YOUR_ODDS_API_KEY_HERE":
            # Return mock data if no API key
            return self._generate_mock_odds()
        
        try:
            url = f"{self.base_urls['odds_api']}/sports/{sport}/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            odds_data = response.json()
            return self._process_odds_data(odds_data)
            
        except Exception as e:
            print(f"Error fetching live odds: {e}")
            return self._generate_mock_odds()
    
    def fetch_team_stats(self, season=2024):
        """
        Fetch comprehensive team statistics
        """
        try:
            url = f"{self.base_urls['espn_api']}/teams"
            response = requests.get(url)
            response.raise_for_status()
            
            teams_data = response.json()
            return self._process_team_stats(teams_data)
            
        except Exception as e:
            print(f"Error fetching team stats: {e}")
            return self._generate_mock_team_stats()
    
    def fetch_player_stats(self, week=None):
        """
        Fetch player statistics and injury reports
        """
        try:
            # This would integrate with ESPN or NFL API
            # For now, return mock data
            return self._generate_mock_player_stats()
            
        except Exception as e:
            print(f"Error fetching player stats: {e}")
            return self._generate_mock_player_stats()
    
    def fetch_weather_data(self, city, date):
        """
        Fetch weather data for game locations
        """
        try:
            # This would integrate with a weather API like OpenWeatherMap
            # For now, return mock data
            return {
                'temperature': np.random.randint(20, 80),
                'wind_speed': np.random.randint(0, 25),
                'precipitation': np.random.choice([0, 0, 0, 0.1, 0.3, 0.5]),
                'humidity': np.random.randint(30, 90),
                'conditions': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Snow'])
            }
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def fetch_historical_matchups(self, team1, team2, years=5):
        """
        Fetch historical head-to-head matchup data
        """
        try:
            # This would scrape Pro Football Reference or use NFL API
            # For now, return mock data
            return self._generate_mock_historical_data(team1, team2, years)
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return self._generate_mock_historical_data(team1, team2, years)
    
    def fetch_betting_trends(self, game_id):
        """
        Fetch betting trends and public money percentages
        """
        try:
            # This would integrate with Action Network or similar
            # For now, return mock data
            return {
                'public_bet_percentage': {
                    'spread': np.random.randint(30, 80),
                    'total': np.random.randint(40, 70),
                    'moneyline': np.random.randint(35, 75)
                },
                'public_money_percentage': {
                    'spread': np.random.randint(25, 75),
                    'total': np.random.randint(35, 65),
                    'moneyline': np.random.randint(30, 70)
                },
                'sharp_indicators': {
                    'reverse_line_movement': np.random.choice([True, False]),
                    'steam_moves': np.random.randint(0, 5),
                    'line_freeze': np.random.choice([True, False])
                }
            }
            
        except Exception as e:
            print(f"Error fetching betting trends: {e}")
            return None
    
    def _process_odds_data(self, raw_data):
        """Process raw odds API data into structured format"""
        processed_games = []
        
        for game in raw_data:
            game_data = {
                'game_id': game['id'],
                'sport': game['sport_key'],
                'commence_time': game['commence_time'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'bookmakers': {}
            }
            
            for bookmaker in game['bookmakers']:
                book_name = bookmaker['title']
                game_data['bookmakers'][book_name] = {}
                
                for market in bookmaker['markets']:
                    market_key = market['key']
                    game_data['bookmakers'][book_name][market_key] = market['outcomes']
            
            processed_games.append(game_data)
        
        return processed_games
    
    def _generate_mock_odds(self):
        """Generate realistic mock odds data"""
        teams = [
            'Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills',
            'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns',
            'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers',
            'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs',
            'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins',
            'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants',
            'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers',
            'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders'
        ]
        
        games = []
        for i in range(16):  # 16 games per week
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Generate realistic odds with slight variations across books
            base_spread = np.random.uniform(-14, 14)
            base_total = np.random.uniform(38, 58)
            
            bookmakers = {}
            for book in ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet']:
                spread_variation = np.random.uniform(-0.5, 0.5)
                total_variation = np.random.uniform(-1, 1)
                
                bookmakers[book] = {
                    'spreads': {
                        'home': base_spread + spread_variation,
                        'away': -(base_spread + spread_variation)
                    },
                    'totals': {
                        'over': base_total + total_variation,
                        'under': base_total + total_variation
                    },
                    'moneylines': {
                        'home': np.random.randint(-300, 300),
                        'away': np.random.randint(-300, 300)
                    }
                }
            
            games.append({
                'game_id': f'game_{i+1}',
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': datetime.now() + timedelta(days=np.random.randint(0, 7)),
                'bookmakers': bookmakers
            })
        
        return games
    
    def _generate_mock_team_stats(self):
        """Generate mock team statistics"""
        teams = [
            'Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills',
            'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns',
            'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers',
            'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs',
            'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins',
            'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants',
            'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers',
            'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders'
        ]
        
        team_stats = {}
        for team in teams:
            team_stats[team] = {
                'offense': {
                    'points_per_game': np.random.uniform(18, 32),
                    'yards_per_game': np.random.uniform(280, 420),
                    'passing_yards_per_game': np.random.uniform(180, 300),
                    'rushing_yards_per_game': np.random.uniform(80, 180),
                    'turnovers_per_game': np.random.uniform(0.8, 2.2),
                    'red_zone_efficiency': np.random.uniform(0.45, 0.75),
                    'third_down_conversion': np.random.uniform(0.35, 0.55)
                },
                'defense': {
                    'points_allowed_per_game': np.random.uniform(18, 32),
                    'yards_allowed_per_game': np.random.uniform(280, 420),
                    'passing_yards_allowed_per_game': np.random.uniform(180, 300),
                    'rushing_yards_allowed_per_game': np.random.uniform(80, 180),
                    'takeaways_per_game': np.random.uniform(0.8, 2.2),
                    'red_zone_defense': np.random.uniform(0.45, 0.75),
                    'third_down_defense': np.random.uniform(0.35, 0.55)
                },
                'special_teams': {
                    'field_goal_percentage': np.random.uniform(0.75, 0.95),
                    'punt_return_average': np.random.uniform(8, 15),
                    'kick_return_average': np.random.uniform(20, 28)
                },
                'record': {
                    'wins': np.random.randint(0, 17),
                    'losses': np.random.randint(0, 17),
                    'ats_record': f"{np.random.randint(5, 12)}-{np.random.randint(5, 12)}-{np.random.randint(0, 3)}",
                    'over_under_record': f"{np.random.randint(6, 11)}-{np.random.randint(6, 11)}-{np.random.randint(0, 2)}"
                }
            }
        
        return team_stats
    
    def _generate_mock_player_stats(self):
        """Generate mock player statistics and injury data"""
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        
        players = []
        for i in range(100):  # Generate 100 random players
            players.append({
                'name': f'Player {i+1}',
                'team': np.random.choice([
                    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
                    'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
                    'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
                    'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
                ]),
                'position': np.random.choice(positions),
                'injury_status': np.random.choice(['Healthy', 'Questionable', 'Doubtful', 'Out'], p=[0.7, 0.15, 0.1, 0.05]),
                'fantasy_points': np.random.uniform(5, 25),
                'projected_points': np.random.uniform(8, 22)
            })
        
        return players
    
    def _generate_mock_historical_data(self, team1, team2, years):
        """Generate mock historical matchup data"""
        games = []
        
        for year in range(2024 - years, 2024):
            # Generate 1-2 games per year
            for game_num in range(np.random.randint(1, 3)):
                games.append({
                    'date': datetime(year, np.random.randint(9, 12), np.random.randint(1, 28)),
                    'home_team': np.random.choice([team1, team2]),
                    'away_team': team2 if games[-1]['home_team'] == team1 else team1 if len(games) > 0 else np.random.choice([team1, team2]),
                    'home_score': np.random.randint(10, 35),
                    'away_score': np.random.randint(10, 35),
                    'total_score': 0,  # Will be calculated
                    'spread_result': 0,  # Will be calculated
                    'weather': {
                        'temperature': np.random.randint(20, 80),
                        'wind': np.random.randint(0, 20),
                        'conditions': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Snow'])
                    }
                })
                
                # Calculate derived stats
                games[-1]['total_score'] = games[-1]['home_score'] + games[-1]['away_score']
                games[-1]['spread_result'] = games[-1]['home_score'] - games[-1]['away_score']
        
        return games

class RealTimeOddsTracker:
    """
    Track odds movements and detect sharp money indicators
    """
    
    def __init__(self):
        self.odds_history = {}
        self.alert_thresholds = {
            'spread_movement': 1.0,  # Alert if spread moves 1+ points
            'total_movement': 2.0,   # Alert if total moves 2+ points
            'reverse_line_movement': True  # Alert on reverse line movement
        }
    
    def track_odds_movement(self, current_odds, game_id):
        """Track and analyze odds movements"""
        if game_id not in self.odds_history:
            self.odds_history[game_id] = []
        
        self.odds_history[game_id].append({
            'timestamp': datetime.now(),
            'odds': current_odds
        })
        
        return self._analyze_movement(game_id)
    
    def _analyze_movement(self, game_id):
        """Analyze odds movement patterns"""
        history = self.odds_history[game_id]
        
        if len(history) < 2:
            return None
        
        current = history[-1]['odds']
        previous = history[-2]['odds']
        
        analysis = {
            'spread_movement': 0,
            'total_movement': 0,
            'reverse_line_movement': False,
            'steam_detected': False,
            'alerts': []
        }
        
        # Calculate movements (using first bookmaker as reference)
        first_book = list(current['bookmakers'].keys())[0]
        
        if first_book in previous['bookmakers']:
            # Spread movement
            current_spread = current['bookmakers'][first_book]['spreads']['home']
            previous_spread = previous['bookmakers'][first_book]['spreads']['home']
            analysis['spread_movement'] = current_spread - previous_spread
            
            # Total movement
            current_total = current['bookmakers'][first_book]['totals']['over']
            previous_total = previous['bookmakers'][first_book]['totals']['over']
            analysis['total_movement'] = current_total - previous_total
            
            # Check for alerts
            if abs(analysis['spread_movement']) >= self.alert_thresholds['spread_movement']:
                analysis['alerts'].append(f"Significant spread movement: {analysis['spread_movement']:+.1f}")
            
            if abs(analysis['total_movement']) >= self.alert_thresholds['total_movement']:
                analysis['alerts'].append(f"Significant total movement: {analysis['total_movement']:+.1f}")
        
        return analysis

# Usage example
if __name__ == "__main__":
    fetcher = NFLDataFetcher()
    
    # Fetch live odds
    odds = fetcher.fetch_live_odds()
    print(f"Fetched {len(odds)} games with live odds")
    
    # Fetch team stats
    team_stats = fetcher.fetch_team_stats()
    print(f"Fetched stats for {len(team_stats)} teams")
    
    # Initialize odds tracker
    tracker = RealTimeOddsTracker()
    
    # Track odds for first game
    if odds:
        movement = tracker.track_odds_movement(odds[0], odds[0]['game_id'])
        if movement:
            print(f"Odds movement detected: {movement}")
