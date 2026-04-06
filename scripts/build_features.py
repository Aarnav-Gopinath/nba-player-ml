import pandas as pd

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("data/player_game_logs.csv")

# -------------------------
# Standardize column names for the script
# -------------------------
df = df.rename(columns={
    'Player_ID': 'player_id',
    'GAME_DATE': 'date',
    'MIN': 'minutes',
    'PTS': 'points',
    'MATCHUP': 'matchup'
})

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by player and date (VERY IMPORTANT for rolling features)
df = df.sort_values(by=['player_id', 'date'])

# -------------------------
# Rolling Features (last 5 games)
# -------------------------
df['last5_minutes'] = (
    df.groupby('Player_ID')['minutes']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

df['last5_points'] = (
    df.groupby('Player_ID')['points']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# -------------------------
# Rest Days Feature
# -------------------------
df['rest_days'] = (
    df.groupby('player_id')['date']
    .diff()
    .dt.days
)
df['rest_days'] = df['rest_days'].fillna(0)  # fill first game NaN with 0

# -------------------------
# Home vs Away Feature
# -------------------------
# Assuming matchup looks like "LAL vs BOS" or "LAL @ BOS"
df['home_game'] = df['matchup'].apply(lambda x: 1 if "vs" in x else 0)

# -------------------------
# Save features
# -------------------------
df.to_csv("data/features.csv", index=False)
print("Saved features to data/features.csv")