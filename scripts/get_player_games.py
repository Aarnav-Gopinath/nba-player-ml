from nba_api.stats.endpoints import playergamelog
import pandas as pd

# Example: LeBron James (player ID: 2544)
player_id = 2544

gamelog = playergamelog.PlayerGameLog(player_id=player_id)
df = gamelog.get_data_frames()[0]

# Save to CSV
df.to_csv("data/player_game_logs.csv", index=False)

print("Saved player game logs to data/player_game_logs.csv")