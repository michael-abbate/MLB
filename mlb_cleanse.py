'''
mlb_cleanse.py

Used to clean and encode mlb csv files
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('mlb_combined_PLOC.csv')

mlb_df = df

# Get temperature
mlb_df['temp'] = mlb_df.apply(lambda row: row.weather.split(', ')[0][0:2], axis=1)
mlb_df['temp'] = mlb_df['temp'].astype('int')

# Get weather condition (forecast)
mlb_df['forecast'] = mlb_df.apply(lambda row: row.weather.split(', ')[1], axis=1)
mlb_df['forecast'] = mlb_df['forecast'].astype('category')

# generate binary forecast columns using pd.get_dummies
mlb_df_new = pd.get_dummies(mlb_df, columns=["forecast"])
mlb_df_new.drop(columns=['weather'], inplace=True)

# binary encoding for p_throws, batters_stance, Batter_HomeAway
le = LabelEncoder()
mlb_df_new['p_throws']= le.fit_transform(mlb_df_new['p_throws'])
mlb_df_new['batters_stance']= le.fit_transform(mlb_df_new['batters_stance'])
mlb_df_new['Batter_HomeAway']= le.fit_transform(mlb_df_new['Batter_HomeAway'])


# Categorize event into 2 groups: out, on-base
def event_outcome(event):
    if event in ["Batter Interference", "Bunt Groundout", "Bunt Lineout", "Bunt Pop Out", "Double Play", "Fielders Choice Out",
                "Flyout", "Forceout", "Grounded Into DP", "Groundout", "Lineout", "Pop Out", "Runner Out", "Sac Bunt", "Sac Fly", "Sac Fly DP",
                "Sacrifice Bunt DP", "Strikeout", "Strikeout - DP", "Triple Play"]:
        return "out"
    else:
        return "on-Base"


#binary encoding for event column
mlb_df_new['event'] = mlb_df_new['event'].apply(lambda x: event_outcome(x))
mlb_df_new['event'] = le.fit_transform(mlb_df_new['event'])


# Categorizes pitch types into 4 groups
def pitchType(pitch):
    if pitch in ['FC','FF','FT']:
        return "regular"
    elif pitch in ['FS','KC','SC','SI', 'SL', 'CU']:
        return "breakingBall"
    elif pitch in ['CH','EP','KN']:
        return "offspeed"
    else:
        return "intentional"

mlb_df_new['pitch_type'] = mlb_df_new['pitch_type'].apply(lambda x: pitchType(x))
mlb_df_new = pd.get_dummies(mlb_df_new, columns=["pitch_type"])

# Reorder columns to put dependent variable in last column
column_reorder = ['ab_id', 'g_id', 'batter_id', 'pitcher_id', 'home_team', 'away_team',
       'inning', 'pitchers_score', 'batters_score', 'p_throws',
       'batters_stance', 'Batter_HomeAway', 'Final_Pitch_Count_Of_At_Bat',
       'start_speed', 'px', 'pz', 'balls', 'strikes', 'outs', 'on_1b', 'on_2b', 'on_3b',
       'temp', 'forecast_clear', 'forecast_cloudy', 'forecast_dome',
       'forecast_drizzle', 'forecast_overcast', 'forecast_partly cloudy',
       'forecast_rain', 'forecast_roof closed', 'forecast_snow',
       'forecast_sunny', 'pitch_type_breakingBall', 'pitch_type_intentional',
       'pitch_type_offspeed', 'pitch_type_regular', 'event']


mlb_df_final = mlb_df_new[column_reorder]

# mlb_df_final is the final, cleaned dataframe
#print(mlb_df_final.head().to_string())
#mlb_df_final.to_csv('mlb_cleaned_updated_PLOC_v2.csv')



