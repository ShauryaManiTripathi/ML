import pandas as pd
# Read the CSV file
df = pd.read_csv('chessboard.csv')

# Exclude the last row and last column
df_trimmed = df.iloc[:-1, :-1]

# If you want to save the trimmed data back to a CSV file
df_trimmed.to_csv('chessboard_trimmed.csv', index=False)

print(df_trimmed)