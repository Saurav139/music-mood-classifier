import pandas as pd
def build_dense_dataframe(fname, num_datapoints):
    songs = pd.read_csv(fname)

    # Select only the required number of datapoints
    songs = songs.head(num_datapoints)

    # Normalization factors
    features_to_normalize = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                             'instrumentalness', 'liveness', 'tempo', 'duration_ms']
    normalization_factors = {feature: (songs[feature].mean(), songs[feature].std()) for feature in features_to_normalize}

    # Creating a list to hold each row of the DataFrame
    data = []

    for _, row in songs.iterrows():
        # Create a dictionary for each song
        song_data = {}

        # Artist features
        artists = eval(row.artist_ids)
        for a in artists:
            song_data[f'artist_{a}'] = 1

        # Explicit feature
        song_data['explicit'] = 1 if row.explicit else 0

        # Normalized features
        for feature in features_to_normalize:
            song_data[feature] = (row[feature] - normalization_factors[feature][0]) / normalization_factors[feature][1]

        # Key, mode, time signature, and release year features
        song_data[f'key_{str(row.key).split(".")[0]}'] = 1
        song_data['major_mode'] = 1 if row['mode'] == 1 else 0
        song_data['minor_mode'] = 0 if row['mode'] == 1 else 1
        song_data[f'time_signature_{str(row.time_signature).split(".")[0]}'] = 1
        song_data[f'release_year_{str(row.year).split(".")[0]}'] = 1

        # Label
        label = 1 if row.valence >= 0.5 else 0
        song_data['label'] = label

        # Append the song data to the list
        data.append(song_data)

    # Create DataFrame
    dense_df = pd.DataFrame(data)
    dense_df.fillna(0, inplace=True)

    return dense_df
dense_df=build_dense_dataframe("tracks_features.csv",50000)
dense_df.to_csv("tracks_features_50000.csv")
print(dense_df.columns)