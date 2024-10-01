import numpy as np
import pandas as pd
from SparseVector import SparseVector


def build_dataset(fname, num_datapoints):
    """
    Build an array of sparsevectors from the dataset
    :param fname: The csv file of the dataset
    :param num_datapoints: The number of datapoints to build the dataset from
    :return: An array of SparseVectors containing the specified number of datapoints
    """
    songs = pd.read_csv(fname)

    data_lists = {'danceability': [],
                  'energy': [],
                  'loudness': [],
                  'speechiness': [],
                  'acousticness': [],
                  'instrumentalness': [],
                  'liveness': [],
                  'tempo': [],
                  'duration_ms': []
                  }
    count = 0
    for _, row in songs.iterrows():
        data_lists['danceability'].append(row.danceability)
        data_lists['energy'].append(row.energy)
        data_lists['loudness'].append(row.loudness)
        data_lists['speechiness'].append(row.speechiness)
        data_lists['acousticness'].append(row.acousticness)
        data_lists['instrumentalness'].append(row.instrumentalness)
        data_lists['liveness'].append(row.liveness)
        data_lists['tempo'].append(row.tempo)
        data_lists['duration_ms'].append(row.duration_ms)
        count += 1
        if count == num_datapoints:
            break

    data_normalization = {'danceability': (np.mean(data_lists['danceability']), np.std(data_lists['danceability'])),
                          'energy': (np.mean(data_lists['energy']), np.std(data_lists['energy'])),
                          'loudness': (np.mean(data_lists['loudness']), np.std(data_lists['loudness'])),
                          'speechiness': (np.mean(data_lists['speechiness']), np.std(data_lists['speechiness'])),
                          'acousticness': (np.mean(data_lists['acousticness']), np.std(data_lists['acousticness'])),
                          'instrumentalness': (
                              np.mean(data_lists['instrumentalness']), np.std(data_lists['instrumentalness'])),
                          'liveness': (np.mean(data_lists['liveness']), np.std(data_lists['liveness'])),
                          'tempo': (np.mean(data_lists['tempo']), np.std(data_lists['tempo'])),
                          'duration_ms': (np.mean(data_lists['duration_ms']), np.std(data_lists['duration_ms']))
                          }
    count = 0
    count_pos = 0
    data = []
    for _, row in songs.iterrows():
        s = SparseVector({})
        artists = eval(row.artist_ids)
        for a in artists:
            s[f'artist_{a}'] = 1

        s['explicit'] = 1 if row.explicit else 0
        s['danceability'] = (row.danceability - data_normalization['danceability'][0]) / \
                            data_normalization['danceability'][1]
        s['energy'] = (row.energy - data_normalization['energy'][0]) / data_normalization['energy'][1]
        s[f'key_{row.key}'] = 1
        s['loudness'] = (row.loudness - data_normalization['loudness'][0]) / data_normalization['loudness'][1]
        if row['mode'] == 1:
            s['major_mode'] = 1
        else:
            s['minor_mode'] = 1
        s['speechiness'] = (row.speechiness - data_normalization['speechiness'][0]) / data_normalization['speechiness'][
            1]
        s['acousticness'] = (row.acousticness - data_normalization['acousticness'][0]) / \
                            data_normalization['acousticness'][1]
        s['instrumentalness'] = (row.instrumentalness - data_normalization['instrumentalness'][0]) / \
                                data_normalization['instrumentalness'][1]
        s['liveness'] = (row.liveness - data_normalization['liveness'][0]) / data_normalization['liveness'][1]
        label = 1 if row.valence >= 0.5 else -1
        s['tempo'] = (row.tempo - data_normalization['tempo'][0]) / data_normalization['tempo'][1]
        s['duration_ms'] = (row.duration_ms - data_normalization['duration_ms'][0]) / data_normalization['duration_ms'][
            1]
        s[f'time_signature_{row.time_signature}'] = 1
        s[f'release_year_{row.year}'] = 1
        count += 1
        data.append((s, label))
        count_pos = count_pos + 1 if label == 1 else count_pos
        if count == num_datapoints:
            break

    print(f'Number of datapoints in positive class: {count_pos}')
    print(f'Number of datapoints in negative class: {num_datapoints - count_pos}')
    return data
