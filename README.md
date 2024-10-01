# 🎵 Music Mood Classifier

This project classifies songs based on their mood using a dataset from Spotify, which contains over 1 million songs. The mood of a song is determined by its **valence**, a Spotify metric that measures the musical positiveness conveyed by a track. Songs with a valence greater than 0.5 are classified as having a positive mood, while those with a valence less than 0.5 are classified as having a negative mood.

## Data and Features
The dataset includes various features such as:
- **Float metrics**: danceability, energy, loudness, valence, etc.
- **Binarized metrics**: artist id, explicit, mode, key, etc.

Data preprocessing involved binarizing non-numeric features and normalizing numeric features using z-scores.

## Classification Methods
We explored three classification approaches:
1. **Logistic Regression**: Achieved the best accuracy (0.76) among the classifiers.
2. **Decision Tree**: Provided reasonable accuracy with time savings using parallelism.
3. **Random Forest**: Increased generalization and performance with multiple decision trees.

## Results
The final accuracy of each classifier on a subset of 50,000 songs:
- Logistic Regression: **76.03%**
- Decision Tree: **74.41%**
- Random Forest: **72%**

## Future Work
Future improvements include using the entire dataset, extracting additional features from the Spotify API, and fine-tuning the model parameters.

---

This project was developed as part of the EECE5645 Parallel Processing for Data Analytics course.
