import numpy as np
import pandas as pd
import matrix_factorization_utilities

# Steps so far:
# 1. Create a matrix of known user reviews
# 2. Factor out a U (user attribute) matrix and an M (movie attribute) matrix from the known reviews
# 3. Multiply the U and M matrices we found to get review scores for every user and every movie (U * M = Movie Ratings)

# Load the incomplete user ratings data
raw_dataset_df = pd.read_csv('movie_ratings_data_set.csv')

# Use pandas pivot table function to build the review matrix
ratings_df = pd.pivot_table(raw_dataset_df, index='user_id', columns='movie_id', aggfunc=np.max)

# Apply matrix factorization to find the latent features (U and M)
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.as_matrix(),
																	num_features=15,
																	regularization_amount=0.1)

# Multiply U, M matrices together using matmul() which requires python 3.5
# np.dot() gets the job done for python 2.7
predicted_ratings = np.dot(U, M)

# Save to csv
# Movies all the way on the right for each user is the highest recommended
predicted_ratings_df = pd.DataFrame(index=ratings_df.index, columns=ratings_df.columns,
															data=predicted_ratings)

predicted_ratings_df.to_csv("predicted_ratings.csv")

# Have users fill out surveys to get their actual ratings, replace our estimated values with their scores,
# and recalculate to improve accuracy