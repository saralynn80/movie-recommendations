import numpy as np
import pandas as pd
import matrix_factorization_utilities

# Load user ratings from both the training and testing csv files
raw_training_dataset_df = pd.read_csv('movie_ratings_data_set_training.csv')
raw_testing_dataset_df = pd.read_csv('movie_ratings_data_set_testing.csv')

# Convert the running list of user ratings into a matrix
ratings_training_df = pd.pivot_table(raw_training_dataset_df, index='user_id', columns='movie_id', aggfunc=np.max)
ratings_testing_df = pd.pivot_table(raw_testing_dataset_df, index='user_id', columns='movie_id', aggfunc=np.max)

# Apply matrix factorization on only the training data
# Regularization amount sets how much weight wil be placed on a single attribute during matrix factorization
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_training_df.as_matrix(),
                                                                    num_features=11,
                                                                    regularization_amount=1.1)

# Find all predicted ratings by multiplying U and M
predicted_ratings = np.dot(U, M)

# Calculate the error rates by calculating RMSE
rmse_training = matrix_factorization_utilities.RMSE(ratings_training_df.as_matrix(), predicted_ratings)
rmse_testing = matrix_factorization_utilities.RMSE(ratings_testing_df.as_matrix(), predicted_ratings)

print("Training RMSE: {}".format(rmse_training))
print("Testing RMSE: {}".format(rmse_testing))
