import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# collaborative filtering test using test dataset
# ratings = pd.read_csv("dataset/toy_dataset.csv", index_col=0)
# ratings = ratings.fillna(0)
# def standardize(row): # bringing the mean to 0 and wirh a range of 1 for the ratings (to prevent too harsh or too lenient ratings
#     new_row = (row-row.mean())/(row.max()-row.min())
#     return new_row
#
#
# ratings_std = ratings.apply(standardize)
# item_similarity = cosine_similarity(ratings_std.T)
# item_similarity_df = pd.DataFrame(item_similarity,index=ratings.columns, columns=ratings.columns)
#
# def get_similar_movies(movie_name, user_rating):
#     similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
#     similar_score = similar_score.sort_values(ascending=False)
#     return similar_score
#
# action_lover = [("action1",5),("romantic1",1),("romantic3",1)]
# similar_movies= pd.DataFrame()
# for movie, rating in action_lover:
#     similar_movies = similar_movies.append(get_similar_movies(movie,rating), ignore_index=True)
# print(similar_movies.head())
# print(similar_movies.sum().sort_values(ascending=False))


# using actual movie and rating dataset to build recommender
ratings = pd.read_csv("dataset/ratings.csv")
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.merge(movies,ratings).drop(['genres', 'timestamp'], axis=1)
user_ratings = ratings.pivot_table(index=['userId'], columns=['title'], values='rating')

# need to drop movies with <10 users who have rated them - 9000+ movies but only around 600+ users
# there are some movies with very few users who have rated them, might create noise in the dataset
user_ratings = user_ratings.dropna(thresh=10, axis=1).fillna(0)

# building similarity matrix
item_similarity_df = user_ratings.corr(method='pearson')
print(item_similarity_df.head)
def get_similar_movies(movie_name, user_rating):
    similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

action_lover = [("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)",5),
                ("12 Years a Slave (2013)",4),
                ("2012 (2009)",3),
                ("(500) Days of Summer (2009)",2)]
similar_movies= pd.DataFrame()
for movie, rating in action_lover:
    similar_movies = similar_movies.append(get_similar_movies(movie,rating), ignore_index=True)

print(similar_movies.sum().sort_values(ascending=False))