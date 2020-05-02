from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text = ["London Paris London", "Paris Paris London"]

vectorizer = CountVectorizer()
vector_matrix = vectorizer.fit_transform(text).toarray()
print(vectorizer.get_feature_names())
print(vectorizer.fit_transform(text))
print(cosine_similarity(vector_matrix))