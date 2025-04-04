#import essential library
import pandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



# Reading data
fake_news = pd.read_csv('Fake.csv')
true_news = pd.read_csv('True.csv')
#print (fake_news.head(10))
#print (true_news.head(10))

#Getting information
#fake_news.info()
# true_news.info()

# randomly dropping rows from fake news dataset
rows_to_remove = len(fake_news) - len(true_news)
fake_news_reduced = fake_news.sample(n=len(fake_news)- rows_to_remove, random_state=42)
print(fake_news_reduced)

# Labelling fake news and true news as 0 and 1 respectively
fake_news['label'] = 0
true_news['label'] = 1

# Merging two datasets using concat
final_news = pd.concat([fake_news, true_news], axis = 0)
#print(final_news.columns)

# Dropping columns
df = final_news.drop(columns=['title', 'subject', 'date'])
#print(df)

# Finding null/nan value in the dataset
null_value_count = df.isnull().sum().sum()
#print(null_value_count)

# doing random shuffling to prevent model bias
df = df.sample(frac=1)
#print(df)
#print(df.columns)

# # converting text to number using CountVectorizer
vectorizer = CountVectorizer()
vectorized_text = vectorizer.fit_transform(df['text'])
#print(vectorized_text)

# Applying X and Y value
x = vectorized_text
y = df['label']

# Train test and split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# Supervised Model - Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
#print(y_predict)

# Finding accuracy
print("Accuracy:", accuracy_score(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))


# Applying unsupervised model - K-Means Clustering

#Merging datasets again to restore the dropped column
data = pd.concat([true_news, fake_news], ignore_index=True)
print(data)

# Feature extraction of title and subject
data['combined_text'] = data['title'] + " " + data['subject']

# Use Count Vectorization on the 'combined_text' column
vectorizer = CountVectorizer(stop_words='english', max_features=500)
title_vectors = vectorizer.fit_transform(data['combined_text'])

# Reduce dimensionality using PCA to improve clustering
pca = PCA(n_components=2)  # Reduce to 2 dimensions for better clustering
reduced_features = pca.fit_transform(title_vectors.toarray())


# Apply K-Means Clustering with K=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data['cluster_kmeans'] = kmeans.fit_predict(reduced_features)

# Print cluster distribution
print("Cluster Distribution:")
print(data['cluster_kmeans'].value_counts())

# Visualization of clusters
plt.figure(figsize=(8,6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=data['cluster_kmeans'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Visualization of News Clusters (K=5)")
plt.show()

