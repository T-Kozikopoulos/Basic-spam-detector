import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# Read the data into pandas dataframes.
df1 = pd.read_csv('Youtube_comments01.csv')
df2 = pd.read_csv('Youtube_comments02.csv')
df3 = pd.read_csv('Youtube_comments03.csv')
df4 = pd.read_csv('Youtube_comments04.csv')
df5 = pd.read_csv('Youtube_comments05.csv')

# Add all the smaller DFs into one main DF, the more data the better.
dfs = [df1, df2, df3, df4, df5]
df = pd.concat(dfs)

# Drop the useless columns.
del df['COMMENT_ID'], df['AUTHOR'], df['DATE']

# Aas per usual, assign X,y to features and columns respectively. A comment is spam when CLASS = 1.
X = df['CONTENT']
y = df['CLASS']

# Split dataset into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

# Use a count vectorizer to keep count of all the words used.
# This is what the classifier will be using to distinguish between ham and spam.
cv = TfidfVectorizer(min_df=1, stop_words='english')

# Fit the count vectorizer, each word counts as a feature.
X_traincv = cv.fit_transform(X_train)
X_traincv = X_traincv.toarray()
cv.inverse_transform(X_traincv[0])

X_testcv = cv.transform(X_test)
X_testcv = X_testcv.toarray()
cv.inverse_transform(X_testcv[0])

# Train the classifier.
clf = MultinomialNB()
clf.fit(X_traincv, y_train)
pred = clf.predict(X_testcv)

# See how well it performed.
accuracy = clf.score(X_traincv, y_train)
print(accuracy)
