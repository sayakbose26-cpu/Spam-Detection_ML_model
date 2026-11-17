import pandas as pd 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import joblib 
from sklearn.model_selection import train_test_split


df = pd.read_csv("dataset/emails.csv")
X = df["text"]
y = df["spam"]
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])
model.fit(X,y)
print("model successfully train")
joblib.dump(model,"spam_dection.pk1")