from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample data (replace with actual dataset)
data = [
    "happy",
    "Good",
    "Outstanding",  # Excellent and remarkable
    "Brilliant",  # Exceptionally clever or talented
    "Remarkable",  # Worthy of attention
    "Spectacular",  # Beautiful in a dramatic way
    "Delightful",
    "Marvelous",  # Extraordinary, causing great wonder
    "Incredible",  # Difficult to believe, extraordinary
    "Amazing",  # Causing great surprise or wonder
    "Impressive",  # Making a strong positive impact
    "Fantastic",  # Extremely good
    "Remarkable",  # Worthy of attention
    "Bad",  # Very pleasing, bringing joy
    "Terrible",  # Very bad or unpleasant
    "Awful",  # Extremely bad or unpleasant
    "Disappointing",  # Failing to meet expectations
    "Horrible",  # Extremely unpleasant or dreadful
    "Mediocre",  # Of average or low quality
    "Difficult",
    "Horrible",  # Extremely unpleasant or dreadful
    "Poor",  # Of low quality or condition
    "Unpleasant",  # Not enjoyable or agreeable
    "Hopeless",  # Without hope, unable to improve
    "Painful",  # Causing physical or emotional pain
    "Unworthy",
    "unhappy",
    "Unsatisfactory",  # Not good enough to meet expectations
    
]
labels = [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # 1=Positive, 0=Negative

# Text vectorization
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(data)

# Model training
model = LogisticRegression()
model.fit(X_vect, labels)

# Save the model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
