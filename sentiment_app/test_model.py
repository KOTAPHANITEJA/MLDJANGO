# test_model.py
import joblib

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Test positive and negative sentences
test_sentences = [
    "I absolutely loved this product, it was amazing!",  # Expected: Positive
    "This is the worst experience I've ever had.",       # Expected: Negative
    "Fantastic service and quality!",                    # Expected: Positive
    "I am extremely disappointed.",                      # Expected: Negative
]

# Transform and predict
test_vectors = vectorizer.transform(test_sentences)
predictions = model.predict(test_vectors)

# Print results
for sentence, prediction in zip(test_sentences, predictions):
    print(f"Text: {sentence} => Prediction: {'Positive' if prediction == 1 else 'Negative'}")
