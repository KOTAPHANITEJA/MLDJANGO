# views.py
from django.shortcuts import render
import joblib

# Load the model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_sentiment(request):
    prediction = None
    if request.method == 'POST':
        user_input = request.POST.get('input_text')
        if user_input:
            # Debug: Print user input
            print(f"User input: {user_input}")
            
            # Transform input and predict sentiment
            input_vector = vectorizer.transform([user_input])
            
            # Debug: Print vectorized input
            print(f"Vectorized input: {input_vector.toarray()}")
            
            prediction = model.predict(input_vector)[0]
            
            # Debug: Print prediction result
            print(f"Raw prediction: {prediction}")
            
            prediction = "Positive" if prediction == 1 else "Negative"
    
    return render(request, 'predict.html', {'prediction': prediction})
