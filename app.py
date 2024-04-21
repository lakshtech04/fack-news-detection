from flask import Flask, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Load stopwords for preprocessing
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        preprocessed_title = preprocess_text(title)
        tfidf_vector = tfidf.transform([preprocessed_title])
        prediction = model.predict(tfidf_vector)[0]
        prediction_result = 'True News' if prediction == 1 else 'Fake News'
        return render_template('index.html', prediction_result=prediction_result)

def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text.lower())
    # Removing stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    preprocessed_text = " ".join(filtered_tokens)
    return preprocessed_text

if __name__ == '__main__':
    app.run(debug=True)
