import streamlit as st
import numpy as np
import joblib
import re
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Define max_sequence_length (same as what was used during training)
max_sequence_length = 186

# Define bad_words list
bad_words = ["মাদারচুদ", "বাইন চুদ", "জুকার", "ফালতু", "শালা", "লেংটা", "বাটপার", "মাদারচুদ", "ফালতু", "শালা", "নাস্তিকের বাচ্চা", "শুয়ার", "কুত্তা", "পুটকি", "নগ্নতায়", "সমকামি", "চুদছে", "চুদতে", "চুদা", "আবাল চোদা", "শুয়োরের বাচ্চা", "কুত্তার বাচ্চা", "হারামির বাচ্চা", "হারামজাদা", "শালার পো", "চুতমারানি", "চুদির ভাই","হাউয়ার নাতি"]

# Load the model
loaded_model = load_model('cyberbullying_model.h5')

# Load the tokenizer and label encoder
loaded_tokenizer = joblib.load('tokenizer.pkl')
loaded_label_encoder = joblib.load('label_encoder.pkl')

# Function to filter out bad words from text and return filtered words
def filter_bad_words_with_model(text):
    filtered_words = []
    for bad_word in bad_words:
        # Create a regular expression pattern for the bad word, ignoring case
        pattern = re.compile(re.escape(bad_word), re.IGNORECASE)
        # Replace occurrences of the bad word with asterisks (*) of the same length
        text, num_replacements = pattern.subn('*' * len(bad_word), text)
        if num_replacements > 0:
            filtered_words.append(bad_word)
    return text, filtered_words

# Streamlit UI
st.title("Cyberbullying Detection App (Bangla)")

# Input text for prediction
input_text = st.text_area("Enter Text:")

if st.button("Predict"):
    if input_text:
        # Filter out bad words using the loaded model
        filtered_text, filtered_bad_words = filter_bad_words_with_model(input_text)

        # Tokenize and pad the filtered input text
        input_sequence = loaded_tokenizer.texts_to_sequences([filtered_text])
        input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length)

        # Make a prediction using the loaded model
        predicted_probabilities = loaded_model.predict(input_sequence)

        # Get the class with the highest probability as the predicted class
        predicted_class = np.argmax(predicted_probabilities)

        # Decode the predicted class back to the original label using the loaded label encoder
        predicted_label = loaded_label_encoder.classes_[predicted_class]

        st.subheader("Prediction Result:")
        if predicted_label == "not bully":
            st.write("Prediction: Not Cyberbullying")
            st.write("No bad words found.")
        else:
            st.write("Prediction: Cyberbullying")
            st.write(f"Cyberbullying Type: {predicted_label}")  # Print the filter type
            st.write(f"Bad Words: {', '.join(filtered_bad_words)}")
        st.write("Filtered Text:")
        st.write(f"<span style='color:red; font-weight:bold'>{filtered_text}</span>", unsafe_allow_html=True)
    else:
        st.warning("Please enter text for prediction.")
