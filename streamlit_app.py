import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Streamlit app layout
st.title("Text Summarization with BART")

st.write("Paste your text below to get a summary:")

# Text input from user
text = st.text_area("Enter the text you want to summarize", height=200)

if st.button("Summarize"):
    if text:
        # Tokenize and summarize the input text
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.write("Summary:")
        st.write(summary)
    else:
        st.write("Please enter some text to summarize.")
