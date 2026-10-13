
# CodeAlpha FAQ Chatbot

## Overview
A Python chatbot that uses NLP techniques to match
user questions with predefined FAQ answers.

## Technologies
- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

## Features
- Interactive chat interface
- Text preprocessing
- FAQ matching
- Similarity score display
- Fallback response
- Clear conversation option

## How to Run
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Run `pip install -r requirements.txt`.
4. Run `python -m streamlit run app.py`.
5. Open the local URL shown in the terminal.

## How It Works
The chatbot cleans the text, converts FAQ questions
into TF-IDF vectors, and uses cosine similarity
to find the closest matching question.

## Limitations
This is a retrieval-based chatbot, not a generative
AI model. Its accuracy depends on the FAQ dataset
and the wording of the user's question.

## Author
Add your name here.
