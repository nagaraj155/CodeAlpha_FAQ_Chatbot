
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")

st.title("💬 FAQ Chatbot")
st.caption("CodeAlpha AI Internship - Task 2")

FAQS = [
    {
        "question": "How can I register?",
        "answer": "Fill in the registration form and submit the required details."
    },
    {
        "question": "What are your working hours?",
        "answer": "Our support hours are Monday to Friday, 9 AM to 5 PM."
    },
    {
        "question": "How can I contact support?",
        "answer": "Contact support through the contact form or official support email."
    },
    {
        "question": "Do you provide certificates?",
        "answer": "Certificates are provided for eligible programs after completion."
    },
    {
        "question": "How do I reset my password?",
        "answer": "Click Forgot Password on the sign-in page and follow the instructions."
    },
    {
        "question": "Is there a fee to join?",
        "answer": "Fees depend on the program. Check the program page for details."
    },
    {
        "question": "Can I use this on my phone?",
        "answer": "Yes, the web application works on desktop and mobile browsers."
    },
    {
        "question": "How do I submit a project?",
        "answer": "Upload the required project files and complete the official submission form if required."
    },
    {
        "question": "Where can I find more information?",
        "answer": "Check the official website or contact the support team."
    }
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

questions = [clean_text(faq["question"]) for faq in FAQS]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

faq_vectors = vectorizer.fit_transform(questions)

def find_answer(user_question):
    cleaned = clean_text(user_question)

    if not cleaned:
        return "Please type a question.", None, 0

    user_vector = vectorizer.transform([cleaned])

    scores = cosine_similarity(
        user_vector, faq_vectors
    ).flatten()

    index = scores.argmax()
    score = float(scores[index])

    if score < 0.16:
        return (
            "Sorry, I couldn't find a matching FAQ. "
            "Try asking about registration, support, "
            "certificates, fees, or project submission.",
            None,
            score
        )

    return (
        FAQS[index]["answer"],
        FAQS[index]["question"],
        score
    )

with st.expander("Example questions"):
    for faq in FAQS:
        st.write("•", faq["question"])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me about registration, support, "
                       "certificates, fees, or project submission."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Type your question...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    answer, matched, score = find_answer(question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        st.markdown(answer)

        if matched:
            st.caption(
                f"Matched FAQ: {matched} | Similarity: {score:.2f}"
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

if st.button("Clear conversation"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Conversation cleared. Ask another question!"
        }
    ]
    st.rerun()

st.divider()
st.caption("Educational NLP project. Answers use a sample FAQ dataset.")
