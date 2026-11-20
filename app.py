# ============================================================
#   MINI PROJECT: SMART FAQ ASSISTANT (Mini RAG Bot) - Streamlit
# ============================================================
# Idea:
#   1. Search a small FAQ list for the questions closest to what
#      the user typed (sentence-transformers + cosine similarity)
#   2. Send the best FAQ answers to an LLM (OpenAI) as "context"
#   3. The LLM writes a friendly answer using only that context
#
# If there is no OpenAI API key, the app still works:
# it just shows the best FAQ answer directly (no LLM step).
#
# Run locally:   pip install streamlit sentence-transformers openai scikit-learn
#                streamlit run app.py
# ============================================================

import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# STEP 1: Our FAQ data (the "knowledge base")
# ------------------------------------------------------------
faq_data = [
    {"question": "How can I get a refund?",
     "answer": "You can request a refund within 7 days of purchase from the Orders page."},
    {"question": "How long does delivery take?",
     "answer": "Delivery usually takes 3 to 5 working days."},
    {"question": "How do I reset my password?",
     "answer": "Click 'Forgot Password' on the login page and follow the email link."},
    {"question": "Can I cancel my order?",
     "answer": "Yes, orders can be cancelled before they are shipped."},
    {"question": "What payment methods do you accept?",
     "answer": "We accept UPI, debit cards, credit cards and net banking."},
    {"question": "How do I contact customer support?",
     "answer": "Email support@example.com or call 1800-123-456 (9 AM - 6 PM)."},
]

faq_questions = [item["question"] for item in faq_data]


# ------------------------------------------------------------
# STEP 2: Load the model and make FAQ embeddings
# ------------------------------------------------------------
# @st.cache_resource keeps the model in memory, so Streamlit
# does not reload it every time the user clicks something
@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    faq_embeddings = model.encode(faq_questions, normalize_embeddings=True)
    return model, faq_embeddings

model, faq_embeddings = load_model_and_embeddings()


# ------------------------------------------------------------
# STEP 3: Search function (same logic as my FAQ bot)
# ------------------------------------------------------------
def search_faq(query, top_k=3):
    # Convert the user's question into an embedding
    query_embedding = model.encode([query], normalize_embeddings=True)

    # Compare it with every FAQ question
    scores = cosine_similarity(query_embedding, faq_embeddings)[0]

    # Sort from highest to lowest score and keep the top_k
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "question": faq_questions[i],
            "answer": faq_data[i]["answer"],
            "score": float(scores[i])
        }
        for i in top_indices
    ]


# ------------------------------------------------------------
# STEP 4: Set up the LLM (same logic as my LLM chatbot)
# ------------------------------------------------------------
client = None
if os.getenv("OPENAI_API_KEY"):
    from openai import OpenAI
    client = OpenAI()


def ask_llm(prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return response.output_text


# ------------------------------------------------------------
# STEP 5: Combine search + LLM to build the final answer
# ------------------------------------------------------------
def get_answer(user_question):
    results = search_faq(user_question, top_k=3)
    best = results[0]

    # Weak match -> question is probably not in our FAQ
    if best["score"] < 0.4:
        return "Sorry, I couldn't find this in our FAQ. Please contact support.", results

    # No API key -> return the best FAQ answer directly
    if client is None:
        return best["answer"], results

    # With API key -> give the FAQs to the LLM as context
    context = "\n".join(f"Q: {r['question']}\nA: {r['answer']}" for r in results)

    prompt = f"""You are a helpful customer support assistant.
Answer the user's question using ONLY the FAQ information below.
If the answer is not there, say you don't know.

FAQ information:
{context}

User question: {user_question}
"""
    return ask_llm(prompt), results


# ------------------------------------------------------------
# STEP 6: Streamlit user interface
# ------------------------------------------------------------
st.set_page_config(page_title="Smart FAQ Assistant", page_icon="🤖")

st.title("🤖 Smart FAQ Assistant")
st.caption("Semantic search + LLM (a mini RAG project)")

# Show which mode the app is running in
if client is None:
    st.info("No API key found - running in FAQ-only mode.")
else:
    st.success("LLM mode ON - answers are written by GPT.")

# Sidebar: show the knowledge base
with st.sidebar:
    st.header("📚 FAQ Knowledge Base")
    for item in faq_data:
        st.markdown(f"- {item['question']}")

# Keep chat history between reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box at the bottom
user_question = st.chat_input("Ask a question...")

if user_question:
    # Show and save the user's message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            answer, matches = get_answer(user_question)
            st.markdown(answer)

            # Expandable box showing the matched FAQs and scores
            with st.expander("🔍 Top matching FAQs"):
                for m in matches:
                    st.write(f"**{m['question']}** — score: {m['score']:.2f}")
                    st.progress(max(0.0, min(1.0, m["score"])))
        except Exception as e:
            answer = f"Error: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
