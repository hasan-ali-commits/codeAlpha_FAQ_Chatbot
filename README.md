# 🤖 Smart FAQ Assistant

**CodeAlpha AI Internship – Task 2: Chatbot for FAQs**

A chatbot that answers customer questions by finding the most similar FAQ using **semantic search** and **cosine similarity**, shown in a simple **Streamlit** chat interface.
It can also use an LLM (OpenAI) to write friendlier answers. This is a mini **RAG** (Retrieval-Augmented Generation) setup.

## Features
- FAQ knowledge base (refunds, delivery, payments, passwords, support)
- Understands meaning, not just keywords. For example, *"Can I get my money back?"* matches *"How can I get a refund?"*
- Shows the **top 3 matching FAQs** with similarity scores
- Replies "not found" if the best score is too low (below 0.4)
- **Optional LLM mode:** with an OpenAI API key, GPT writes the answer using only the matched FAQs
- Works **without an API key** (FAQ-only mode)
- Chat history kept on screen

## How It Works
1. **Collect FAQs:** stored as question–answer pairs
2. **Embed:** each FAQ question is turned into a vector using `all-MiniLM-L6-v2`
3. **Match:** the user's question is embedded and compared with all FAQs using **cosine similarity**
4. **Respond:** the best matching answer is shown (or sent to the LLM as context)

## Tech Stack
| Tool | Purpose |
|---|---|
| Python | Main language |
| Sentence-Transformers | Text embeddings |
| scikit-learn | Cosine similarity |
| NumPy | Sorting top scores |
| Streamlit | Chat web interface |
| OpenAI API (optional) | Writing natural answers |

## How to Run
```bash
git clone https://github.com/<your-username>/CodeAlpha_FAQ_Chatbot.git
cd CodeAlpha_FAQ_Chatbot
python -m venv venv
venv\Scripts\activate        # on Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Optional LLM mode:** set your key before running:
```bash
set OPENAI_API_KEY=your_key_here     # on Mac/Linux: export OPENAI_API_KEY=your_key_here
```

## Note
The first run downloads the embedding model (~90 MB), so it takes a minute.
