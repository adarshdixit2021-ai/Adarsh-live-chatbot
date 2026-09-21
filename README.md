# 🤖 Aether-AD27

### Your Personal AI Assistant

Aether-AD27 is a live AI-powered chatbot built as a personal learning project by **Adarsh Dixit**, a 2nd-year BCA student specializing in **Data Science & Artificial Intelligence**.

The project combines conversational AI, image analysis, persistent chat history, RAG-based information retrieval, user feedback, and cloud database integration into a single Streamlit application.

---

## 🚀 Live Demo

👉 **Try Aether-AD27:**  
https://adarsh-live-chatbot-a42krd5hyuawywj2rev992.streamlit.app/

---

## 📸 About the Project

Aether-AD27 started with the basic chatbot concepts I learned during an **IBM Chatbot Creation Workshop at Babu Banarasi Das University (BBDU)** on **14th September 2026**.

After the workshop, I continued developing and improving the project independently by researching concepts, writing code, debugging errors, testing features, and deploying the application.

The project was developed using **Python and Streamlit without HTML, CSS, or JavaScript**.

---

## ✨ Features

### 💬 AI Chat

- Conversational AI powered by Groq
- Beginner-friendly response mode
- Programming and technical question support
- Context-aware conversation
- Current conversation memory

### 🖼️ Image Analysis

- Upload images directly in the chat
- Ask questions about uploaded images
- AI-powered image understanding
- Supports multiple images within the configured limits

### 🧠 RAG-Based Information Retrieval

Aether-AD27 includes a lightweight Retrieval-Augmented Generation (RAG) system.

It retrieves relevant information from trusted sources before generating an answer.

The application includes focused retrieval for:

- BBDU-related information
- Programming concepts
- Technical topics

For BBDU-related questions, the application prioritizes information retrieved from available official BBDU sources.

> Note: The BBDU functionality is retrieval-based. The BBDU website content is not used to train or fine-tune an AI model.

### 👤 User Accounts

Users can create an account using:

- Name
- Date of Birth
- Gender

User accounts allow conversations to be associated with the user's profile.

### 💾 Persistent Chat History

Logged-in users can:

- Save conversations
- Create new chats
- Restore previous conversations
- Continue earlier conversations
- Store chat messages in MySQL

### 📝 Feedback System

Users can submit feedback directly from the application.

Feedback is stored using **Google Sheets** for easy collection and review.

### 🔍 Current Information Search

The application includes an optional current-information search feature for questions that require fresh information.

### 🎓 Beginner-Friendly Mode

Aether-AD27 can provide explanations in a beginner-friendly format with:

- Simple explanations
- Examples
- Step-by-step guidance
- Technical concepts explained in easier language

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web application framework |
| Groq API | AI-powered responses |
| MySQL | User accounts and chat history |
| Aiven | Cloud MySQL hosting |
| Google Sheets | User feedback storage |
| RAG | Information retrieval |
| scikit-learn | Lightweight text retrieval |
| BeautifulSoup | Web content extraction |
| Requests | Web requests |
| python-dotenv | Environment variable management |
| Git & GitHub | Version control |
| Streamlit Cloud | Application deployment |

---

# 🏗️ High-Level Architecture

```text
                    ┌──────────────────────┐
                    │      User            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    │    Aether-AD27       │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
         ┌────────────┐ ┌────────────┐ ┌──────────────┐
         │ Groq API   │ │ RAG Engine │ │ Image Input  │
         └────────────┘ └─────┬──────┘ └──────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Trusted Information  │
                    │ Sources              │
                    └──────────────────────┘

                               │
                               ▼
                    ┌──────────────────────┐
                    │      MySQL           │
                    │                      │
                    │ Users                │
                    │ Chats                │
                    │ Messages             │
                    └──────────────────────┘

                               │
                               ▼
                    ┌──────────────────────┐
                    │    Google Sheets     │
                    │      Feedback        │
                    └──────────────────────┘
