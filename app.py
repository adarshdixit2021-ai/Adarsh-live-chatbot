import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import gspread
import streamlit as st
from dotenv import load_dotenv
from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Adarsh AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# ENVIRONMENT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
google_sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
google_credentials_file = os.getenv(
    "GOOGLE_CREDENTIALS_FILE", ""
).strip()


def get_credentials_path():
    """Return an absolute path for the Google service-account JSON."""
    if not google_credentials_file:
        return None

    path = Path(google_credentials_file).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


if not groq_api_key:
    st.error("❌ GROQ_API_KEY is missing. Check your .env file.")
    st.stop()


client = Groq(api_key=groq_api_key)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "messages": [],
    "chat_history": [],
    "user_profile": None,
    "show_login": False,
    "beginner_mode": True,
    "web_search": True,
    "feedback_message": "",
    "feedback_rating": 5,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# FEEDBACK WINDOW
# =========================================================

FEEDBACK_START_DATE = date(2026, 9, 15)
FEEDBACK_END_DATE = FEEDBACK_START_DATE + timedelta(days=10)
feedback_available = date.today() <= FEEDBACK_END_DATE


# =========================================================
# GOOGLE SHEETS
# =========================================================

FEEDBACK_HEADERS = [
    "Date & Time",
    "Username",
    "Feedback",
    "Rating",
    "Profession",
    "Status",
]


def get_feedback_sheet():
    """
    Connect to the first worksheet and return (worksheet, error).
    No cached connection is used so credential/sheet changes take
    effect immediately after restarting the app.
    """
    if not google_sheet_id:
        return None, "GOOGLE_SHEET_ID is missing in .env."

    credentials_path = get_credentials_path()

    if credentials_path is None:
        return None, "GOOGLE_CREDENTIALS_FILE is missing in .env."

    if not credentials_path.is_file():
        return None, f"Credential file not found: {credentials_path}"

    try:
        google_client = gspread.service_account(
            filename=str(credentials_path)
        )

        spreadsheet = google_client.open_by_key(google_sheet_id)
        worksheet = spreadsheet.sheet1

        # Make sure the dedicated feedback sheet has the expected columns.
        current_headers = worksheet.row_values(1)

        if current_headers != FEEDBACK_HEADERS:
            worksheet.update(
                "A1:F1",
                [FEEDBACK_HEADERS],
                value_input_option="USER_ENTERED",
            )

        return worksheet, None

    except gspread.exceptions.SpreadsheetNotFound:
        return None, (
            "Google Sheet was not found or the service account "
            "does not have access. Share the Sheet with the "
            "service-account email as Editor."
        )

    except gspread.exceptions.APIError as error:
        return None, f"Google Sheets API error: {error}"

    except Exception as error:
        return None, str(error)


def save_feedback(feedback_text, rating):
    """Append feedback to Google Sheets."""
    worksheet, error = get_feedback_sheet()

    if worksheet is None:
        return False, error

    profile = st.session_state.user_profile

    if profile:
        user_name = profile.get("name", "Guest")
        profession = profile.get("profession", "Not provided")
    else:
        user_name = "Guest"
        profession = "Not provided"

    india_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    formatted_time = india_time.strftime("%d %B %Y, %I:%M:%S %p")

    try:
        worksheet.append_row(
            [
                formatted_time,
                user_name,
                feedback_text.strip(),
                int(rating),
                profession,
                "New",
            ],
            value_input_option="USER_ENTERED",
        )
        return True, None

    except gspread.exceptions.APIError as error:
        return False, f"Google Sheets API error: {error}"

    except Exception as error:
        return False, str(error)


# =========================================================
# CHAT HISTORY
# =========================================================

def save_current_chat():
    if not st.session_state.messages:
        return

    first_question = next(
        (
            message["content"]
            for message in st.session_state.messages
            if message.get("role") == "user"
        ),
        "",
    ).strip()

    if not first_question:
        return

    chat_copy = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages
    ]

    # Do not save the exact same current chat repeatedly.
    if st.session_state.chat_history:
        last_chat = st.session_state.chat_history[-1]
        if last_chat.get("messages") == chat_copy:
            return

    st.session_state.chat_history.append(
        {
            "title": first_question[:45],
            "messages": chat_copy,
        }
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    if st.session_state.user_profile:
        user_name = st.session_state.user_profile["name"]
        st.markdown(f"# 👤 {user_name}")
    else:
        st.markdown("# 🤖 Adarsh AI")

    st.caption("Your Personal AI Assistant")
    st.divider()

    # -----------------------------------------------------
    # LOGIN / PROFILE
    # -----------------------------------------------------

    if st.session_state.user_profile is None:

        if st.button("👤 User Login", use_container_width=True):
            st.session_state.show_login = (
                not st.session_state.show_login
            )
            st.rerun()

        if st.session_state.show_login:
            st.markdown("### 👤 Create Profile")
            st.caption("Enter your basic information.")

            login_name = st.text_input(
                "Name",
                placeholder="Enter your name",
                key="login_name",
            )

            login_dob = st.date_input(
                "Date of Birth",
                value=date(2000, 1, 1),
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                key="login_dob",
            )

            login_gender = st.selectbox(
                "Gender",
                [
                    "Select Gender",
                    "Male",
                    "Female",
                    "Other",
                    "Prefer not to say",
                ],
                key="login_gender",
            )

            login_profession = st.text_input(
                "Profession",
                placeholder="e.g. Student, Developer",
                key="login_profession",
            )

            if st.button("✅ Submit", use_container_width=True):
                if not login_name.strip():
                    st.warning("⚠️ Please enter your name.")
                elif login_gender == "Select Gender":
                    st.warning("⚠️ Please select your gender.")
                elif not login_profession.strip():
                    st.warning("⚠️ Please enter your profession.")
                else:
                    st.session_state.user_profile = {
                        "name": login_name.strip(),
                        "dob": login_dob.strftime("%d %B %Y"),
                        "gender": login_gender,
                        "profession": login_profession.strip(),
                    }
                    st.session_state.show_login = False
                    st.toast("✅ Welcome to Adarsh AI!")
                    st.rerun()

    else:

        if st.button("✏️ Edit Profile", use_container_width=True):
            profile = st.session_state.user_profile

            st.session_state.edit_name = profile.get("name", "")
            st.session_state.edit_dob = datetime.strptime(
                profile.get("dob", "01 January 2000"),
                "%d %B %Y",
            ).date()
            st.session_state.edit_gender = profile.get(
                "gender", "Prefer not to say"
            )
            st.session_state.edit_profession = profile.get(
                "profession", ""
            )
            st.session_state.show_login = True
            st.session_state.user_profile = None
            st.rerun()

    st.divider()

    # -----------------------------------------------------
    # CHAT CONTROLS
    # -----------------------------------------------------

    if st.button("➕ New Chat", use_container_width=True):
        save_current_chat()
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # -----------------------------------------------------
    # CHAT HISTORY
    # -----------------------------------------------------

    st.markdown("### 💬 Chat History")

    if not st.session_state.chat_history:
        st.caption("No previous chats yet.")
    else:
        for index, chat in enumerate(
            reversed(st.session_state.chat_history)
        ):
            if st.button(
                f"💬 {chat['title']}",
                key=f"history_{index}",
                use_container_width=True,
            ):
                st.session_state.messages = [
                    message.copy()
                    for message in chat["messages"]
                ]
                st.rerun()

    st.divider()

    # -----------------------------------------------------
    # HELP & FEEDBACK
    # -----------------------------------------------------

    if feedback_available:
        with st.expander("📣 Help & Feedback", expanded=False):
            st.markdown("### 💬 Help improve Adarsh AI")
            st.caption(
                "Found a problem or have an idea? Tell us what you think."
            )

            with st.form("feedback_form", clear_on_submit=True):
                feedback_text = st.text_area(
                    "Your Feedback",
                    placeholder=(
                        "Describe a bug, problem, suggestion "
                        "or anything you would like to improve..."
                    ),
                    height=120,
                )

                feedback_rating = st.slider(
                    "⭐ Rate Adarsh AI",
                    min_value=1,
                    max_value=5,
                    value=5,
                )

                submit_feedback = st.form_submit_button(
                    "📤 Submit Feedback",
                    use_container_width=True,
                )

            if submit_feedback:
                if not feedback_text.strip():
                    st.warning("⚠️ Please enter your feedback.")
                else:
                    with st.spinner("📤 Saving your feedback..."):
                        saved, error = save_feedback(
                            feedback_text,
                            feedback_rating,
                        )

                    if saved:
                        st.success(
                            "✅ Thank you! Your feedback has been submitted."
                        )
                    else:
                        st.error("❌ Feedback could not be saved.")
                        with st.expander("🔧 See the exact reason"):
                            st.code(error or "Unknown Google Sheets error")

    # -----------------------------------------------------
    # SETTINGS
    # -----------------------------------------------------

    with st.expander("⚙️ Settings", expanded=False):
        st.markdown("#### 🎛️ Chat Settings")

        st.session_state.beginner_mode = st.toggle(
            "👶 Beginner Mode",
            value=st.session_state.beginner_mode,
        )

        st.session_state.web_search = st.toggle(
            "🌐 Web Search",
            value=st.session_state.web_search,
        )

        st.divider()

        st.caption(
            "👶 Simple explanations are enabled."
            if st.session_state.beginner_mode
            else "🎓 Advanced explanations are enabled."
        )

        st.caption(
            "🌐 Current information search is enabled."
            if st.session_state.web_search
            else "⚡ Fast AI mode is enabled. Web Search is off."
        )

    st.divider()

    st.caption("👨‍💻 Developer")
    st.caption("Adarsh Dixit")
    st.caption("BCA • Data Science & AI")


# =========================================================
# MAIN SCREEN
# =========================================================

if st.session_state.user_profile:
    profile = st.session_state.user_profile

    st.markdown(f"# 🤖 Welcome, {profile['name']}!")
    st.markdown("### Your Personal AI Assistant")
    st.caption("💡 Ask questions • Learn • Explore • Get answers")
    st.divider()

    if not st.session_state.messages:
        st.markdown("### 👤 Your Profile")

        col1, col2 = st.columns(2)

        with col1:
            st.info(
                f"**👤 Name**\n\n{profile['name']}\n\n"
                f"**🎂 Date of Birth**\n\n{profile['dob']}"
            )

        with col2:
            st.info(
                f"**⚧️ Gender**\n\n{profile['gender']}\n\n"
                f"**💼 Profession**\n\n{profile['profession']}"
            )

        st.divider()
        st.markdown("### ✨ What can I help you with today?")
        st.write(
            "Ask me about programming, technology, Artificial Intelligence, "
            "Data Science, education, current information, or general topics."
        )

        card1, card2, card3 = st.columns(3)

        with card1:
            st.info("🤖 **AI Assistant**\n\nAsk questions and get clear answers.")

        with card2:
            st.info("📚 **Learning**\n\nUnderstand concepts step-by-step.")

        with card3:
            st.info("🌐 **Live Information**\n\nUse Web Search when you need current information.")

        st.divider()
        st.markdown("### 💡 Try asking")

        ex1, ex2 = st.columns(2)

        with ex1:
            st.write("🐍 Explain Python loops for beginners")
            st.write("☕ What is inheritance in Java?")
            st.write("⚛️ Explain React props simply")

        with ex2:
            st.write("🤖 What is Artificial Intelligence?")
            st.write("💻 Write a simple Java program")
            st.write("📱 What is the latest smartphone price?")


else:
    st.markdown("# 🤖 Adarsh AI")
    st.markdown("### Your Personal AI Assistant")
    st.caption("💡 Ask questions • Learn • Explore • Get answers")
    st.divider()

    st.markdown("## 👋 Welcome to Adarsh AI")
    st.write(
        "Your personal AI assistant for learning, programming, technology "
        "and everyday questions."
    )
    st.write("Please use **👤 User Login** in the sidebar to get started.")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🤖 **AI Assistant**\n\nAsk questions and get clear answers.")

    with col2:
        st.info("📚 **Learning**\n\nUnderstand concepts step-by-step.")

    with col3:
        st.info("🌐 **Live Information**\n\nSearch current information.")


# =========================================================
# DISPLAY CHAT
# =========================================================

for message in st.session_state.messages:
    role = message.get("role")
    content = message.get("content", "")

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(content)

    elif role == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)


# =========================================================
# CHAT INPUT
# =========================================================

user_message = st.chat_input("💬 Ask Adarsh AI anything...")


# =========================================================
# AI SYSTEM INSTRUCTION
# =========================================================

if user_message:

    st.session_state.messages.append(
        {"role": "user", "content": user_message}
    )

    with st.chat_message("user", avatar="👤"):
        st.write(user_message)

    if st.session_state.beginner_mode:
        level_instruction = """
Always explain in simple, beginner-friendly language.
Give the simplest working solution first.
For programming, avoid unnecessary functions, classes,
frameworks and complexity.
For a simple Hello World question, prefer:
print("Hello, World!")
Do not unnecessarily create main() or
if __name__ == "__main__": unless requested.
"""
    else:
        level_instruction = """
Give technically accurate explanations appropriate for
an experienced learner. Still keep the answer clear and organized.
"""

    system_instruction = f"""
You are Adarsh AI, a personal AI assistant developed by Adarsh Dixit.

Developer information:
Adarsh Dixit is currently pursuing a Bachelor of Computer
Applications (BCA), specializing in Data Science and
Artificial Intelligence.

IDENTITY:
- If asked "Who are you?", say:
  "I'm Adarsh AI, a personal AI assistant developed by Adarsh Dixit."
- If asked who created/developed you, say:
  "My developer is Adarsh Dixit."
- If asked for more developer information, give the BCA/Data Science &
  AI information above.
- Never claim Adarsh Dixit created the underlying AI model or infrastructure.
- Do not mention the underlying AI provider unless the user specifically
  asks about the technical implementation.

{level_instruction}

CONVERSATION:
Use the recent conversation context for follow-up questions.
Understand references such as "this", "that", "it", "same", "continue",
"previous" and "above".

CURRENT INFORMATION:
When web search is available and the user asks for current information,
use it. Never invent facts or sources.

ANSWER STYLE:
Be friendly, professional and clear.
Use headings, bullets and examples when useful.
Keep simple questions concise.
Give detail when necessary.
"""

    # Keep the full conversation visible locally, but send only a
    # bounded recent context to prevent oversized API requests.
    MAX_CONTEXT_MESSAGES = 8
    MAX_MESSAGE_CHARS = 5000

    recent_messages = st.session_state.messages[-MAX_CONTEXT_MESSAGES:]

    messages_for_ai = [{"role": "system", "content": system_instruction}]

    for message in recent_messages:
        content = str(message.get("content", ""))

        if len(content) > MAX_MESSAGE_CHARS:
            content = content[-MAX_MESSAGE_CHARS:]

        messages_for_ai.append(
            {
                "role": message["role"],
                "content": content,
            }
        )

    # =====================================================
    # GENERATE RESPONSE
    # =====================================================

    with st.chat_message("assistant", avatar="🤖"):

        try:

            # -------------------------------------------------
            # WEB SEARCH / GROQ COMPOUND
            # -------------------------------------------------

            if st.session_state.web_search:

                with st.status(
                    "⚡ Searching and thinking...",
                    expanded=False,
                ):
                    response = client.chat.completions.create(
                        model="groq/compound",
                        messages=messages_for_ai,
                    )

                answer = response.choices[0].message.content or ""

                if not answer:
                    answer = "Sorry, I couldn't generate an answer."

                st.markdown(answer)

            # -------------------------------------------------
            # FAST MODE WITH STREAMING
            # -------------------------------------------------

            else:

                with st.status(
                    "⚡ Adarsh AI is thinking...",
                    expanded=False,
                ):
                    stream = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=messages_for_ai,
                        temperature=0.3,
                        max_tokens=2048,
                        stream=True,
                    )

                    answer_parts = []
                    answer_placeholder = st.empty()

                    for chunk in stream:
                        try:
                            delta = chunk.choices[0].delta.content
                        except (AttributeError, IndexError):
                            delta = None

                        if delta:
                            answer_parts.append(delta)
                            answer_placeholder.markdown(
                                "".join(answer_parts) + "▌"
                            )

                    answer = "".join(answer_parts).strip()

                answer_placeholder.markdown(
                    answer or "Sorry, I couldn't generate an answer."
                )

                if not answer:
                    answer = "Sorry, I couldn't generate an answer."

            # -------------------------------------------------
            # SAVE ANSWER
            # -------------------------------------------------

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            # -------------------------------------------------
            # WEB SOURCES
            # -------------------------------------------------

            if st.session_state.web_search:
                message_data = response.choices[0].message
                executed_tools = getattr(
                    message_data,
                    "executed_tools",
                    None,
                )

                sources = []

                if executed_tools:
                    for tool in executed_tools:
                        search_results = getattr(
                            tool,
                            "search_results",
                            None,
                        )

                        if not search_results:
                            continue

                        if isinstance(search_results, dict):
                            results = search_results.get("results", [])
                        else:
                            results = search_results

                        for result in results:
                            if isinstance(result, dict):
                                title = result.get("title", "Web Source")
                                url = result.get("url", "")
                            else:
                                title = getattr(
                                    result,
                                    "title",
                                    "Web Source",
                                )
                                url = getattr(result, "url", "")

                            if url:
                                sources.append((title, url))

                unique_sources = []
                seen_urls = set()

                for title, url in sources:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        unique_sources.append((title, url))

                if unique_sources:
                    st.divider()
                    st.markdown("### 📚 Sources")

                    for index, (title, url) in enumerate(
                        unique_sources,
                        start=1,
                    ):
                        st.markdown(
                            f"{index}. [{title}]({url})"
                        )

        except Exception as error:

            # Remove the failed user message so the next retry does
            # not resend a broken request as if it were successful.
            if (
                st.session_state.messages
                and st.session_state.messages[-1].get("role") == "user"
                and st.session_state.messages[-1].get("content") == user_message
            ):
                st.session_state.messages.pop()

            error_text = str(error)

            st.error(
                "❌ I couldn't complete that request. "
                "Please try again."
            )

            with st.expander("🔧 Technical Details"):
                st.code(error_text)
