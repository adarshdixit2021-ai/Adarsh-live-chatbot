import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import gspread
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from google.oauth2.service_account import Credentials


# =========================================================
# STREAMLIT UI CONFIGURATION
# =========================================================

try:
    st.set_option("client.toolbarMode", "minimal")
    st.set_option("client.showSidebarNavigation", False)
except Exception:
    pass

st.set_page_config(
    page_title="Adarsh AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    },
)


# =========================================================
# ENVIRONMENT / SECRETS
# =========================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_sheet_id = os.getenv("GOOGLE_SHEET_ID")

if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

if not google_sheet_id:
    try:
        google_sheet_id = st.secrets["GOOGLE_SHEET_ID"]
    except Exception:
        google_sheet_id = None

if not groq_api_key:
    st.error(
        "❌ Adarsh AI is not configured correctly. "
        "GROQ_API_KEY is missing."
    )
    st.stop()

client = Groq(api_key=groq_api_key)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "messages": [],
    "chat_history": [],
    "user_profile": None,
    "show_login": False,
    "beginner_mode": True,
    "web_search": False,
    "feedback_sent": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CONSTANTS
# =========================================================

SUPPORT_EMAIL = "adarshdixit2021@gmail.com"

FEEDBACK_START_DATE = date(2026, 9, 15)
FEEDBACK_END_DATE = FEEDBACK_START_DATE + timedelta(days=10)

feedback_available = date.today() <= FEEDBACK_END_DATE


# =========================================================
# HELPERS
# =========================================================

def is_current_information_question(text):
    """Use web search only for questions that clearly need fresh information."""
    text = text.lower().strip()

    current_words = [
        "latest",
        "today",
        "currently",
        "current",
        "right now",
        "recent",
        "recently",
        "this week",
        "this month",
        "this year",
        "news",
        "price today",
        "live score",
        "weather today",
        "stock price",
        "share price",
        "what happened today",
        "who is the current",
        "current ceo",
        "current president",
        "current prime minister",
    ]

    return any(word in text for word in current_words)


def clean_answer_for_display(answer):
    """Prevent common LaTeX delimiters from appearing as raw text."""
    answer = answer.replace(r"\(", "$")
    answer = answer.replace(r"\)", "$")
    answer = answer.replace(r"\[", "$$")
    answer = answer.replace(r"\]", "$$")
    answer = answer.replace("\\**", "**")
    return answer.strip()


def save_current_chat():
    """Save the current conversation with a useful title."""
    messages = st.session_state.messages

    if not messages:
        return

    first_user_message = next(
        (
            message["content"]
            for message in messages
            if message["role"] == "user"
        ),
        "",
    )

    if not first_user_message:
        return

    title = " ".join(first_user_message.split())

    if len(title) > 55:
        title = title[:55].rstrip() + "..."

    st.session_state.chat_history.append(
        {
            "id": datetime.now().timestamp(),
            "title": title,
            "messages": messages.copy(),
            "pinned": False,
        }
    )


def start_new_chat():
    if st.session_state.messages:
        save_current_chat()

    st.session_state.messages = []


def pin_chat(chat_id):
    for chat in st.session_state.chat_history:
        if chat["id"] == chat_id:
            chat["pinned"] = not chat["pinned"]
            break


def delete_chat(chat_id):
    st.session_state.chat_history = [
        chat
        for chat in st.session_state.chat_history
        if chat["id"] != chat_id
    ]


def connect_google_sheet():
    """Use Streamlit Secrets on Cloud or local JSON during local development."""
    try:
        if not google_sheet_id:
            return None

        if "gcp_service_account" in st.secrets:
            service_account_info = dict(
                st.secrets["gcp_service_account"]
            )

            credentials = Credentials.from_service_account_info(
                service_account_info,
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive",
                ],
            )

            google_client = gspread.authorize(credentials)

        else:
            google_credentials_file = os.getenv(
                "GOOGLE_CREDENTIALS_FILE"
            )

            if not google_credentials_file:
                return None

            if not os.path.exists(google_credentials_file):
                return None

            google_client = gspread.service_account(
                filename=google_credentials_file
            )

        spreadsheet = google_client.open_by_key(google_sheet_id)
        return spreadsheet.sheet1

    except Exception:
        return None


@st.cache_resource
def get_google_sheet():
    return connect_google_sheet()


def save_feedback(feedback_text, rating):
    worksheet = get_google_sheet()

    if worksheet is None:
        return False

    try:
        profile = st.session_state.user_profile

        user_name = profile["name"] if profile else "Guest"
        profession = profile["profession"] if profile else "Not provided"

        india_time = datetime.now(
            ZoneInfo("Asia/Kolkata")
        )

        formatted_time = india_time.strftime(
            "%d %B %Y, %I:%M:%S %p"
        )

        worksheet.append_row(
            [
                formatted_time,
                user_name,
                feedback_text.strip(),
                rating,
                profession,
                "New",
            ],
            value_input_option="USER_ENTERED",
        )

        return True

    except Exception:
        return False


def build_system_prompt():
    return """
You are Adarsh AI, a personal AI assistant developed by Adarsh Dixit.

DEVELOPER IDENTITY:
If asked who you are:
"I'm Adarsh AI, a personal AI assistant developed by Adarsh Dixit."

If asked who your developer is:
"My developer is Adarsh Dixit."

Adarsh Dixit is currently pursuing Bachelor of Computer Applications (BCA),
specializing in Data Science and Artificial Intelligence.

Do not claim that Adarsh Dixit created the underlying AI model.

==================================================
CORE ANSWER RULE
==================================================

Answer EXACTLY what the user asked.

Do NOT add unrelated information.
Do NOT repeat the question unnecessarily.
Do NOT add random sections just to make the answer longer.
Do NOT give unnecessary web results.
Do NOT give unnecessary tables.
Do NOT end with generic offers such as "Let me know if you need anything else."

Use simple, beginner-friendly language.

IMPORTANT KEYWORDS should be in **bold**.

==================================================
EDUCATIONAL / THEORY QUESTIONS
==================================================

When the user asks for a definition, concept, exam note, or theory answer:

Use this structure when appropriate:

### Definition
Give a simple 1-3 sentence definition.

### Key Points
Give the most important points using short bullets.
Bold important keywords.

### Example
Give 1 or 2 realistic real-life examples ONLY when they actually help.

### Exam Answer
Give a compact answer suitable for approximately 4 marks when the
question looks like an academic/exam question.

Do not force all four sections if the question is very simple.

For a 4-mark answer:
- Focus on definition + 3-4 key points.
- Keep it organized.
- Do not add unrelated advanced information.

==================================================
PROGRAMMING QUESTIONS
==================================================

For coding questions:

1. Give the simplest correct solution first.
2. Give code.
3. Explain the important keywords/logic briefly.
4. Give a small example/output when useful.

Do not over-engineer beginner problems.

For a simple Hello World question, prefer:
print("Hello, World!")

Do not unnecessarily create main functions/classes/frameworks.

==================================================
MATHEMATICS
==================================================

For mathematics:

- Show the formula clearly.
- Use readable Markdown/LaTeX.
- Use $...$ for inline math.
- Use $$...$$ for display equations.
- NEVER use round-bracket LaTeX delimiters or square-bracket LaTeX delimiters.
- Solve step-by-step.
- Do not leave raw LaTeX commands visible.

==================================================
FOLLOW-UP QUESTIONS
==================================================

Remember the recent conversation.

If the user says "this", "that", "continue", "above", "same",
or similar words, use the previous conversation context.

==================================================
CURRENT INFORMATION / WEB
==================================================

Only use web search when the user asks for information that genuinely
needs current information, such as latest news, current prices, today's
weather, current events, live scores, or recent developments.

For normal educational, programming, mathematics, general knowledge,
or conversational questions, answer directly without web search.

Never invent current information or sources.

==================================================
TONE
==================================================

Be friendly, clear, concise and professional.

Prefer:
- Simple language
- Short paragraphs
- Bullet points
- Bold keywords
- Useful examples

Avoid:
- Unnecessary complexity
- Repetition
- Excessive headings
- Irrelevant facts
- Long introductions
"""


def get_chat_response(user_message):
    recent_messages = st.session_state.messages[-12:]

    messages_for_ai = [
        {
            "role": "system",
            "content": build_system_prompt(),
        }
    ]

    for message in recent_messages:
        messages_for_ai.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    needs_web = (
        st.session_state.web_search
        and is_current_information_question(user_message)
    )

    if needs_web:
        response = client.chat.completions.create(
            model="groq/compound-mini",
            messages=messages_for_ai,
        )
    else:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages_for_ai,
            reasoning_effort="low",
            max_tokens=2048,
        )

    answer = response.choices[0].message.content or ""

    return clean_answer_for_display(answer), response, needs_web


def extract_sources(response):
    sources = []

    try:
        executed_tools = getattr(
            response.choices[0].message,
            "executed_tools",
            None,
        )

        if not executed_tools:
            return []

        for tool in executed_tools:
            search_results = getattr(
                tool,
                "search_results",
                None,
            )

            if not search_results:
                continue

            if isinstance(search_results, dict):
                results = search_results.get(
                    "results",
                    [],
                )
            else:
                results = search_results

            for result in results:
                if isinstance(result, dict):
                    title = result.get(
                        "title",
                        "Web Source",
                    )
                    url = result.get(
                        "url",
                        "",
                    )
                else:
                    title = getattr(
                        result,
                        "title",
                        "Web Source",
                    )
                    url = getattr(
                        result,
                        "url",
                        "",
                    )

                if url:
                    sources.append(
                        (
                            title,
                            url,
                        )
                    )

    except Exception:
        return []

    unique_sources = []
    seen_urls = set()

    for title, url in sources:
        if url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(
                (
                    title,
                    url,
                )
            )

    return unique_sources[:5]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    if st.session_state.user_profile:
        st.markdown(
            f"## 👤 {st.session_state.user_profile['name']}"
        )
        st.caption("Your Personal AI Assistant")
    else:
        st.markdown("## 🤖 Adarsh AI")
        st.caption("Your Personal AI Assistant")

    st.divider()

    # -----------------------------------------------------
    # PROFILE
    # -----------------------------------------------------

    with st.expander("👤 Profile", expanded=False):

        if st.session_state.user_profile:

            profile = st.session_state.user_profile

            st.write(f"**Name:** {profile['name']}")
            st.write(f"**Date of Birth:** {profile['dob']}")
            st.write(f"**Gender:** {profile['gender']}")
            st.write(f"**Profession:** {profile['profession']}")

            if st.button(
                "✏️ Edit Profile",
                use_container_width=True,
            ):
                st.session_state.show_login = True
                st.session_state.user_profile = None
                st.rerun()

        else:

            st.caption(
                "Create your profile for a personalized experience."
            )

            if st.button(
                "👤 Create Profile",
                use_container_width=True,
            ):
                st.session_state.show_login = True
                st.rerun()

    # -----------------------------------------------------
    # LOGIN FORM
    # -----------------------------------------------------

    if (
        st.session_state.show_login
        and st.session_state.user_profile is None
    ):

        st.markdown("### Create Profile")

        user_name = st.text_input(
            "Name",
            placeholder="Enter your name",
            key="login_name",
        )

        user_dob = st.date_input(
            "Date of Birth",
            value=date(2000, 1, 1),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
            key="login_dob",
        )

        user_gender = st.selectbox(
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

        user_profession = st.text_input(
            "Profession",
            placeholder="e.g. Student, Developer",
            key="login_profession",
        )

        if st.button(
            "✅ Save Profile",
            use_container_width=True,
        ):

            if not user_name.strip():
                st.warning("Please enter your name.")

            elif user_gender == "Select Gender":
                st.warning("Please select your gender.")

            elif not user_profession.strip():
                st.warning("Please enter your profession.")

            else:
                st.session_state.user_profile = {
                    "name": user_name.strip(),
                    "dob": user_dob.strftime("%d %B %Y"),
                    "gender": user_gender,
                    "profession": user_profession.strip(),
                }

                st.session_state.show_login = False

                st.success("Profile saved.")
                st.rerun()

    st.divider()

    # -----------------------------------------------------
    # NEW CHAT
    # -----------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True,
    ):
        start_new_chat()
        st.rerun()

    # -----------------------------------------------------
    # CHAT HISTORY
    # -----------------------------------------------------

    st.markdown("### 💬 Chat History")

    if not st.session_state.chat_history:

        st.caption("No saved chats yet.")

    else:

        pinned_chats = [
            chat
            for chat in st.session_state.chat_history
            if chat.get("pinned")
        ]

        normal_chats = [
            chat
            for chat in st.session_state.chat_history
            if not chat.get("pinned")
        ]

        ordered_chats = pinned_chats + list(
            reversed(normal_chats)
        )

        for index, chat in enumerate(ordered_chats):

            row1, row2, row3 = st.columns(
                [7, 1, 1],
                gap="small",
            )

            with row1:

                prefix = "📌 " if chat.get("pinned") else "💬 "

                if st.button(
                    prefix + chat["title"],
                    key=f"open_chat_{chat['id']}_{index}",
                    use_container_width=True,
                ):

                    st.session_state.messages = (
                        chat["messages"].copy()
                    )

                    st.rerun()

            with row2:

                if st.button(
                    "📌",
                    key=f"pin_chat_{chat['id']}_{index}",
                    help="Pin / unpin chat",
                ):

                    pin_chat(chat["id"])
                    st.rerun()

            with row3:

                if st.button(
                    "🗑️",
                    key=f"delete_chat_{chat['id']}_{index}",
                    help="Delete chat",
                ):

                    delete_chat(chat["id"])
                    st.rerun()

    if st.session_state.chat_history:

        if st.button(
            "🗑️ Delete All History",
            use_container_width=True,
        ):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # -----------------------------------------------------
    # SETTINGS
    # -----------------------------------------------------

    with st.expander("⚙️ Settings", expanded=False):

        st.session_state.beginner_mode = st.toggle(
            "👶 Beginner-friendly answers",
            value=st.session_state.beginner_mode,
            help="Keeps explanations simple and exam-friendly.",
        )

        st.session_state.web_search = st.toggle(
            "🌐 Current information search",
            value=st.session_state.web_search,
            help=(
                "Search is used only for questions that need "
                "current information."
            ),
        )

        st.caption(
            "🎨 Light/Dark theme is controlled by Streamlit's "
            "native theme and your device/browser preference. "
            "The app does not inject custom HTML/CSS."
        )

    # -----------------------------------------------------
    # HELP & SUPPORT
    # -----------------------------------------------------

    if feedback_available:

        with st.expander(
            "📣 Help & Feedback",
            expanded=False,
        ):

            feedback_text = st.text_area(
                "Your Feedback",
                placeholder=(
                    "Tell us about a bug, problem or suggestion..."
                ),
                height=110,
            )

            feedback_rating = st.slider(
                "⭐ Rating",
                min_value=1,
                max_value=5,
                value=5,
            )

            if st.button(
                "📤 Submit Feedback",
                use_container_width=True,
            ):

                if not feedback_text.strip():

                    st.warning("Please enter your feedback.")

                elif save_feedback(
                    feedback_text,
                    feedback_rating,
                ):

                    st.success(
                        "✅ Thank you! Your feedback has been submitted."
                    )

                else:

                    st.error(
                        "❌ Feedback could not be saved right now."
                    )

    with st.expander("🆘 Support", expanded=False):

        st.write(
            "For support, bugs or project-related questions:"
        )

        st.markdown(
            f"📧 **{SUPPORT_EMAIL}**"
        )

        st.link_button(
            "✉️ Email Support",
            f"mailto:{SUPPORT_EMAIL}",
            use_container_width=True,
        )

    st.divider()

    st.caption("👨‍💻 Developer")
    st.caption("Adarsh Dixit")
    st.caption("BCA • Data Science & AI")


# =========================================================
# MAIN CONTENT
# =========================================================

if st.session_state.user_profile:

    name = st.session_state.user_profile["name"]

    if not st.session_state.messages:

        st.markdown(
            f"# 🤖 Hi {name}!"
        )

        st.markdown(
            "### Your Personal AI Assistant"
        )

        st.caption(
            "Ask a question and get a clear, organized answer."
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(
                "**📚 Learn**\n\n"
                "Simple definitions, key points and examples."
            )

        with col2:
            st.info(
                "**💻 Code**\n\n"
                "Simple solutions with useful explanations."
            )

        with col3:
            st.info(
                "**🌐 Current Info**\n\n"
                "Search current information when needed."
            )

        st.divider()

        st.markdown("### 💡 Try asking")

        st.write(
            "• What is inheritance in Java?"
        )
        st.write(
            "• Explain matrices for 4 marks."
        )
        st.write(
            "• Write a simple Python program to reverse a string."
        )

    else:

        st.markdown(
            "# 🤖 Adarsh AI"
        )

        st.caption(
            "Clear answers. Simple explanations. No unnecessary information."
        )

else:

    st.markdown("# 🤖 Adarsh AI")

    st.markdown(
        "### Your Personal AI Assistant"
    )

    st.caption(
        "Ask questions • Learn • Explore • Get answers"
    )

    st.divider()

    st.markdown(
        "## 👋 Welcome"
    )

    st.write(
        "Ask me about programming, mathematics, technology, "
        "Artificial Intelligence, Data Science or everyday questions."
    )

    st.info(
        "👤 Create a profile from the sidebar if you want "
        "a personalized experience."
    )


# =========================================================
# RENDER EXISTING CHAT
# =========================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message(
            "user",
            avatar="👤",
        ):

            st.write(message["content"])

    else:

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            st.markdown(message["content"])

            with st.expander(
                "📋 Copy answer",
                expanded=False,
            ):

                st.code(
                    message["content"],
                    language="markdown",
                )


# =========================================================
# CHAT INPUT
# =========================================================

user_message = st.chat_input(
    "💬 Ask Adarsh AI anything..."
)


# =========================================================
# PROCESS NEW QUESTION
# =========================================================

if user_message:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message(
        "user",
        avatar="👤",
    ):

        st.write(user_message)

    with st.chat_message(
        "assistant",
        avatar="🤖",
    ):

        try:

            needs_web = (
                st.session_state.web_search
                and is_current_information_question(
                    user_message
                )
            )

            if needs_web:

                with st.spinner(
                    "🔎 Finding current information..."
                ):

                    answer, response, used_web = (
                        get_chat_response(
                            user_message
                        )
                    )

            else:

                with st.spinner(
                    "🧠 Thinking..."
                ):

                    answer, response, used_web = (
                        get_chat_response(
                            user_message
                        )
                    )

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            with st.expander(
                "📋 Copy answer",
                expanded=False,
            ):

                st.code(
                    answer,
                    language="markdown",
                )

            if used_web:

                sources = extract_sources(response)

                if sources:

                    with st.expander(
                        "📚 Sources",
                        expanded=False,
                    ):

                        for source_index, (
                            title,
                            url,
                        ) in enumerate(
                            sources,
                            start=1,
                        ):

                            st.markdown(
                                f"{source_index}. "
                                f"[{title}]({url})"
                            )

        except Exception as error:

            error_text = str(error).lower()

            if "401" in error_text or "authentication" in error_text:

                friendly_error = (
                    "❌ The AI service authentication failed. "
                    "Please check the API key in Streamlit Secrets."
                )

            elif "429" in error_text or "rate limit" in error_text:

                friendly_error = (
                    "⏳ The AI service is temporarily rate-limited. "
                    "Please wait a moment and try again."
                )

            elif "413" in error_text:

                friendly_error = (
                    "⚠️ The conversation became too large. "
                    "Start a New Chat and try again."
                )

            elif "timeout" in error_text:

                friendly_error = (
                    "⏱️ The request took too long. "
                    "Please try again."
                )

            else:

                friendly_error = (
                    "❌ I couldn't generate the answer right now. "
                    "Please try again."
                )

            st.error(friendly_error)
