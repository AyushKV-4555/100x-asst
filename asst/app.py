import streamlit as st
from streamlit_mic_recorder import mic_recorder

from speech_to_text import transcribe_audio
from text_to_speech import speak
from bot import ask_bot

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Personal AI Voice Bot",
    layout="centered"
)

# ---------------- HEADER ----------------
st.title("🎙️ Personal AI Voice Bot")
st.divider()

# ---------------- MODE STATE ----------------
if "mode" not in st.session_state:
    st.session_state.mode = "Voice"

# ---------------- MODE SELECTION ----------------
st.header("Talk to me using voice or text")

col1, col2 = st.columns(2)

with col1:
    if st.button("🎙 Voice Mode", use_container_width=True):
        st.session_state.mode = "Voice"

with col2:
    if st.button("💬 Text Mode", use_container_width=True):
        st.session_state.mode = "Text"

st.divider()

# ================= TEXT MODE =================
if st.session_state.mode == "Text":
    st.header("💬 Ask Your Question")

    user_input = st.text_input(
        "Type your question below",
        placeholder="Example: What is your research proposal about?"
    )

    if st.button("Ask Question", use_container_width=True):
        if user_input.strip():
            with st.spinner("Thinking..."):
                answer = ask_bot(user_input)

            st.divider()
            st.header("🧠 Answer")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

# ================= VOICE MODE =================
else:
    st.header("🎤 Speak Your Question")
    st.markdown("Click below, speak clearly, then stop recording")

    audio = mic_recorder(
        start_prompt="🎙 Start Recording",
        stop_prompt="⏹ Stop Recording",
        use_container_width=True
    )

    if audio:
        st.divider()
        st.header("🗣 You Said")

        with st.spinner("Transcribing..."):
            text = transcribe_audio(audio["bytes"])

        st.write(text)

        st.divider()
        st.header("🤖 Answer")

        with st.spinner("Thinking..."):
            answer = ask_bot(text)

        st.write(answer)

        with st.spinner("Speaking..."):
            audio_file = speak(answer)
            st.audio(audio_file)
