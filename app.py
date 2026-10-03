import streamlit as st
from pypdf import PdfReader
from transformers import pipeline

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="Fast Text Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Fast AI Text Summarizer")
st.write("Paste text or upload a TXT/PDF file and get a quick summary.")


# -----------------------------
# Load AI Model
# -----------------------------
@st.cache_resource
def load_model():

    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )


# -----------------------------
# Read PDF
# -----------------------------
def read_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Split Text
# -----------------------------
def split_text(text, max_words=350):

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):
        chunks.append(
            " ".join(words[i:i + max_words])
        )

    return chunks


# -----------------------------
# Summarize
# -----------------------------
def summarize_text(text, summarizer):

    chunks = split_text(text)

    summaries = []

    progress = st.progress(0)

    for i, chunk in enumerate(chunks):

        # Very small chunks don't need summarization
        if len(chunk.split()) < 40:
            summaries.append(chunk)
            continue

        result = summarizer(
            chunk,
            max_length=100,
            min_length=25,
            do_sample=False
        )

        summaries.append(
            result[0]["summary_text"]
        )

        progress.progress(
            (i + 1) / len(chunks)
        )

    progress.empty()

    return " ".join(summaries)


# -----------------------------
# Input
# -----------------------------
input_type = st.radio(
    "Choose Input",
    ["Paste Text", "Upload File"],
    horizontal=True
)

text = ""


# -----------------------------
# Paste Text
# -----------------------------
if input_type == "Paste Text":

    text = st.text_area(
        "Enter your text",
        height=300,
        placeholder="Paste your article, notes or report here..."
    )


# -----------------------------
# Upload File
# -----------------------------
else:

    file = st.file_uploader(
        "Upload TXT or PDF",
        type=["txt", "pdf"]
    )

    if file:

        try:

            if file.name.lower().endswith(".pdf"):

                text = read_pdf(file)

            else:

                text = file.read().decode(
                    "utf-8",
                    errors="ignore"
                )

            if text:

                with st.expander("📄 Preview"):

                    st.write(text[:3000])

        except Exception as e:

            st.error(f"Error reading file: {e}")


# -----------------------------
# Summarize Button
# -----------------------------
if st.button(
    "⚡ Generate Summary",
    type="primary",
    use_container_width=True
):

    text = text.strip()

    word_count = len(text.split())

    if word_count < 40:

        st.warning(
            "Please enter at least 40 words."
        )

    else:

        try:

            with st.spinner(
                "🤖 AI is generating your summary..."
            ):

                summarizer = load_model()

                summary = summarize_text(
                    text,
                    summarizer
                )

            # -----------------------------
            # Results
            # -----------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("📄 Original")

                st.caption(
                    f"{word_count} words"
                )

                st.write(text[:5000])

            with col2:

                st.subheader("✨ Summary")

                summary_words = len(
                    summary.split()
                )

                st.caption(
                    f"{summary_words} words"
                )

                st.success(summary)

                st.download_button(
                    "⬇️ Download Summary",
                    summary,
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )

