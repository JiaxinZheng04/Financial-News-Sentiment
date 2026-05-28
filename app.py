import time
import streamlit as st
from transformers import pipeline


# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Financial News Sentiment and Risk Summary Tool",
    page_icon="📈",
    layout="wide"
)


# ==================================================
# Model Loading
# ==================================================
@st.cache_resource
def load_sentiment_model():
    """
    Load the fine-tuned DistilBERT sentiment classification model.
    """
    return pipeline(
        task="text-classification",
        model="Xinn94L/fiqa-financial-sentiment-distilbert",
        tokenizer="Xinn94L/fiqa-financial-sentiment-distilbert"
    )


@st.cache_resource
def load_summary_model():
    """
    Load a lightweight text-to-text generation model for summarization.
    This uses text2text-generation instead of summarization to avoid
    Streamlit Cloud task compatibility issues.
    """
    return pipeline(
        task="text2text-generation",
        model="t5-small",
        tokenizer="t5-small"
    )


# Load sentiment model at startup
classifier = load_sentiment_model()


# ==================================================
# Helper Functions
# ==================================================
def build_model_input(headline: str, target: str, aspect: str) -> str:
    """
    Keep the app input format consistent with the model training format.
    """
    return f"Headline: {headline} Target: {target} Aspect: {aspect}"


def generate_summary(text: str) -> tuple[str, float]:
    """
    Generate a concise summary using T5 text2text-generation.
    Returns summary text and runtime.
    """
    start_time = time.time()

    if len(text.split()) < 15:
        runtime = time.time() - start_time
        return "The input is relatively short, so no additional summary is required.", runtime

    try:
        summarizer = load_summary_model()
        prompt = "summarize: " + text

        result = summarizer(
            prompt,
            max_length=60,
            min_length=10,
            do_sample=False
        )

        summary = result[0]["generated_text"]
        runtime = time.time() - start_time
        return summary, runtime

    except Exception as error:
        runtime = time.time() - start_time
        return f"Summary generation is currently unavailable. Error: {str(error)}", runtime


def generate_risk_note(label: str, confidence: float, aspect: str) -> str:
    """
    Generate a simple risk note based on the sentiment result.
    This is not investment advice. It is only an explanatory business-facing output.
    """
    label = label.upper()

    if confidence < 0.5:
        return (
            "The model confidence is relatively low. Manual review is recommended before "
            "using this result for financial interpretation."
        )

    if label == "NEGATIVE":
        return (
            f"This news may indicate potential downside risk related to {aspect}. "
            "Users may need to monitor earnings guidance, regulatory developments, "
            "market reaction, or company-specific exposure."
        )

    if label == "POSITIVE":
        return (
            f"This news may indicate positive market sentiment related to {aspect}. "
            "Users may monitor whether this signal is supported by company fundamentals, "
            "future guidance, or broader market conditions."
        )

    return (
        f"This news appears relatively neutral regarding {aspect}. "
        "Users may continue monitoring related updates before drawing strong conclusions."
    )


def format_sentiment_label(label: str) -> str:
    """
    Add visual indicator to sentiment label.
    """
    label = label.upper()

    if label == "POSITIVE":
        return "🟢 POSITIVE"
    if label == "NEGATIVE":
        return "🔴 NEGATIVE"
    return "🟡 NEUTRAL"


# ==================================================
# Main App Interface
# ==================================================
st.title("📈 Financial News Sentiment and Risk Summary Tool")

st.write(
    "This application helps users screen financial news by classifying market sentiment "
    "and generating a concise risk note using Hugging Face deep learning models."
)

st.info(
    "Pipeline 1: Fine-tuned DistilBERT sentiment classifier  \n"
    "Pipeline 2: T5 text-to-text generation model for short summary generation"
)


# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    st.header("About This App")

    st.write(
        "This prototype is designed for a digital brokerage or wealth management context, "
        "where users need to quickly screen financial news and identify possible risk signals."
    )

    st.markdown("**Sentiment model:**")
    st.code("Xinn94L/fiqa-financial-sentiment-distilbert")

    st.markdown("**Summary model:**")
    st.code("t5-small")

    st.warning(
        "Disclaimer: This tool is for information screening only. "
        "It does not provide investment advice."
    )


# ==================================================
# Tabs
# ==================================================
tab1, tab2 = st.tabs(["Single News Analysis", "Example Inputs"])


# ==================================================
# Tab 1: Single News Analysis
# ==================================================
with tab1:
    st.subheader("Enter Financial News Information")

    col1, col2 = st.columns([2, 1])

    with col1:
        headline = st.text_area(
            "Headline or news text",
            height=180,
            placeholder=(
                "Example: Apple reported stronger-than-expected quarterly earnings "
                "and raised its revenue guidance."
            )
        )

    with col2:
        target = st.text_input(
            "Target company / asset",
            placeholder="Example: Apple"
        )

        aspect = st.text_input(
            "Aspect",
            placeholder="Example: earnings, revenue guidance, regulation"
        )

    analyze_button = st.button("Analyze News", type="primary")

    if analyze_button:
        if not headline.strip():
            st.warning("Please enter a financial news headline or text.")
        elif not target.strip():
            st.warning("Please enter a target company or asset.")
        elif not aspect.strip():
            st.warning("Please enter an aspect.")
        else:
            model_input = build_model_input(headline, target, aspect)

            # ------------------------------
            # Sentiment classification
            # ------------------------------
            with st.spinner("Running sentiment classification..."):
                sentiment_start = time.time()
                sentiment_result = classifier(model_input)[0]
                sentiment_runtime = time.time() - sentiment_start

            label = sentiment_result["label"]
            confidence = float(sentiment_result["score"])

            # ------------------------------
            # Summary generation
            # ------------------------------
            with st.spinner("Generating short summary..."):
                summary, summary_runtime = generate_summary(headline)

            # ------------------------------
            # Risk note
            # ------------------------------
            risk_note = generate_risk_note(label, confidence, aspect)

            # ------------------------------
            # Display results
            # ------------------------------
            st.divider()
            st.subheader("Analysis Results")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Sentiment", format_sentiment_label(label))

            with metric_col2:
                st.metric("Confidence", f"{confidence:.2%}")

            with metric_col3:
                st.metric("Sentiment Runtime", f"{sentiment_runtime:.3f}s")

            if confidence < 0.5:
                st.warning(
                    "Low-confidence result detected. Manual review is recommended."
                )

            st.subheader("Short Summary")
            st.write(summary)
            st.caption(f"Summary runtime: {summary_runtime:.3f}s")

            st.subheader("Risk Note")
            st.write(risk_note)

            with st.expander("Show model input format"):
                st.code(model_input)

            st.caption(
                "This output is generated for information screening only and should not be interpreted as financial advice."
            )


# ==================================================
# Tab 2: Example Inputs
# ==================================================
with tab2:
    st.subheader("Sample Financial News Inputs")

    examples = [
        {
            "headline": "Apple reported stronger-than-expected quarterly earnings and raised its revenue guidance.",
            "target": "Apple",
            "aspect": "earnings and revenue guidance"
        },
        {
            "headline": "Tesla shares fell after the company missed earnings expectations.",
            "target": "Tesla",
            "aspect": "earnings performance"
        },
        {
            "headline": "The company announced a new board appointment on Monday.",
            "target": "The company",
            "aspect": "corporate governance"
        },
        {
            "headline": "The retailer cut its annual forecast due to weak consumer spending.",
            "target": "The retailer",
            "aspect": "annual forecast"
        },
        {
            "headline": "The company faces a regulatory investigation over data privacy issues.",
            "target": "The company",
            "aspect": "regulatory investigation"
        },
    ]

    for i, example in enumerate(examples, start=1):
        with st.expander(f"Example {i}"):
            st.markdown("**Headline:**")
            st.write(example["headline"])

            st.markdown("**Target:**")
            st.write(example["target"])

            st.markdown("**Aspect:**")
            st.write(example["aspect"])

            st.markdown("**Model input format:**")
            st.code(
                build_model_input(
                    example["headline"],
                    example["target"],
                    example["aspect"]
                )
            )
