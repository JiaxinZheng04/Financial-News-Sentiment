import time
import streamlit as st
from transformers import pipeline


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Financial News Sentiment and Risk Summary Tool",
    page_icon="📈",
    layout="wide"
)


# --------------------------------------------------
# Load models
# --------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    classifier = pipeline(
        "text-classification",
        model="Xinn94L/fiqa-financial-sentiment-distilbert",
        tokenizer="Xinn94L/fiqa-financial-sentiment-distilbert"
    )
    return classifier


@st.cache_resource
def load_summarization_model():
    summarizer = pipeline(
        "summarization",
        model="t5-small",
        tokenizer="t5-small"
    )
    return summarizer


classifier = load_sentiment_model()
summarizer = load_summarization_model()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def build_model_input(headline, target, aspect):
    """
    Keep the inference input format consistent with the training format.
    """
    return f"Headline: {headline} Target: {target} Aspect: {aspect}"


def generate_summary(text):
    """
    Generate a short summary.
    If the input is too short, return a simple message instead of forcing summarization.
    """
    word_count = len(text.split())

    if word_count < 15:
        return "The input is relatively short, so no additional summary is required."

    try:
        summary = summarizer(
            text,
            max_length=60,
            min_length=10,
            do_sample=False
        )[0]["summary_text"]
        return summary
    except Exception:
        return "Summary generation is currently unavailable for this input."


def generate_risk_note(label, confidence, aspect):
    """
    Generate a simple business-facing risk note based on sentiment output.
    This is rule-based and used only for explanation, not as investment advice.
    """
    label = label.upper()

    if confidence < 0.5:
        return (
            "The model confidence is relatively low. This result should be reviewed manually "
            "before being used for business or investment-related interpretation."
        )

    if label == "NEGATIVE":
        return (
            f"The news may indicate potential downside risk related to {aspect}. "
            "Users may need to monitor earnings guidance, regulatory updates, market reaction, "
            "or company-specific exposure."
        )

    if label == "POSITIVE":
        return (
            f"The news may indicate positive market sentiment related to {aspect}. "
            "Users may monitor whether this signal is supported by fundamentals, future guidance, "
            "or broader market conditions."
        )

    return (
        f"The news appears relatively neutral regarding {aspect}. "
        "Users may continue monitoring related updates before drawing strong conclusions."
    )


def get_sentiment_badge(label):
    label = label.upper()

    if label == "POSITIVE":
        return "🟢 POSITIVE"
    if label == "NEGATIVE":
        return "🔴 NEGATIVE"
    return "🟡 NEUTRAL"


# --------------------------------------------------
# Main interface
# --------------------------------------------------
st.title("Financial News Sentiment and Risk Summary Tool")

st.write(
    "This application helps users screen financial news by classifying market sentiment "
    "and generating a concise risk summary using deep learning models."
)

st.info(
    "Model pipeline: Fine-tuned DistilBERT sentiment classifier + T5 summarization model."
)

with st.sidebar:
    st.header("About this tool")
    st.write(
        "This prototype is designed for financial news screening in a digital brokerage "
        "or wealth management context."
    )
    st.write("**Sentiment model:** Xinn94L/fiqa-financial-sentiment-distilbert")
    st.write("**Summary model:** t5-small")
    st.warning(
        "Disclaimer: This tool is for information screening only and does not provide investment advice."
    )


tab1, tab2 = st.tabs(["Single News Analysis", "Example Inputs"])


# --------------------------------------------------
# Tab 1: Single News Analysis
# --------------------------------------------------
with tab1:
    st.subheader("Enter financial news information")

    col1, col2 = st.columns(2)

    with col1:
        headline = st.text_area(
            "Headline or news text",
            height=160,
            placeholder="Example: Apple reported stronger-than-expected quarterly earnings and raised its revenue guidance."
        )

    with col2:
        target = st.text_input(
            "Target company / asset",
            placeholder="Example: Apple"
        )

        aspect = st.text_input(
            "Aspect",
            placeholder="Example: earnings, revenue guidance, regulation, market outlook"
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

            with st.spinner("Running sentiment classification..."):
                sentiment_start = time.time()
                sentiment_result = classifier(model_input)[0]
                sentiment_runtime = time.time() - sentiment_start

            label = sentiment_result["label"]
            confidence = sentiment_result["score"]

            with st.spinner("Generating short summary..."):
                summary_start = time.time()
                summary = generate_summary(headline)
                summary_runtime = time.time() - summary_start

            risk_note = generate_risk_note(label, confidence, aspect)

            st.divider()

            st.subheader("Analysis Results")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Sentiment", get_sentiment_badge(label))

            with metric_col2:
                st.metric("Confidence", f"{confidence:.2%}")

            with metric_col3:
                st.metric("Sentiment Runtime", f"{sentiment_runtime:.3f}s")

            if confidence < 0.5:
                st.warning(
                    "Low confidence result. Manual review is recommended before interpretation."
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


# --------------------------------------------------
# Tab 2: Example Inputs
# --------------------------------------------------
with tab2:
    st.subheader("Sample financial news inputs")

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
        }
    ]

    for i, example in enumerate(examples, start=1):
        with st.expander(f"Example {i}"):
            st.write("**Headline:**", example["headline"])
            st.write("**Target:**", example["target"])
            st.write("**Aspect:**", example["aspect"])
            st.code(
                build_model_input(
                    example["headline"],
                    example["target"],
                    example["aspect"]
                )
            )
