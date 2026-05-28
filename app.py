import time
import torch
import streamlit as st
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Financial News Risk Screening Tool",
    page_icon="📈",
    layout="wide"
)


# ==================================================
# Constants
# ==================================================
SENTIMENT_MODEL_NAME = "Xinn94L/fiqa-financial-sentiment-distilbert"
SUMMARY_MODEL_NAME = "t5-small"


# ==================================================
# Model Loading
# ==================================================
@st.cache_resource
def load_sentiment_resources():
    """
    Load the fine-tuned DistilBERT sentiment model.

    The app first tries to use Hugging Face's text-classification pipeline.
    If the pipeline fails on Streamlit Cloud, the app falls back to manual
    tokenizer + model inference so that deployment remains stable.
    """
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    model.eval()

    sentiment_pipe = None

    try:
        sentiment_pipe = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1
        )
    except Exception:
        sentiment_pipe = None

    return tokenizer, model, sentiment_pipe


@st.cache_resource
def load_summary_pipeline():
    """
    Load T5-small as a text2text-generation pipeline.
    This is used as the second Hugging Face pipeline in the app.
    """
    return pipeline(
        task="text2text-generation",
        model=SUMMARY_MODEL_NAME,
        tokenizer=SUMMARY_MODEL_NAME,
        device=-1
    )


sentiment_tokenizer, sentiment_model, sentiment_pipeline = load_sentiment_resources()


# ==================================================
# Helper Functions
# ==================================================
def build_model_input(headline: str, target: str, aspect: str) -> str:
    """
    Keep the inference format consistent with the training format.
    """
    return f"Headline: {headline} Target: {target} Aspect: {aspect}"


def manual_sentiment_inference(text: str) -> dict:
    """
    Manual model inference used as fallback if the Hugging Face pipeline fails.
    This version removes token_type_ids because DistilBERT does not use them.
    """
    inputs = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # DistilBERT does not accept token_type_ids.
    # Some tokenizers may still return it, so remove it before model inference.
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_id = int(torch.argmax(probabilities, dim=-1).item())
        confidence = float(probabilities[0][predicted_id].item())

    id2label = sentiment_model.config.id2label
    label = id2label.get(predicted_id, f"LABEL_{predicted_id}")

    return {
        "label": label.upper(),
        "score": confidence,
        "method": "Manual model inference fallback"
    }


def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment using Hugging Face pipeline first.
    If it fails, use manual inference to avoid app crash.
    """
    if sentiment_pipeline is not None:
        try:
            result = sentiment_pipeline(text)[0]
            return {
                "label": result["label"].upper(),
                "score": float(result["score"]),
                "method": "Hugging Face text-classification pipeline"
            }
        except Exception:
            return manual_sentiment_inference(text)

    return manual_sentiment_inference(text)


def generate_summary(text: str) -> tuple[str, float, str]:
    """
    Generate a short summary using T5 text2text-generation pipeline.
    """
    start_time = time.time()

    try:
        summary_pipeline = load_summary_pipeline()

        prompt = "summarize: " + text

        result = summary_pipeline(
            prompt,
            max_length=60,
            min_length=8,
            do_sample=False,
            truncation=True
        )

        summary = result[0]["generated_text"].strip()
        runtime = time.time() - start_time

        if not summary:
            summary = "The input is brief, so the original headline already functions as a concise summary."

        return summary, runtime, "Hugging Face text2text-generation pipeline"

    except Exception as error:
        runtime = time.time() - start_time
        fallback_summary = (
            "Summary generation is currently unavailable. "
            "The original news text can be reviewed directly."
        )
        return fallback_summary, runtime, f"Fallback used due to error: {str(error)}"


def generate_risk_note(label: str, confidence: float, aspect: str) -> str:
    """
    Generate a business-facing risk note based on the sentiment output.
    """
    label = label.upper()

    if confidence < 0.5:
        return (
            "The model confidence is relatively low. Manual review is recommended "
            "before using this result for financial interpretation."
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
            "Users may monitor whether the positive signal is supported by fundamentals, "
            "future guidance, or broader market conditions."
        )

    return (
        f"This news appears relatively neutral regarding {aspect}. "
        "Users may continue monitoring related updates before drawing strong conclusions."
    )


def format_sentiment_label(label: str) -> str:
    label = label.upper()

    if label == "POSITIVE":
        return "🟢 POSITIVE"
    if label == "NEGATIVE":
        return "🔴 NEGATIVE"
    return "🟡 NEUTRAL"


def get_example_data():
    return {
        "Positive earnings news": {
            "headline": "Apple reported stronger-than-expected quarterly earnings and raised its revenue guidance.",
            "target": "Apple",
            "aspect": "earnings and revenue guidance"
        },
        "Negative earnings news": {
            "headline": "Tesla shares fell after the company missed earnings expectations.",
            "target": "Tesla",
            "aspect": "earnings performance"
        },
        "Neutral governance news": {
            "headline": "The company announced a new board appointment on Monday.",
            "target": "The company",
            "aspect": "corporate governance"
        },
        "Forecast risk news": {
            "headline": "The retailer cut its annual forecast due to weak consumer spending.",
            "target": "The retailer",
            "aspect": "annual forecast"
        },
        "Regulatory risk news": {
            "headline": "The company faces a regulatory investigation over data privacy issues.",
            "target": "The company",
            "aspect": "regulatory investigation"
        }
    }


# ==================================================
# Main Interface
# ==================================================
st.title("📈 Financial News Risk Screening Tool")

st.write(
    "A deep-learning-based prototype for screening financial news sentiment "
    "and generating concise risk summaries for digital brokerage or wealth management users."
)

st.info(
    "**Final application structure:**  \n"
    "Pipeline 1: Financial sentiment classification using a fine-tuned DistilBERT model  \n"
    "Pipeline 2: Financial news summarization using a T5 text-to-text generation model"
)


# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    st.header("About This App")

    st.write(
        "This prototype is designed for a digital brokerage or wealth management context. "
        "It helps users quickly screen financial news and identify possible risk signals."
    )

    st.markdown("### Final Deployed Pipelines")
    st.markdown("**Pipeline 1:** Sentiment Classification")
    st.code(SENTIMENT_MODEL_NAME)

    st.markdown("**Pipeline 2:** Text-to-Text Summary Generation")
    st.code(SUMMARY_MODEL_NAME)

    st.markdown("### Models Explored")
    st.write("BERT, FinBERT, and DistilBERT were compared during model selection.")

    st.warning(
        "Disclaimer: This tool is for information screening only. "
        "It does not provide investment advice."
    )


# ==================================================
# Tabs
# ==================================================
tab1, tab2, tab3 = st.tabs(
    ["Analyze News", "Model Design", "Example Inputs"]
)


# ==================================================
# Tab 1: Analyze News
# ==================================================
with tab1:
    st.subheader("Enter Financial News Information")

    examples = get_example_data()

    selected_example = st.selectbox(
        "Optional: load a sample input",
        ["Custom input"] + list(examples.keys())
    )

    if selected_example != "Custom input":
        default_headline = examples[selected_example]["headline"]
        default_target = examples[selected_example]["target"]
        default_aspect = examples[selected_example]["aspect"]
    else:
        default_headline = ""
        default_target = ""
        default_aspect = ""

    col1, col2 = st.columns([2, 1])

    with col1:
        headline = st.text_area(
            "Headline or news text",
            value=default_headline,
            height=170,
            placeholder=(
                "Example: Apple reported stronger-than-expected quarterly earnings "
                "and raised its revenue guidance."
            )
        )

    with col2:
        target = st.text_input(
            "Target company / asset",
            value=default_target,
            placeholder="Example: Apple"
        )

        aspect = st.text_input(
            "Aspect",
            value=default_aspect,
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

            with st.spinner("Running sentiment classification..."):
                sentiment_start = time.time()
                sentiment_result = predict_sentiment(model_input)
                sentiment_runtime = time.time() - sentiment_start

            label = sentiment_result["label"]
            confidence = sentiment_result["score"]
            sentiment_method = sentiment_result["method"]

            with st.spinner("Generating short summary..."):
                summary, summary_runtime, summary_method = generate_summary(headline)

            risk_note = generate_risk_note(label, confidence, aspect)

            st.divider()
            st.subheader("Analysis Results")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Sentiment", format_sentiment_label(label))

            with metric_col2:
                st.metric("Confidence", f"{confidence:.2%}")

            with metric_col3:
                st.metric("Runtime", f"{sentiment_runtime:.3f}s")

            if confidence < 0.5:
                st.warning(
                    "Low-confidence result detected. Manual review is recommended."
                )

            st.markdown("### Short Summary")
            st.write(summary)

            st.markdown("### Risk Note")
            st.write(risk_note)

            with st.expander("Technical Details"):
                st.markdown("**Model input format:**")
                st.code(model_input)

                st.markdown("**Sentiment method used:**")
                st.write(sentiment_method)

                st.markdown("**Summary method used:**")
                st.write(summary_method)

                st.markdown("**Summary runtime:**")
                st.write(f"{summary_runtime:.3f}s")

            st.caption(
                "This output is generated for information screening only and should not be interpreted as financial advice."
            )


# ==================================================
# Tab 2: Model Design
# ==================================================
with tab2:
    st.subheader("Model and Pipeline Design")

    st.markdown(
        """
        The project uses a two-pipeline structure for the final Streamlit application:

        **Pipeline 1: Financial Sentiment Classification**  
        - Task: classify financial news as POSITIVE, NEUTRAL, or NEGATIVE  
        - Final model: fine-tuned DistilBERT  
        - Output: sentiment label and confidence score  

        **Pipeline 2: Financial News Summarization**  
        - Task: generate a concise summary of the input news text  
        - Model: T5-small text-to-text generation model  
        - Output: short summary for fast information screening  
        """
    )

    st.markdown("### Models Explored During Model Selection")

    model_table = {
        "Model": ["BERT", "FinBERT", "DistilBERT"],
        "Purpose": [
            "Baseline transformer model for text classification",
            "Finance-domain transformer model",
            "Lightweight transformer model selected for final deployment"
        ],
        "Deployment Status": [
            "Compared in experiments",
            "Compared in experiments",
            "Selected as final sentiment model"
        ]
    }

    st.table(model_table)

    st.markdown("### Final Workflow")

    st.code(
        """
Financial news input
        ↓
Format input as: Headline + Target + Aspect
        ↓
Pipeline 1: Fine-tuned DistilBERT sentiment classification
        ↓
Sentiment label + confidence score
        ↓
Pipeline 2: T5 text-to-text summary generation
        ↓
Short summary + business-facing risk note
        """
    )


# ==================================================
# Tab 3: Example Inputs
# ==================================================
with tab3:
    st.subheader("Sample Financial News Inputs")

    examples = get_example_data()

    for title, example in examples.items():
        with st.expander(title):
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
