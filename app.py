"""Streamlit interface for drawing and classifying handwritten digits."""

from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from torch import nn, optim
from torchvision import transforms


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pth"
DEVICE = torch.device("cpu")
ONLINE_LEARNING_RATE = 1e-5


class DigitCNN(nn.Module):
    """The same CNN architecture used by train.py."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


@st.cache_resource
def load_model() -> DigitCNN:
    """Load model weights once per Streamlit process."""
    model = DigitCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


@st.cache_resource
def load_online_optimizer(_model: DigitCNN) -> optim.Optimizer:
    """Create one small optimizer that keeps learning during the app session."""
    return optim.Adam(_model.parameters(), lr=ONLINE_LEARNING_RATE)


def learn_from_feedback(
    model: DigitCNN, optimizer: optim.Optimizer, image: torch.Tensor, label: int
) -> float:
    """Take one feedback-guided update and return the loss used for learning."""
    model.train()
    optimizer.zero_grad()
    logits = model(image)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([label], device=DEVICE))
    loss.backward()
    optimizer.step()
    model.eval()
    return loss.item()


def preprocess_canvas(canvas_data: np.ndarray) -> torch.Tensor:
    """Convert the canvas RGBA pixels into a normalized MNIST tensor."""
    rgba_image = Image.fromarray(canvas_data.astype("uint8"), mode="RGBA")
    grayscale_image = ImageOps.invert(rgba_image.convert("L"))
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return transform(grayscale_image).unsqueeze(0)


def apply_app_styles() -> None:
    """Apply the app's visual system without changing Streamlit behavior."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #f4f7fb;
            --muted: #91a2bb;
            --panel: #111c2d;
            --panel-strong: #16243a;
            --line: #253752;
            --blue: #3193ff;
            --violet: #8065ff;
        }

        .stApp {
            background: radial-gradient(circle at 12% 0%, #142640 0, #0a111d 42%, #080e17 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stAppViewContainer"] > .main { padding-top: 2rem; }
        .block-container { max-width: 1100px; padding-bottom: 3rem; }
        h1, h2, h3, p, label, button, [data-testid="stMarkdownContainer"] {
            font-family: 'DM Sans', sans-serif;
        }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
        h1 { font-size: clamp(2rem, 4vw, 3.15rem); margin: 0; }
        h2 { font-size: 1.35rem; }
        .eyebrow { color: #74b9ff; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
        .subtitle { color: var(--muted); font-size: 1.05rem; margin-top: .25rem; }
        .hero { align-items: center; display: flex; justify-content: space-between; gap: 2rem; margin-bottom: 1.5rem; }
        .hint { background: #10223a; border: 1px solid #1e426c; border-radius: 14px; color: #a9d5ff; max-width: 300px; padding: .85rem 1rem; font-size: .88rem; line-height: 1.4; }
        .panel { background: linear-gradient(145deg, rgba(22,36,58,.96), rgba(13,25,42,.96)); border: 1px solid var(--line); border-radius: 16px; padding: 1.1rem; min-height: 100%; }
        .panel-title { align-items: center; display: flex; gap: .65rem; color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 700; margin-bottom: .9rem; }
        .panel-title span { color: #64b1ff; }
        .result-empty { align-items: center; color: var(--muted); display: flex; justify-content: center; min-height: 365px; text-align: center; }
        .prediction-badge { background: #0f8c4f; border-radius: 9px; color: white; display: inline-block; float: right; font-size: .9rem; font-weight: 700; padding: .45rem .7rem; }
        .confidence-note { background: rgba(35,52,79,.72); border: 1px solid #2c4261; border-radius: 13px; color: var(--muted); margin-top: .8rem; padding: .8rem 1rem; }
        .confidence-note strong { color: #60b3ff; }
        .feedback { background: linear-gradient(100deg, rgba(19,34,54,.96), rgba(26,27,66,.96)); border: 1px solid var(--line); border-radius: 15px; margin-top: 1.25rem; padding: 1rem 1.2rem .9rem; }
        .feedback-title { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; }
        .feedback-copy { color: var(--muted); font-size: .9rem; margin-top: .2rem; }
        [data-testid="stCanvas"] { border-radius: 11px; overflow: hidden; }
        div.stButton > button { border: 1px solid #30435f; border-radius: 10px; font-weight: 700; min-height: 2.7rem; }
        div.stButton > button[kind="primary"] { background: linear-gradient(100deg, #2789f5, #3a9dff); border: 0; }
        div[data-baseweb="select"] > div { background: #182740; border-color: #334a6b; border-radius: 9px; }
        @media (max-width: 760px) {
            .hero { align-items: flex-start; flex-direction: column; gap: 1rem; }
            .hint { max-width: none; width: 100%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Digit Classifier", page_icon="8", layout="wide")
    apply_app_styles()
    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="eyebrow">Neural handwriting lab</div>
                <h1>Handwritten Digit Classifier</h1>
                <div class="subtitle">Draw a digit from 0 to 9 and let the model read it.</div>
            </div>
            <div class="hint">Tip: use a single, centered stroke style for the clearest prediction and confidence scores.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    if not MODEL_PATH.exists():
        st.error("Model weights are missing. Run `python train.py` first.")
        st.stop()

    try:
        model = load_model()
        optimizer = load_online_optimizer(model)
    except (RuntimeError, OSError) as error:
        st.error(f"Could not load model weights: {error}")
        st.stop()

    draw_column, result_column = st.columns([1, 1], gap="large")
    with draw_column:
        st.markdown('<div class="panel-title"><span>✎</span> Draw a digit</div>', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=18,
            stroke_color="#000000",
            background_color="#ffffff",
            height=300,
            width=300,
            drawing_mode="freedraw",
            display_toolbar=False,
            key=f"digit_canvas_{st.session_state.canvas_key}",
        )
        predict_column, clear_column = st.columns(2)
        with predict_column:
            predict_clicked = st.button("✦  Predict", type="primary", width="stretch")
        with clear_column:
            clear_clicked = st.button("↻  Clear", width="stretch")

    if clear_clicked:
        st.session_state.canvas_key += 1
        st.session_state.pop("last_image", None)
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("last_probabilities", None)
        st.rerun()

    with result_column:
        st.markdown('<div class="panel-title"><span>▥</span> Prediction result</div>', unsafe_allow_html=True)
        if predict_clicked:
            if canvas_result.image_data is None or not np.any(
                canvas_result.image_data[:, :, :3] < 245
            ):
                st.warning("Draw a digit on the canvas first.")
            else:
                image_tensor = preprocess_canvas(canvas_result.image_data).to(DEVICE)
                with torch.no_grad():
                    probabilities = torch.softmax(model(image_tensor), dim=1)[0].numpy()
                predicted_digit = int(np.argmax(probabilities))
                st.session_state.last_image = image_tensor.detach()
                st.session_state.last_prediction = predicted_digit
                st.session_state.last_probabilities = probabilities

        if "last_prediction" in st.session_state:
            probabilities = st.session_state.get("last_probabilities")
            if probabilities is not None:
                predicted_digit = st.session_state.last_prediction
                confidence_figure = px.bar(
                    x=list(range(10)),
                    y=probabilities,
                    labels={"x": "Digit", "y": "Confidence"},
                    range_y=[0, 1],
                )
                confidence_figure.update_layout(
                    xaxis=dict(dtick=1, title="Digit"),
                    yaxis=dict(title="Confidence"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9d7eb", family="DM Sans"),
                    margin=dict(l=5, r=5, t=5, b=5),
                    height=270,
                )
                st.markdown(
                    f'<div class="prediction-badge">✓ Predicted digit: {predicted_digit}</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(confidence_figure, width="stretch")
                st.markdown(
                    f'<div class="confidence-note">Model confidence: <strong>{predicted_digit} ({probabilities[predicted_digit]:.1%})</strong>.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="result-empty">Your prediction and confidence scores will appear here.</div>', unsafe_allow_html=True)

    if "last_image" in st.session_state:
        st.markdown(
            '<div class="feedback"><div class="feedback-title">◎ Teach the model</div><div class="feedback-copy">Choose the correct label to give the model a small reward-guided update. Learning lasts for this app session.</div></div>',
            unsafe_allow_html=True,
        )
        feedback_column, learn_column = st.columns([1, 1])
        with feedback_column:
            feedback_label = st.selectbox(
                "What digit did you draw?",
                options=list(range(10)),
                index=st.session_state.get("last_prediction", 0),
            )
        with learn_column:
            st.write("")
            learn_clicked = st.button("▣  Learn from this example", width="stretch")
        if learn_clicked:
            feedback_loss = learn_from_feedback(model, optimizer, st.session_state.last_image, feedback_label)
            st.success(f"Updated from your feedback: digit {feedback_label} (loss {feedback_loss:.4f}).")


if __name__ == "__main__":
    main()
