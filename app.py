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


def main() -> None:
    st.set_page_config(page_title="Digit Classifier", page_icon="8", layout="centered")
    st.title("Handwritten Digit Classifier")
    st.write("Draw a digit from 0 to 9, then ask the model to classify it.")

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

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#ffffff",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"digit_canvas_{st.session_state.canvas_key}",
    )

    predict_column, clear_column = st.columns(2)
    with predict_column:
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)
    with clear_column:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.canvas_key += 1
        st.session_state.pop("last_image", None)
        st.session_state.pop("last_prediction", None)
        st.rerun()

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
            st.subheader(f"Predicted digit: {predicted_digit}")
            confidence_figure = px.bar(
                x=list(range(10)),
                y=probabilities,
                labels={"x": "Digit", "y": "Confidence"},
                range_y=[0, 1],
            )
            confidence_figure.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(confidence_figure, use_container_width=True)

    if "last_image" in st.session_state:
        st.divider()
        st.subheader("Teach the model")
        st.caption(
            "Choose the correct label to give the model a small reward-guided update. "
            "Learning lasts for this app session."
        )
        feedback_label = st.selectbox(
            "What digit did you draw?",
            options=list(range(10)),
            index=st.session_state.get("last_prediction", 0),
        )
        learn_clicked = st.button("Learn from this example", use_container_width=True)
        if learn_clicked:
            feedback_loss = learn_from_feedback(
                model,
                optimizer,
                st.session_state.last_image,
                feedback_label,
            )
            st.success(
                f"Updated from your feedback: digit {feedback_label} "
                f"(loss {feedback_loss:.4f})."
            )


if __name__ == "__main__":
    main()
