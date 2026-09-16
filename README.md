# Handwritten Digit Classifier

A beginner-friendly PyTorch CNN that learns to recognize handwritten digits from the MNIST dataset. The included Streamlit app provides a canvas for drawing a digit and shows the model's prediction and confidence for every class.

## Setup

Use Python 3.10 or newer, then install the dependencies from the project folder:

```bash
python -m pip install -r requirements.txt
```

The dependency versions are pinned for compatibility with Streamlit Community Cloud's current Python runtime, including a prebuilt Pillow wheel for Python 3.14.

## Train the model

Run the training script:

```bash
python train.py
```

The script downloads MNIST into the local `data/` folder, trains for 9 epochs, prints training and validation metrics after every epoch, reports final test accuracy, and saves weights to `model.pth`. With the supplied CNN and settings, test accuracy should typically exceed 98%.

## Run the Streamlit app

After training, start the app:

```bash
streamlit run app.py
```

Open the displayed local URL, draw a digit, and select **Predict**. **Clear** resets the drawing canvas.

After predicting, use **Teach the model** to select the correct digit and submit feedback. The app performs one low-learning-rate online update from that labeled example. This is feedback-driven online learning rather than autonomous reinforcement learning: the model never trains on its own prediction, which would reinforce mistakes. Updates persist for the current app process/session; retrain or save a new model file if you want to keep them permanently.

## Model architecture

The model has two convolutional blocks, each with ReLU activation and 2x2 max pooling. The resulting features pass through a 128-unit fully connected layer and a 10-class output layer. Training uses raw logits with `CrossEntropyLoss`; the app applies softmax to show class probabilities.

## Final test accuracy

The included `model.pth` was trained for 9 epochs on CPU and achieved **99.05% test accuracy**. Re-running training may produce a slightly different result depending on the PyTorch version and runtime hardware.
