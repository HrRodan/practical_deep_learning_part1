import gradio as gr
from fastai.vision.all import load_learner
from pathlib import Path
from PIL import Image
import __main__

# FastAI learner export requires the exact label_func definition in __main__
def is_cat(x):
    return x[0].isupper()

__main__.is_cat = is_cat

# Load exported fastai model relative to app path
model_path = Path(__file__).parent / "model.pkl"
learn = load_learner(model_path)

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

@spaces.GPU
def classify_image(img):
    if img is None:
        return None
    pred, pred_idx, probs = learn.predict(img)
    # FastAI vocab: [False, True] -> False is Dog, True is Cat
    return {
        "Dog": float(probs[0]),
        "Cat": float(probs[1])
    }

# Soft custom CSS for compact layout
css = """
.main-container { max-width: 600px; margin: 0 auto; padding: 15px; }
.title { text-align: center; color: #1E293B; margin-bottom: 4px; font-weight: 700; }
.subtitle { text-align: center; color: #64748B; margin-bottom: 20px; font-size: 0.95rem; }
"""

with gr.Blocks() as demo:
    with gr.Column(elem_classes="main-container"):
        gr.Markdown("<h2 class='title'>🐱 Cat vs Dog Classifier 🐶</h2>")
        gr.Markdown("<p class='subtitle'>Upload an image to determine whether it contains a cat or a dog.</p>")
        
        with gr.Row():
            image_input = gr.Image(
                type="pil", 
                label="Upload Image", 
                height=260,
                sources=["upload", "clipboard", "webcam"]
            )
            label_output = gr.Label(num_top_classes=2, label="Prediction")
            
        with gr.Row():
            clear_btn = gr.ClearButton(components=[image_input, label_output], value="Clear")
            submit_btn = gr.Button("Classify Image", variant="primary")
            
        submit_btn.click(fn=classify_image, inputs=image_input, outputs=label_output)
        image_input.change(fn=classify_image, inputs=image_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Soft(), css=css)
