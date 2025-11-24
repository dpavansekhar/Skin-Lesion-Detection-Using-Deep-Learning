from pathlib import Path
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os

# interface root (folder that contains app/, models/, static/, templates/)
INTERFACE_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR_PATH = INTERFACE_ROOT / "models"

# Flask app with correct template & static folders
app = Flask(
    __name__,
    template_folder=str(INTERFACE_ROOT / "templates"),
    static_folder=str(INTERFACE_ROOT / "static"),
)

# Import the core logic from sibling models/ package (relative import)
try:
    from ..models.model_utils import load_all_models, run_analysis_pipeline, CLASS_NAMES
except ImportError as e:
    print("CRITICAL ERROR: Failed to import model_utils. Ensure 'interface/models/model_utils.py' exists.")
    print("Error details:", e)

    class DummyUtils:
        CLASS_NAMES = ["Error Loading Models"]

        def load_all_models(self):
            return {}

        def run_analysis_pipeline(self, *args, **kwargs):
            raise ImportError("Model utilities failed to load.")

    dummy = DummyUtils()
    load_all_models = dummy.load_all_models
    run_analysis_pipeline = dummy.run_analysis_pipeline
    CLASS_NAMES = dummy.CLASS_NAMES

# --- Configuration ---

UPLOAD_FOLDER = INTERFACE_ROOT / "static" / "uploads"
GRADCAM_FOLDER = INTERFACE_ROOT / "static" / "gradcam"

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["GRADCAM_FOLDER"] = str(GRADCAM_FOLDER)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR_PATH, exist_ok=True)

# --- Global Model Loading ---
MODELS = load_all_models()
print("All models loaded successfully for inference.")

# --- Routes ---

@app.route("/", methods=["GET"])
def index():
    """Renders the main image upload page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and runs the selected ML prediction pipeline."""
    main_model = request.form.get("main_model")
    secondary_model = request.form.get("secondary_model", "N/A")

    if "file" not in request.files:
        return render_template("error.html", message="No image file provided in the request."), 400

    file = request.files["file"]
    if file.filename == "":
        return render_template("error.html", message="No selected file."), 400

    if file:
        filename = secure_filename(file.filename)
        unique_filename = f"{os.urandom(8).hex()}_{filename}"
        img_path = UPLOAD_FOLDER / unique_filename
        file.save(str(img_path))

        try:
            prediction_class, confidence, gradcam_filename, model_used = run_analysis_pipeline(
                MODELS, main_model, secondary_model, str(img_path), str(GRADCAM_FOLDER)
            )
        except Exception as e:
            if img_path.exists():
                img_path.unlink()

            error_message = (
                f"An error occurred during model analysis (Model: {main_model}/{secondary_model}). "
                f"This often means a model file is missing or the input data was incorrect. Error: {e}"
            )
            print("Prediction failed:", error_message)
            return render_template("error.html", message=error_message), 500

        image_url = url_for("static", filename=f"uploads/{unique_filename}")
        gradcam_url = url_for("static", filename=f"gradcam/{gradcam_filename}")

        return render_template(
            "results.html",              # matches templates/results.html
            prediction=prediction_class,
            confidence=f"{confidence:.2f}%",
            image_url=image_url,
            gradcam_url=gradcam_url,
            main_model=main_model,
            secondary_model=secondary_model if secondary_model != "N/A" else main_model,
            model_pipeline=model_used,
        )

if __name__ == "__main__":
    app.run(debug=True)
