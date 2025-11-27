from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

app = Flask(__name__)
CORS(app)


# Define your model architecture (matching your training code exactly)
class AgeGenderModel(nn.Module):
    def __init__(self):
        super().__init__()

        # EfficientNet-B3 backbone (1536 features)
        self.backbone = models.efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features  # 1536 for B3
        self.backbone.classifier = nn.Identity()

        # Attention mechanism - matches your training code
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 8),  # 1536 -> 192
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 8, in_features),  # 192 -> 1536
            nn.Sigmoid(),
        )

        # Shared layers - matches your training code
        self.shared = nn.Sequential(
            nn.Linear(in_features, 1024),  # 1536 -> 1024
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),  # 1024 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Age prediction head - matches your training code
        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Gender prediction head - matches your training code
        self.gender_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        att_weights = self.attention(features)
        features = features * att_weights
        shared = self.shared(features)
        age = self.age_head(shared).squeeze(-1)
        gender = self.gender_head(shared)
        return age, gender


# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
age_model = AgeGenderModel().to(device)
gender_model = AgeGenderModel().to(device)

# Load weights
try:
    age_model.load_state_dict(torch.load("best_Age_mae.pth", map_location=device))
    age_model.eval()
    print("Age model loaded successfully")
except Exception as e:
    print(f"Error loading age model: {e}")

try:
    gender_model.load_state_dict(torch.load("best_gender_Acc.pth", map_location=device))
    gender_model.eval()
    print("Gender model loaded successfully")
except Exception as e:
    print(f"Error loading gender model: {e}")

# Image preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_and_process_image(file):
    """
    Load image from various formats (DICOM, PNG, JPEG, JPG)
    Returns: PIL Image for processing, base64 string for display
    """
    filename = file.filename.lower()

    try:
        # Handle DICOM files
        if filename.endswith(".dcm") or filename.endswith(".dicom"):
            file_bytes = file.read()
            dicom_data = pydicom.dcmread(io.BytesIO(file_bytes))

            # Extract pixel array
            image_array = dicom_data.pixel_array.astype(np.float32)

            # Apply VOI LUT if available (window/level adjustments)
            try:
                image_array = apply_voi_lut(image_array, dicom_data)
            except:
                pass

            # Normalize to 0-255 range
            image_min = image_array.min()
            image_max = image_array.max()
            if image_max > image_min:
                image_array = (image_array - image_min) / (image_max - image_min) * 255
            image_array = image_array.astype(np.uint8)

            # Convert to PIL Image
            pil_image = Image.fromarray(image_array).convert("L")

        # Handle standard image formats
        else:
            pil_image = Image.open(file.stream).convert("L")

        # Create base64 for display
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return pil_image, img_base64

    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image from request
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]

        # Validate file extension
        filename = image_file.filename.lower()
        allowed_extensions = [".dcm", ".dicom", ".png", ".jpg", ".jpeg"]

        if not any(filename.endswith(ext) for ext in allowed_extensions):
            return jsonify(
                {
                    "error": "Unsupported file format. Please upload DICOM (.dcm), PNG, or JPEG files."
                }
            ), 400

        # Load and process image
        pil_image, img_base64 = load_and_process_image(image_file)

        # Apply CLAHE preprocessing (matching training)
        image_np = np.array(pil_image)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        image_np = clahe.apply(image_np)
        processed_image = Image.fromarray(image_np)

        # Transform for model
        image_tensor = transform(processed_image).unsqueeze(0).to(device)

        # Predict age
        with torch.no_grad():
            age_output, _ = age_model(image_tensor)
            predicted_age_norm = age_output.item()

        # Predict gender
        with torch.no_grad():
            _, gender_output = gender_model(image_tensor)
            gender_probs = torch.softmax(gender_output, dim=1)
            predicted_gender_idx = torch.argmax(gender_probs, dim=1).item()
            gender_confidence = gender_probs[0][predicted_gender_idx].item()

        # Denormalize age
        MIN_AGE_MONTHS = 0
        MAX_AGE_MONTHS = 228
        predicted_age_months = (
            predicted_age_norm * (MAX_AGE_MONTHS - MIN_AGE_MONTHS) + MIN_AGE_MONTHS
        )
        predicted_age_years = predicted_age_months / 12

        # Map gender index to label
        gender_labels = ["Female", "Male"]
        predicted_gender = gender_labels[predicted_gender_idx]

        return jsonify(
            {
                "age": round(predicted_age_years, 1),
                "age_months": round(predicted_age_months, 1),
                "gender": predicted_gender,
                "gender_confidence": round(gender_confidence * 100, 2),
                "normalized_age": round(predicted_age_norm, 4),
                "image_base64": f"data:image/png;base64,{img_base64}",
            }
        )

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
