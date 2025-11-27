from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import pydicom
import warnings
import base64
import io

# Fix NumPy compatibility issue
import numpy

if not hasattr(numpy, "dtypes"):
    numpy.dtypes = numpy.dtype

from ultralytics import YOLO

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Configuration parameters"""

    MODEL_PATH = "best_metacarpal_model.pt"
    PIXEL_SPACING = 0.143  # mm per pixel (default if not in DICOM)
    IMG_SIZE = 640


# ============================================================================
# DICOM PROCESSING
# ============================================================================


class DICOMProcessor:
    """Handles DICOM file processing"""

    @staticmethod
    def read_dicom_from_bytes(dicom_bytes):
        """Read DICOM file from bytes and extract pixel spacing"""
        try:
            dicom = pydicom.dcmread(io.BytesIO(dicom_bytes))
            image = dicom.pixel_array

            # Try to get pixel spacing from DICOM metadata
            pixel_spacing = None
            if hasattr(dicom, "PixelSpacing"):
                pixel_spacing = float(dicom.PixelSpacing[0])
            elif hasattr(dicom, "ImagerPixelSpacing"):
                pixel_spacing = float(dicom.ImagerPixelSpacing[0])

            return image, pixel_spacing
        except Exception as e:
            print(f"❌ Error reading DICOM: {e}")
            return None, None

    @staticmethod
    def normalize_image(image):
        """Normalize image to 0-255 range"""
        if image is None:
            return None

        image = image.astype(np.float32)
        img_range = image.max() - image.min()

        if img_range == 0:
            return np.zeros_like(image, dtype=np.uint8)

        image = (image - image.min()) / img_range
        image = (image * 255).astype(np.uint8)
        return image

    @staticmethod
    def convert_to_rgb(image):
        """Convert grayscale to RGB"""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    @staticmethod
    def preprocess_for_yolo(dicom_bytes, output_size=640):
        """Preprocess DICOM image for YOLO inference"""
        image, pixel_spacing = DICOMProcessor.read_dicom_from_bytes(dicom_bytes)
        if image is None:
            return None, None

        # Normalize to 0-255
        image = DICOMProcessor.normalize_image(image)
        if image is None:
            return None, None

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # Convert to RGB
        image = DICOMProcessor.convert_to_rgb(image)

        # Resize with padding to maintain aspect ratio
        h, w = image.shape[:2]
        scale = output_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))

        # Pad to square
        pad_h = (output_size - new_h) // 2
        pad_w = (output_size - new_w) // 2
        image = cv2.copyMakeBorder(
            image,
            pad_h,
            output_size - new_h - pad_h,
            pad_w,
            output_size - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        # Use pixel spacing from DICOM or default
        if pixel_spacing is None:
            pixel_spacing = Config.PIXEL_SPACING
            print(
                f"⚠️  Pixel spacing not found in DICOM, using default: {pixel_spacing} mm/px"
            )

        return image, pixel_spacing


# ============================================================================
# BONE MEASUREMENT EXTRACTION
# ============================================================================


class BoneMeasurementExtractor:
    """Extract length, width, and cortical thickness from bone mask"""

    @staticmethod
    def extract_measurements(mask, pixel_spacing):
        """
        Extract L, W, T from segmentation mask

        Returns:
            length (mm), width (mm), cortical_thickness (mm)
        """
        if mask is None or mask.sum() == 0:
            return None, None, None

        # Binarize mask
        mask = (mask > 0.5).astype(np.uint8)

        # Get bone coordinates
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None, None, None

        # ===== CALCULATE LENGTH (L) =====
        # Length = vertical extent of bone
        min_y, max_y = coords[0].min(), coords[0].max()
        bone_length_pixels = max_y - min_y
        bone_length_mm = bone_length_pixels * pixel_spacing

        if bone_length_pixels < 10:  # Too small to measure
            return None, None, None

        # ===== CALCULATE WIDTH (W) AND CORTICAL THICKNESS (T) =====
        # Measure in the mid-diaphysis region (middle third of bone)
        mid_start = min_y + bone_length_pixels // 3
        mid_end = max_y - bone_length_pixels // 3

        widths = []
        cortical_thicknesses = []

        # Sample multiple horizontal slices in mid-diaphysis
        num_samples = max(10, min(30, (mid_end - mid_start) // 5))

        for y in np.linspace(mid_start, mid_end, num_samples, dtype=int):
            if y >= mask.shape[0]:
                continue

            row = mask[y, :]
            if row.sum() == 0:
                continue

            bone_pixels = np.where(row > 0)[0]
            if len(bone_pixels) < 2:
                continue

            # Width = distance from left edge to right edge
            left_edge = bone_pixels[0]
            right_edge = bone_pixels[-1]
            width_mm = (right_edge - left_edge) * pixel_spacing

            if width_mm < 1:  # Too narrow
                continue

            widths.append(width_mm)

            # ===== CORTICAL THICKNESS =====
            # Method: Measure outer 25% on each side as cortical bone
            bone_width_pixels = right_edge - left_edge
            cortical_region_pixels = max(1, int(bone_width_pixels * 0.25))

            # Cortical thickness (average of both sides)
            cortical_thickness_mm = cortical_region_pixels * pixel_spacing
            cortical_thicknesses.append(cortical_thickness_mm)

        if not widths or not cortical_thicknesses:
            return None, None, None

        # Return averages
        avg_width = np.mean(widths)
        avg_cortical = np.mean(cortical_thicknesses)

        return bone_length_mm, avg_width, avg_cortical


# ============================================================================
# BHI CALCULATOR
# ============================================================================


class BHICalculator:
    """Calculate Bone Health Index: BHI = πT(1−T/W)/(LW)^0.33"""

    @staticmethod
    def calculate_bhi(T, W, L):
        """
        Calculate BHI using formula: BHI = πT(1−T/W)/(LW)^0.33

        Args:
            T: Average cortical thickness (mm)
            W: Average bone width (mm)
            L: Average bone length (mm)

        Returns:
            BHI value
        """
        if T is None or W is None or L is None:
            return None

        if W == 0 or L == 0:
            return None

        try:
            # BHI = πT(1−T/W)/(LW)^0.33
            numerator = np.pi * T * (1 - T / W)
            denominator = (L * W) ** 0.33

            if denominator == 0:
                return None

            bhi = numerator / denominator
            return bhi

        except Exception as e:
            print(f"❌ Error calculating BHI: {e}")
            return None


# ============================================================================
# MAIN ANALYZER
# ============================================================================


class MetacarpalBHIAnalyzer:
    """Main class for detecting metacarpals and calculating BHI"""

    def __init__(self, model_path):
        """Initialize with trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"📦 Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.bone_names = ["2nd Metacarpal", "3rd Metacarpal", "4th Metacarpal"]
        print("✓ Model loaded successfully\n")

    def process_dicom_bytes(self, dicom_bytes, conf_threshold=0.25):
        """
        Process DICOM image from bytes and calculate BHI

        Args:
            dicom_bytes: DICOM file bytes
            conf_threshold: Detection confidence threshold

        Returns:
            Dictionary with results
        """
        print("=" * 70)
        print(f"PROCESSING DICOM")
        print("=" * 70)

        # Preprocess DICOM
        image, pixel_spacing = DICOMProcessor.preprocess_for_yolo(dicom_bytes)
        if image is None:
            return {"error": "Failed to process DICOM"}

        print(f"✓ Image preprocessed (pixel spacing: {pixel_spacing:.4f} mm/px)")

        # Run detection
        try:
            results = self.model(image, conf=conf_threshold, verbose=False)[0]
        except Exception as e:
            return {"error": f"Detection failed: {str(e)}"}

        boxes = results.boxes
        print(f"✓ Detected {len(boxes)} bones")

        if len(boxes) == 0:
            return {"error": "No bones detected"}

        # Check if masks are available
        if not hasattr(results, "masks") or results.masks is None:
            return {"error": "No segmentation masks found. Use a segmentation model."}

        # Extract measurements for each bone
        individual_bones = []
        all_lengths = []
        all_widths = []
        all_thicknesses = []

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bone_name = (
                self.bone_names[cls_id]
                if cls_id < len(self.bone_names)
                else f"Bone {cls_id}"
            )

            # Get segmentation mask
            mask = results.masks.data[i].cpu().numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

            # Extract measurements
            length, width, thickness = BoneMeasurementExtractor.extract_measurements(
                mask, pixel_spacing
            )

            if length is not None and width is not None and thickness is not None:
                individual_bones.append(
                    {
                        "name": bone_name,
                        "length": round(length, 2),
                        "width": round(width, 2),
                        "thickness": round(thickness, 2),
                        "confidence": round(conf, 4),
                    }
                )

                all_lengths.append(length)
                all_widths.append(width)
                all_thicknesses.append(thickness)

        # Calculate BHI using averages
        if len(all_lengths) > 0:
            avg_L = np.mean(all_lengths)
            avg_W = np.mean(all_widths)
            avg_T = np.mean(all_thicknesses)

            bhi = BHICalculator.calculate_bhi(avg_T, avg_W, avg_L)

            # Generate annotated image
            annotated_img = results.plot()
            _, buffer = cv2.imencode(".png", annotated_img)
            annotated_base64 = base64.b64encode(buffer).decode("utf-8")

            result = {
                "bones": individual_bones,
                "averages": {
                    "length": round(avg_L, 2),
                    "width": round(avg_W, 2),
                    "thickness": round(avg_T, 2),
                },
                "bhi": round(bhi, 4),
                "bones_measured": len(all_lengths),
                "pixel_spacing": pixel_spacing,
                "annotated_image": annotated_base64,
            }

            return result
        else:
            return {"error": "Insufficient measurements for BHI calculation"}


# Initialize analyzer globally
analyzer = None


def get_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = MetacarpalBHIAnalyzer(Config.MODEL_PATH)
    return analyzer


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": analyzer is not None})


@app.route("/api/analyze", methods=["POST"])
def analyze_dicom():
    """
    Analyze DICOM file and calculate BHI

    Expected: multipart/form-data with 'file' field containing DICOM file
    Returns: JSON with BHI results
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Read file bytes
    dicom_bytes = file.read()

    # Get analyzer
    try:
        analyzer_instance = get_analyzer()
    except Exception as e:
        return jsonify({"error": f"Model loading failed: {str(e)}"}), 500

    # Process DICOM
    result = analyzer_instance.process_dicom_bytes(dicom_bytes)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


@app.route("/")
def index():
    """Serve info page"""
    return """
    <html>
        <head><title>BHI Calculator API</title></head>
        <body style="font-family: Arial; padding: 40px; max-width: 800px; margin: 0 auto;">
            <h1>🩻 DICOM BHI Calculator API</h1>
            <p>Flask backend for calculating Bone Health Index from hand X-ray DICOM files.</p>
            
            <h2>Endpoints:</h2>
            <ul>
                <li><code>GET /api/health</code> - Health check</li>
                <li><code>POST /api/analyze</code> - Analyze DICOM file</li>
            </ul>
            
            <h2>Usage:</h2>
            <pre>
curl -X POST -F "file=@X244.dcm" http://localhost:5000/api/analyze
            </pre>
            
            <p>Model: <code>Output/best_metacarpal_model.pt</code></p>
        </body>
    </html>
    """


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Starting BHI Calculator API Server")
    print("=" * 70)
    print(f"Model path: {Config.MODEL_PATH}")
    print("Initializing model...")

    # Pre-load model
    try:
        get_analyzer()
        print("\n✓ Server ready!")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("Server will start but model needs to be fixed.\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
