"""
Replicate Model - Advanced Face Analysis

Comprehensive face analysis using InsightFace, DeepFace, and OpenCV.
Extracts 15+ facial attributes for AI-powered matching applications.

Input: Image file (JPEG, PNG) - supports URLs and file uploads
Output: JSON with face embeddings, age, gender, emotion, quality metrics, etc.

Best Practices:
- Type hints for better API documentation
- Proper error handling with detailed messages
- GPU acceleration with CPU fallback
- Input validation
- Efficient model loading
"""

from cog import BasePredictor, Input, Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
from sklearn.cluster import KMeans
import logging
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Load models into memory (called once on container start).
        This method runs when the container is initialized.

        Best Practice: Load heavy models here, not in predict()
        This ensures models are loaded once and reused across predictions.
        """
        logger.info("Initializing face analysis models...")

        try:
            # Initialize InsightFace with GPU support and CPU fallback
            # Providers are tried in order: CUDA first, then CPU
            self.face_app = FaceAnalysis(
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))

            logger.info("✓ InsightFace model loaded successfully!")
            logger.info("Note: DeepFace models will load on first emotion detection")

        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def predict(
        self,
        image: Path = Input(
            description="Input image file containing a face (JPEG, PNG, WEBP). "
                       "Supports both file uploads and URLs."
        )
    ) -> Dict[str, Any]:
        """
        Run comprehensive face analysis on input image.

        Best Practice: Return consistent error structure for failed predictions.
        All errors return {"face_detected": false, "error": "message"}

        Args:
            image: Path to image file (local or from URL)

        Returns:
            Dictionary with comprehensive facial attributes including:
            - face_detected (bool): Whether a face was found
            - embedding (list): 512-dimensional face embedding vector
            - age (int): Estimated age
            - gender (str): "male" or "female"
            - expression (dict): Emotion analysis with confidence scores
            - quality (dict): Image quality metrics (blur, illumination)
            - symmetry_score (float): Facial symmetry (0-1)
            - skin_tone (dict): Dominant skin color in LAB and hex
            - geometry (dict): Facial proportion ratios
            - error (str): Error message if face_detected is false
        """
        try:
            # Validate and read image
            logger.info(f"Processing image: {image}")
            img = cv2.imread(str(image))

            if img is None:
                logger.error("Failed to read image file")
                return {
                    "face_detected": False,
                    "error": "Invalid image format or unable to read file. "
                           "Supported formats: JPEG, PNG, WEBP"
                }

            # Validate image dimensions
            if img.shape[0] < 50 or img.shape[1] < 50:
                logger.error(f"Image too small: {img.shape}")
                return {
                    "face_detected": False,
                    "error": f"Image too small ({img.shape[1]}x{img.shape[0]}). "
                           "Minimum size: 50x50 pixels"
                }

            # Detect faces using InsightFace
            faces = self.face_app.get(img)

            if len(faces) == 0:
                logger.warning("No face detected in image")
                return {
                    "face_detected": False,
                    "error": "No face detected in image. "
                           "Please ensure image contains a clear, visible face."
                }

            # Use first detected face (highest confidence by default)
            face = faces[0]
            logger.info(f"Detected {len(faces)} face(s), using primary face with "
                       f"confidence: {face.det_score:.3f}")

            # Extract basic attributes
            embedding = face.embedding.tolist()
            bbox = face.bbox.tolist()
            confidence = float(face.det_score)

            # Extract age and gender from InsightFace
            age = int(face.age) if hasattr(face, 'age') else 25
            gender = "male" if (hasattr(face, 'gender') and face.gender == 1) else "female"

            # Extract landmarks
            landmarks_68 = self.extract_landmarks_68(face)

            # Extract pose
            pose = self.extract_pose(face)

            # Calculate quality metrics
            blur_score = self.calculate_blur_score(img, bbox)
            illumination = self.calculate_illumination(img, bbox)
            overall_quality = (blur_score + illumination) / 2.0

            # Calculate symmetry
            symmetry_score = self.calculate_symmetry_score(landmarks_68)

            # Extract skin tone
            skin_tone_lab = self.extract_skin_tone(img, bbox)
            hex_color = self.lab_to_hex(skin_tone_lab)

            # Detect emotion
            dominant_emotion, emotion_scores = self.detect_emotion(img, bbox)

            # Calculate geometry ratios
            geometry = self.calculate_geometry_ratios(landmarks_68)

            logger.info(
                f"✓ Analysis complete: age={age}, gender={gender}, "
                f"expression={dominant_emotion}, quality={overall_quality:.2f}"
            )

            return {
                "face_detected": True,
                "embedding": embedding,
                "bbox": bbox,
                "confidence": confidence,
                "age": age,
                "gender": gender,
                "landmarks_68": landmarks_68,
                "pose": pose,
                "quality": {
                    "blur_score": blur_score,
                    "illumination": illumination,
                    "overall": overall_quality
                },
                "symmetry_score": symmetry_score,
                "skin_tone": {
                    "dominant_color_lab": skin_tone_lab,
                    "hex": hex_color
                },
                "expression": {
                    "dominant": dominant_emotion,
                    "confidence": float(emotion_scores.get(dominant_emotion, 0.5)),
                    "emotions": emotion_scores
                },
                "geometry": geometry
            }

        except Exception as e:
            logger.error(f"Error in face analysis: {str(e)}", exc_info=True)
            return {
                "face_detected": False,
                "error": str(e)
            }

    # ===== HELPER METHODS =====

    def extract_landmarks_68(self, face) -> Optional[List[List[float]]]:
        """
        Extract 68-point facial landmarks from InsightFace face object.

        Args:
            face: InsightFace face detection object

        Returns:
            List of [x, y] coordinates or None if not available
        """
        # InsightFace provides 106 landmarks, we use first 68 for compatibility
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            return face.landmark_2d_106[:68].tolist()
        elif hasattr(face, 'kps') and face.kps is not None:
            # Fallback to 5-point landmarks if 106 not available
            return face.kps.tolist()
        return None

    def extract_pose(self, face) -> Dict[str, float]:
        """
        Extract head pose angles (yaw, pitch, roll) from face object.

        Args:
            face: InsightFace face detection object

        Returns:
            Dictionary with yaw, pitch, roll angles in degrees
        """
        pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        if hasattr(face, 'pose') and face.pose is not None:
            pose_array = face.pose if isinstance(face.pose, (list, np.ndarray)) else [0, 0, 0]
            pose = {
                "yaw": float(pose_array[0]) if len(pose_array) > 0 else 0.0,
                "pitch": float(pose_array[1]) if len(pose_array) > 1 else 0.0,
                "roll": float(pose_array[2]) if len(pose_array) > 2 else 0.0
            }

        return pose

    def calculate_symmetry_score(self, landmarks: Optional[List]) -> float:
        """
        Calculate facial symmetry by comparing left vs right features.

        Args:
            landmarks: List of facial landmark coordinates

        Returns:
            Symmetry score from 0.0 to 1.0 (1.0 = perfect symmetry)
        """
        if landmarks is None or len(landmarks) < 68:
            return 0.75  # Default moderate symmetry if landmarks unavailable

        landmarks = np.array(landmarks)

        # Split landmarks into left and right halves
        left_half = landmarks[:34]
        right_half = landmarks[34:68]

        # Mirror right half horizontally
        mirrored_right = np.copy(right_half)
        center_x = np.mean(landmarks[:, 0])
        mirrored_right[:, 0] = 2 * center_x - mirrored_right[:, 0]

        # Calculate average distance between mirrored halves
        if len(left_half) != len(mirrored_right):
            min_len = min(len(left_half), len(mirrored_right))
            left_half = left_half[:min_len]
            mirrored_right = mirrored_right[:min_len]

        distance = np.mean(np.linalg.norm(left_half - mirrored_right, axis=1))

        # Normalize: typical face width is ~200px, max asymmetry ~50px
        symmetry = 1.0 - min(distance / 50.0, 1.0)

        return float(max(symmetry, 0.0))

    def extract_skin_tone(self, image, bbox):
        """
        Extract dominant skin color using K-means clustering in CIELAB color space
        Returns: [L, a, b] values in CIELAB
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Extract face region
            face_region = image[y1:y2, x1:x2]

            if face_region.size == 0:
                return [65.0, 10.0, 20.0]  # Default skin tone

            # Convert to LAB color space (perceptually uniform)
            lab_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)

            # Reshape for K-means
            pixels = lab_image.reshape(-1, 3).astype(np.float32)

            # Use K-means to find 3 dominant colors
            kmeans = KMeans(n_clusters=3, random_state=0, n_init=10, max_iter=100)
            kmeans.fit(pixels)

            # Get the dominant color (assume largest cluster)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_cluster = labels[np.argmax(counts)]
            dominant_color = kmeans.cluster_centers_[dominant_cluster]

            return dominant_color.tolist()

        except Exception as e:
            logger.warning(f"Error extracting skin tone: {e}")
            return [65.0, 10.0, 20.0]  # Default skin tone

    def lab_to_hex(self, lab_color):
        """Convert LAB color to hex string"""
        try:
            # Create a 1x1 LAB image and convert to BGR
            lab_pixel = np.uint8([[lab_color]])
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
            b, g, r = bgr_pixel[0][0]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            return hex_color
        except:
            return "#d4a373"  # Default skin color

    def calculate_blur_score(self, image, bbox):
        """
        Detect image blur using Laplacian variance
        Returns: 0.0-1.0 (1.0 = sharp, 0.0 = very blurry)
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            face_region = image[y1:y2, x1:x2]

            if face_region.size == 0:
                return 0.5

            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            # Normalize: variance > 500 is sharp, < 100 is blurry
            blur_score = min(variance / 500.0, 1.0)

            return float(blur_score)

        except Exception as e:
            logger.warning(f"Error calculating blur: {e}")
            return 0.5

    def calculate_illumination(self, image, bbox):
        """
        Check lighting quality using histogram analysis
        Returns: 0.0-1.0 (1.0 = well-lit, 0.0 = poor lighting)
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            face_region = image[y1:y2, x1:x2]

            if face_region.size == 0:
                return 0.5

            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # Calculate brightness statistics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)

            # Good lighting: mean around 100-150, std > 30
            contrast_score = min(std_brightness / 50.0, 1.0)
            brightness_score = 1.0 - abs(mean_brightness - 130.0) / 130.0

            illumination = contrast_score * max(brightness_score, 0.0)

            return float(max(illumination, 0.0))

        except Exception as e:
            logger.warning(f"Error calculating illumination: {e}")
            return 0.5

    def detect_emotion(self, image, bbox):
        """
        Detect facial expression using DeepFace
        Returns: (dominant_emotion, emotion_scores_dict)
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            face_region = image[y1:y2, x1:x2]

            if face_region.size == 0:
                return "neutral", {"neutral": 1.0}

            # Analyze emotion
            result = DeepFace.analyze(
                face_region,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            emotions = result[0]['emotion']
            dominant = result[0]['dominant_emotion']

            # Normalize emotion scores to 0-1
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/100.0 for k, v in emotions.items()}

            return dominant, emotions

        except Exception as e:
            logger.warning(f"Error detecting emotion: {e}")
            return "neutral", {"neutral": 1.0}

    def calculate_geometry_ratios(self, landmarks):
        """
        Calculate facial proportions from landmarks
        Returns: Dictionary of key ratios
        """
        if landmarks is None or len(landmarks) < 68:
            # Return default ratios if landmarks unavailable
            return {
                "face_width_height_ratio": 0.75,
                "eye_spacing_face_width": 0.42,
                "jawline_width_face_width": 0.68,
                "nose_width_face_width": 0.25
            }

        try:
            landmarks = np.array(landmarks)

            # Calculate key facial dimensions (68-point landmark indices)
            # Face outline: 0-16 (left to right jawline)
            face_width = np.linalg.norm(landmarks[16] - landmarks[0])

            # Face height: chin (8) to forehead (approximate from eye level)
            if len(landmarks) > 27:
                face_height = np.linalg.norm(landmarks[8] - landmarks[27])
            else:
                face_height = face_width * 1.3  # Approximate ratio

            # Eye spacing: right eye outer corner (45) to left eye outer corner (36)
            if len(landmarks) > 45:
                eye_spacing = np.linalg.norm(landmarks[45] - landmarks[36])
            else:
                eye_spacing = face_width * 0.42

            # Jawline width (approximate from lower face)
            if len(landmarks) > 14:
                jawline_width = np.linalg.norm(landmarks[14] - landmarks[2])
            else:
                jawline_width = face_width * 0.68

            # Nose width (approximate from landmarks)
            if len(landmarks) > 35:
                nose_width = np.linalg.norm(landmarks[35] - landmarks[31])
            else:
                nose_width = face_width * 0.25

            return {
                "face_width_height_ratio": float(face_width / max(face_height, 1)),
                "eye_spacing_face_width": float(eye_spacing / max(face_width, 1)),
                "jawline_width_face_width": float(jawline_width / max(face_width, 1)),
                "nose_width_face_width": float(nose_width / max(face_width, 1))
            }

        except Exception as e:
            logger.warning(f"Error calculating geometry ratios: {e}")
            return {
                "face_width_height_ratio": 0.75,
                "eye_spacing_face_width": 0.42,
                "jawline_width_face_width": 0.68,
                "nose_width_face_width": 0.25
            }
