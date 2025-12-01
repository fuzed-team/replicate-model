# Advanced Face Analysis - Replicate Model

Comprehensive face analysis model deployed on Replicate using Cog. Extracts 15+
facial attributes for AI-powered matching applications.

## Features

This model provides comprehensive facial analysis including:

1. **Face Detection** - High-confidence face detection with bounding boxes
2. **Face Embeddings** - 512-dimensional ArcFace embeddings for similarity
   matching
3. **Demographics** - Age estimation and gender classification
4. **Facial Landmarks** - 68-point facial landmarks
5. **Head Pose** - Yaw, pitch, and roll angles
6. **Quality Metrics** - Blur detection and illumination analysis
7. **Symmetry Analysis** - Facial symmetry score (0-1)
8. **Skin Tone** - Dominant skin color in CIELAB and hex format
9. **Emotion Detection** - 7 emotions (happy, sad, angry, fear, surprise,
   disgust, neutral)
10. **Facial Geometry** - Key facial proportion ratios

## Models Used

- **InsightFace (Buffalo_l)** - Face detection, embeddings, age, gender,
  landmarks, pose
- **DeepFace** - Emotion/expression recognition
- **OpenCV** - Image quality metrics (blur, illumination)
- **K-means Clustering** - Skin tone extraction

## Input

- **image**: Image file (JPEG, PNG) containing a face
- **analysis_level** (optional): Analysis depth mode (default: "full")
  - `"health"` - Ultra-fast health check (<100ms, no image processing). Perfect
    for keep-warm pings.
  - `"full"` - Complete analysis with all 15+ features (~3-4 seconds)

## Output

JSON object with comprehensive facial attributes:

```json
{
  "face_detected": true,
  "embedding": [512 floats],
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.99,
  "age": 25,
  "gender": "male",
  "landmarks_68": [[x, y], ...],
  "pose": {"yaw": 5.2, "pitch": -2.1, "roll": 0.8},
  "quality": {
    "blur_score": 0.85,
    "illumination": 0.75,
    "overall": 0.80
  },
  "symmetry_score": 0.88,
  "skin_tone": {
    "dominant_color_lab": [65, 10, 20],
    "hex": "#d4a373"
  },
  "expression": {
    "dominant": "happy",
    "confidence": 0.85,
    "emotions": {
      "happy": 0.85,
      "neutral": 0.10,
      "sad": 0.02,
      "angry": 0.01,
      "surprise": 0.01,
      "fear": 0.005,
      "disgust": 0.005
    }
  },
  "geometry": {
    "face_width_height_ratio": 0.75,
    "eye_spacing_face_width": 0.42,
    "jawline_width_face_width": 0.68,
    "nose_width_face_width": 0.25
  }
}
```

### Health Check Mode Output

Ultra-fast health check (no image processing, <100ms):

```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Model is warm and ready for predictions"
}
```

### Error Response

If no face is detected:

```json
{
  "face_detected": false,
  "error": "No face detected in image"
}
```

## Local Development

### Prerequisites

- Docker installed and running
- [Cog](https://github.com/replicate/cog) installed

### Install Cog

```bash
# macOS/Linux
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog

# Windows (WSL2 required)
wsl
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog
```

### Build Container

```bash
cd replicate-model
cog build
```

First build takes 10-20 minutes to download and install all dependencies.

### Test Locally

```bash
# Full analysis (default)
cog predict -i image=@path/to/test-image.jpg

# Or explicitly specify full mode
cog predict -i image=@path/to/test-image.jpg -i analysis_level=full

# Health check (ultra-fast, no processing)
cog predict -i image=@path/to/test-image.jpg -i analysis_level=health

# Example with actual file
cog predict -i image=@../test-images/sample-face.jpg -i analysis_level=full
```

### Run Interactive Shell

```bash
cog run python
```

## Deployment to Replicate

### Prerequisites

- Docker installed and running
- [Cog](https://github.com/replicate/cog) installed (see installation above)
- Replicate account ([replicate.com](https://replicate.com))
- Replicate API token from https://replicate.com/account/api-tokens

### Deployment Steps

#### 1. Open WSL2 Terminal (Windows) or Terminal (Mac/Linux)

**Windows:**

```bash
wsl
```

**Mac/Linux:** Use your regular terminal

#### 2. Navigate to Project Directory

```bash
cd /mnt/d/Nemo/Others/side-projects/ai-matching/replicate-model  # Windows WSL2
# or
cd ~/path/to/ai-matching/replicate-model  # Mac/Linux
```

#### 3. Set Replicate API Token

```bash
export REPLICATE_API_TOKEN=r8_your_token_here
```

To make it permanent:

```bash
echo 'export REPLICATE_API_TOKEN=r8_your_token_here' >> ~/.bashrc
source ~/.bashrc
```

#### 4. Build Model Container (First Time)

```bash
cog build
```

⏱️ **First build takes 15-30 minutes** (downloads CUDA, dependencies, models) ⚡
**Subsequent builds are much faster** (2-5 min with cache)

#### 5. Test Locally (Recommended)

```bash
cog predict -i image=@path/to/test-face.jpg
```

#### 6. Create a Model on Replicate

1. Go to [replicate.com/create](https://replicate.com/create)
2. Enter a name for your model (e.g., `face-analysis`)
3. Set visibility to "Public" or "Private"
4. Click "Create"

#### 7. Push to Replicate

```bash
cog push r8.im/your-username/your-model-name
```

Replace `your-username` with your Replicate username and `your-model-name` with
the name you just created.

This will:

- Build and upload Docker image (~5-10 min)
- Create a new model version
- Provide URL to access your model

#### 8. Verify on Replicate

1. Visit `https://replicate.com/your-username/your-model-name`
2. Upload a test image via the web UI
3. Verify the output
4. Note the version ID for API use

### Troubleshooting Deployment

**Docker not accessible in WSL2:**

1. Open Docker Desktop
2. Settings → Resources → WSL Integration
3. Enable for "Ubuntu" distribution
4. Restart WSL2: `wsl --shutdown` (from PowerShell)

**Cog command not found:**

```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog
```

**Out of disk space:**

```bash
docker system prune -a  # Clean Docker cache
```

## API Usage

### Node.js/TypeScript

````typescript
import Replicate from "replicate";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

// Full analysis (default - all features)
const fullOutput = await replicate.run(
  "your-username/your-model-name:version-id",
  {
    input: {
      image: "https://example.com/face.jpg",
      analysis_level: "full", // Optional, default is "full"
    },
  },
);

// Health check (ultra-fast keep-warm ping)
const healthCheck = await replicate.run(
  "your-username/your-model-name:version-id",
  {
    input: {
      image: "https://example.com/any-image.jpg", // Image not processed in health mode
      analysis_level: "health", // <100ms, no processing
    },
  },
);

console.log(fullOutput);


## Error Handling

The model returns structured error responses:

```json
{
  "face_detected": false,
  "error": "Error message describing the issue"
}
````

Common errors:

- "No face detected in image" - Image contains no detectable faces
- "Invalid image format or unable to read file" - Corrupt or unsupported image
- Other errors include stack trace for debugging

## Limitations

- **Single Face**: Analyzes only the first detected face (highest confidence)
- **Quality Threshold**: Best results with well-lit, frontal faces
- **Image Size**: Optimal face size: 200x200 to 640x640 pixels
- **File Size**: Replicate has input file size limits (check platform docs)

## Project Structure

```
replicate-model/
├── cog.yaml          # Cog configuration (dependencies, Python version)
├── predict.py        # Main prediction logic (Predictor class)
└── README.md        # This file
```

## Migration from Flask Service

This model is a direct migration from the Python Flask service
(`ai-service/app.py`). Key changes:

- ✅ Removed Flask/CORS/auth code (handled by Replicate)
- ✅ Changed from multipart/base64 input to Cog's `Path` input
- ✅ Added GPU support (`CUDAExecutionProvider`)
- ✅ Returns dict directly (Cog handles JSON serialization)
- ✅ All 15+ features preserved and identical output format

## Troubleshooting

### Build Fails

```bash
# Check Docker is running
docker info

# Clear Docker cache
docker system prune -a

# Rebuild
cog build
```

### Prediction Errors

```bash
# Check logs
cog predict -i image=@test.jpg

# Run interactive Python shell
cog run python
>>> from predict import Predictor
>>> p = Predictor()
>>> p.setup()
```

### Slow Predictions

- Cold starts (30-60s) are normal after idle periods
- Keep model warm with periodic health check predictions
- Use Replicate's deployment features for guaranteed warm instances

## Support

- **Replicate Docs**: https://replicate.com/docs
- **Cog Docs**: https://github.com/replicate/cog
- **InsightFace**: https://insightface.ai/
- **DeepFace**: https://github.com/serengil/deepface

## License

This model is for use in the AI matching application. Commercial use requires
appropriate licensing for InsightFace and DeepFace models.
