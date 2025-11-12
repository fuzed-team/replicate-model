# Advanced Face Analysis - Replicate Model

Comprehensive face analysis model deployed on Replicate using Cog. Extracts 15+ facial attributes for AI-powered matching applications.

## Features

This model provides comprehensive facial analysis including:

1. **Face Detection** - High-confidence face detection with bounding boxes
2. **Face Embeddings** - 512-dimensional ArcFace embeddings for similarity matching
3. **Demographics** - Age estimation and gender classification
4. **Facial Landmarks** - 68-point facial landmarks
5. **Head Pose** - Yaw, pitch, and roll angles
6. **Quality Metrics** - Blur detection and illumination analysis
7. **Symmetry Analysis** - Facial symmetry score (0-1)
8. **Skin Tone** - Dominant skin color in CIELAB and hex format
9. **Emotion Detection** - 7 emotions (happy, sad, angry, fear, surprise, disgust, neutral)
10. **Facial Geometry** - Key facial proportion ratios

## Models Used

- **InsightFace (Buffalo_l)** - Face detection, embeddings, age, gender, landmarks, pose
- **DeepFace** - Emotion/expression recognition
- **OpenCV** - Image quality metrics (blur, illumination)
- **K-means Clustering** - Skin tone extraction

## Input

- **image**: Image file (JPEG, PNG) containing a face

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
# Test with a sample image
cog predict -i image=@path/to/test-image.jpg

# Example
cog predict -i image=@../test-images/sample-face.jpg
```

### Run Interactive Shell

```bash
cog run python
```

## Deployment to Replicate

### Prerequisites

- Replicate account ([replicate.com](https://replicate.com))
- Model created on Replicate platform

### Create Model

1. Visit https://replicate.com/create
2. Create new model: `<your-username>/face-analysis-advanced`
3. Set visibility (private or public)

### Deploy

```bash
# Login to Replicate
cog login

# Push model
cog push r8.im/<your-username>/face-analysis-advanced
```

First push takes 15-30 minutes to upload Docker layers and model weights.

### Test on Replicate

After deployment:
1. Visit your model page on Replicate
2. Upload a test image via the web UI
3. Verify the output matches expected format
4. Note the model version ID for API integration

## API Usage

### Node.js/TypeScript

```typescript
import Replicate from 'replicate';

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

// Run prediction
const output = await replicate.run(
  "your-username/face-analysis-advanced:version-id",
  {
    input: {
      image: "https://example.com/face.jpg"
      // or base64: "data:image/jpeg;base64,..."
    }
  }
);

console.log(output);
```

### Python

```python
import replicate

output = replicate.run(
    "your-username/face-analysis-advanced:version-id",
    input={"image": "https://example.com/face.jpg"}
)

print(output)
```

### cURL

```bash
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "version-id",
    "input": {
      "image": "https://example.com/face.jpg"
    }
  }' \
  https://api.replicate.com/v1/predictions
```

## Performance

- **Cold Start**: 30-60 seconds (first prediction after idle)
- **Warm Prediction**: 1-3 seconds (subsequent predictions)
- **GPU**: Nvidia T4 or better
- **Cost**: ~$0.00022 per prediction (estimated)

## Error Handling

The model returns structured error responses:

```json
{
  "face_detected": false,
  "error": "Error message describing the issue"
}
```

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

This model is a direct migration from the Python Flask service (`ai-service/app.py`). Key changes:

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

This model is for use in the AI matching application. Commercial use requires appropriate licensing for InsightFace and DeepFace models.
