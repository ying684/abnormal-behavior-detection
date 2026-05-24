# Contributing Guide

Thank you for your interest in contributing to the Abnormal Behavior Detection System!

## 🏗️ Project Structure

```
abnormal-behavior-detection/
├── api/                 # FastAPI backend
├── web/                 # Streamlit frontend
├── core/                # Core ML pipeline
├── config/              # Configuration
├── scripts/             # Training scripts
├── tests/               # Unit tests
├── weights/             # Pre-trained models
├── data/                # Data directory
└── notebooks/           # Jupyter notebooks
```

## 🔧 Development Setup

```bash
# Clone repository
git clone <repo-url>
cd abnormal-behavior-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies with dev tools
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

## 📝 Code Style

We follow PEP 8 standards:

```bash
# Format code
black .

# Check style
flake8 --max-line-length=100

# Run tests
pytest tests/
pytest --cov=core tests/  # With coverage
```

## 🚀 Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new features
   - Update docstrings

3. **Test locally**
   ```bash
   pytest tests/
   python run_web.py  # or run_api.py
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat: add new detection feature"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 💡 Types of Contributions

### Bug Fixes
- Identify and fix bugs
- Add regression tests
- Update documentation

### Features
- Implement new behaviors/detections
- Improve model accuracy
- Add new visualization options

### Documentation
- Improve README
- Add code comments
- Create tutorials

### Testing
- Expand test coverage
- Test edge cases
- Improve test reliability

## 📋 Pull Request Process

1. **Update documentation** if your change affects usage
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update CHANGELOG** if applicable
5. **Request review** from maintainers

### PR Title Format
```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Scopes: api, web, core, config, scripts
```

Example:
```
feat(core): add new behavior detection model
fix(api): resolve video upload timeout issue
docs(readme): improve setup instructions
```

## 🧪 Testing Guidelines

```python
# tests/test_detection.py
import pytest
from core.detection.pose_estimator import PoseEstimator

def test_pose_estimation():
    """Test pose estimation on sample image"""
    estimator = PoseEstimator("weights/detection/yolov8s-pose.pt")
    # Add test logic
    assert estimator is not None

def test_edge_case():
    """Test edge case handling"""
    # Test with empty input, None, etc.
    pass
```

## 📚 Documentation Standards

```python
def detect_behavior(video_path: str, confidence: float = 0.4) -> dict:
    """
    Detect abnormal behaviors in video.
    
    Args:
        video_path: Path to video file
        confidence: Confidence threshold (0-1)
    
    Returns:
        Dictionary with detection results:
        {
            'behaviors': [list of detected behaviors],
            'timestamps': [frame timestamps],
            'confidence_scores': [confidence values]
        }
    
    Raises:
        FileNotFoundError: If video file not found
        ValueError: If confidence not in range [0, 1]
    
    Example:
        >>> results = detect_behavior('video.mp4')
        >>> print(results['behaviors'])
    """
    pass
```

## 🎯 Performance Guidelines

- Process videos within 25ms per frame (GPU)
- Keep model size < 500MB
- Memory usage < 4GB for inference
- API response time < 5 seconds

## 🔐 Security

- Never commit credentials or API keys
- Use environment variables for secrets
- Validate all user inputs
- Check for vulnerabilities: `safety check`

## 📞 Need Help?

- Check existing issues and discussions
- Read documentation in `docs/`
- Ask in PR comments
- Email maintainers

## ✅ Checklist Before PR

- [ ] Code follows style guidelines
- [ ] All tests pass: `pytest tests/`
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No hardcoded secrets or credentials
- [ ] Git history is clean

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

**Happy Contributing! 🎉**
