# Deployment Guide

## 🚀 Local Deployment

### Windows
```bash
# PowerShell or Command Prompt
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_web.py
```

### Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_web.py
```

## 🐳 Docker Deployment

### Requirements
- Docker 20.10+
- Docker Compose 2.0+

### Build and Run

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f web

# Stop services
docker-compose down
```

### Access Points
- **Web App**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## ☁️ Cloud Deployment

### AWS EC2

```bash
# 1. Launch EC2 instance (Ubuntu 20.04+)
# 2. SSH into instance
ssh -i key.pem ubuntu@your-instance-ip

# 3. Install dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv git

# 4. Clone and setup
git clone <repo-url>
cd abnormal-behavior-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Run with nohup (background)
nohup python run_web.py > web.log 2>&1 &
nohup python run_api.py > api.log 2>&1 &

# 6. Configure Security Group
# - Allow port 8501 (web)
# - Allow port 8000 (API)
# - Allow port 22 (SSH)
```

### Heroku

```bash
# 1. Install Heroku CLI
# 2. Login
heroku login

# 3. Create app
heroku create your-app-name

# 4. Set environment variables
heroku config:set PYTHONUNBUFFERED=1

# 5. Deploy
git push heroku main

# 6. View logs
heroku logs --tail
```

### Google Cloud Run

```bash
# 1. Install gcloud CLI
# 2. Authenticate
gcloud auth login

# 3. Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/behavior-detection

# 4. Deploy
gcloud run deploy behavior-detection \
  --image gcr.io/PROJECT_ID/behavior-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 3600
```

## 🔐 Production Checklist

- [ ] Use environment variables for secrets
- [ ] Set `reload=False` in production
- [ ] Enable HTTPS/SSL
- [ ] Set proper CORS headers
- [ ] Rate limiting enabled
- [ ] Error logging configured
- [ ] Database credentials secured
- [ ] API authentication enabled

## 📊 Monitoring

### Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Performance Metrics
- Track API response times
- Monitor GPU/CPU usage
- Log video processing times
- Track error rates

## 🔄 CI/CD Pipeline

### GitHub Actions Example

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/
      
      - name: Deploy to production
        env:
          DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
        run: |
          # Deployment script here
          echo "Deploying to production"
```

## 🚨 Troubleshooting

### Port Already in Use
```bash
# Linux/Mac
lsof -i :8501
kill -9 <PID>

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Out of Memory
- Reduce batch size
- Enable GPU
- Process videos one at a time
- Increase server resources

### Slow Performance
- Check GPU utilization
- Monitor network bandwidth
- Profile code with cProfile
- Optimize model inference

## 📈 Scaling

### Horizontal Scaling (Multiple Servers)
- Use load balancer (Nginx, HAProxy)
- Deploy multiple API instances
- Use message queue (Redis, RabbitMQ) for task distribution

### Vertical Scaling (More Resources)
- Increase CPU cores
- Add more RAM
- Use better GPU

---

For more information, see [README.md](README.md)
