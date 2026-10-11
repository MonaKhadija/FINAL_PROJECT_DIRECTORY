**FINAL_PROJECT_DIRECTORY/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD workflow configuration
├── notebooks/
│   └── sentiment_training.ipynb   # Google Colab notebook for data training & serialization
├── app.py                         # Flask REST API implementation
├── Dockerfile                     # Production Docker build definition
├── test_app.py                    # Unit tests for API endpoints using Pytest
├── deployment.yaml                # Kubernetes Deployment and Service manifests
├── requirements.txt               # Python package dependencies
├── sentiment_model.pkl            # Trained Logistic Regression model artifact
└── tfidf_vectorizer.pkl           # Serialized TF-IDF Vectorizer**