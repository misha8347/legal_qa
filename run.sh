#!/bin/bash

# Optional: activate your virtual environment
source venv/bin/activate

# Start FastAPI server using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload