import sys
from app import Predict

try:
    p = Predict()
    p.setup()
    print("Successfully imported and setup predictor.")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
