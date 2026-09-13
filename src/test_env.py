import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MASSIVE_API_KEY")

print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("statsmodels version:", sm.__version__)
print("API key loaded:", "Yes" if api_key else "No")