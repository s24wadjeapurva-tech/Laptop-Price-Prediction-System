import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import re

BASE_DIR = os.path.dirname(__file__)
# Load dataset (file in workspace is `laptop.csv`)
csv_path = os.path.join(BASE_DIR, "laptop.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found in workspace")

df = pd.read_csv(csv_path, encoding='latin1')

# Normalize column names we care about and print them for debugging
print('Available columns:', list(df.columns))
rename_map = {
    'Manufacturer': 'Company',
    'RAM': 'Ram',
    'Price (Euros)': 'Price_euros'
}
df = df.rename(columns=rename_map)

# Keep only the columns we need if present
required_cols = ['Company', 'Ram', 'CPU', 'Price_euros', 'GPU', ' Storage']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

df = df[required_cols].copy()

# Clean `Ram` column: extract numeric GB value
df['Ram'] = df['Ram'].astype(str).str.extract(r"(\d+)")[0]
df['Ram'] = pd.to_numeric(df['Ram'], errors='coerce')

# Clean `CPU` column and preserve exact processor text
df['CPU'] = df['CPU'].astype(str).str.strip()

# Clean `Price_euros`
df['Price_euros'] = df['Price_euros'].astype(str)
df['Price_euros'] = df['Price_euros'].str.replace('"', '')
df['Price_euros'] = df['Price_euros'].str.replace("'", '')
df['Price_euros'] = df['Price_euros'].str.replace(',', '.', regex=False)
df['Price_euros'] = df['Price_euros'].str.replace(r'[^0-9\.]', '', regex=True)
df['Price_euros'] = pd.to_numeric(df['Price_euros'], errors='coerce')

# Extract GPU Brand
df['Gpu_brand'] = df['GPU'].astype(str).apply(lambda x: x.split()[0] if len(x.split()) > 0 else 'Unknown')
# Some GPUs might be ARM, let's keep only major ones if needed, or just let LabelEncoder handle it.

# Extract Storage
def parse_storage(row):
    ssd, hdd = 0, 0
    parts = str(row).split('+')
    for p in parts:
        p = p.strip()
        match = re.search(r'(\d+)(GB|TB)\s*(SSD|HDD|Flash Storage|Hybrid)', p)
        if match:
            val, unit, type_ = match.groups()
            val = int(val)
            if unit == 'TB':
                val *= 1024
            if type_ in ['SSD', 'Flash Storage']:
                ssd += val
            elif type_ in ['HDD', 'Hybrid']:
                hdd += val
    return pd.Series([ssd, hdd])

df[['SSD_GB', 'HDD_GB']] = df[' Storage'].apply(parse_storage)

# Drop rows with missing values in our selected columns
df = df.dropna(subset=['Company', 'Ram', 'CPU', 'Price_euros', 'Gpu_brand', 'SSD_GB', 'HDD_GB'])

# Convert string features to numeric labels and save the encoders
le = LabelEncoder()
df['Company_code'] = le.fit_transform(df['Company'].astype(str))

cpu_le = LabelEncoder()
df['CPU_code'] = cpu_le.fit_transform(df['CPU'].astype(str))

gpu_le = LabelEncoder()
df['Gpu_code'] = gpu_le.fit_transform(df['Gpu_brand'].astype(str))

# Convert prices from EUR to INR
EXCHANGE_RATE_EUR_TO_INR = 88.0  # adjust if you want a different rate
df['Price_inr'] = df['Price_euros'] * EXCHANGE_RATE_EUR_TO_INR

# Features and target (target in INR)
X = df[['Company_code', 'Ram', 'CPU_code', 'Gpu_code', 'SSD_GB', 'HDD_GB']]
y = df['Price_inr']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print(f"Model R^2 on test set (prices in INR, rate={EXCHANGE_RATE_EUR_TO_INR}):", score)

# Save model and label encoder
with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "le.pkl"), "wb") as f:
    pickle.dump(le, f)

with open(os.path.join(BASE_DIR, "cpu_le.pkl"), "wb") as f:
    pickle.dump(cpu_le, f)

with open(os.path.join(BASE_DIR, "gpu_le.pkl"), "wb") as f:
    pickle.dump(gpu_le, f)

print("Saved model.pkl, le.pkl, cpu_le.pkl, and gpu_le.pkl")