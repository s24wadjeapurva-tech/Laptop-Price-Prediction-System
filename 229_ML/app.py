import os
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
logo_path = os.path.join(BASE_DIR, 'logo.png')

st.set_page_config(page_title="Laptop Price Predictor", page_icon=logo_path, layout="centered")

import pickle
import numpy as np
import re
import pandas as pd

model_path = os.path.join(BASE_DIR, 'model.pkl')
le_path = os.path.join(BASE_DIR, 'le.pkl')
cpu_le_path = os.path.join(BASE_DIR, 'cpu_le.pkl')
gpu_le_path = os.path.join(BASE_DIR, 'gpu_le.pkl')

# Load trained model and label encoders
if not os.path.exists(model_path) or not os.path.exists(le_path) or not os.path.exists(cpu_le_path) or not os.path.exists(gpu_le_path):
    st.error('Required model or encoder files not found. Please run train.py first.')
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(le_path, 'rb') as f:
    le = pickle.load(f)

with open(cpu_le_path, 'rb') as f:
    cpu_le = pickle.load(f)

with open(gpu_le_path, 'rb') as f:
    gpu_le = pickle.load(f)

st.logo(logo_path)

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image(logo_path, use_container_width=True)
with col_title:
    st.title('Laptop Price Predictor')

# Input Fields
col1, col2 = st.columns(2)

with col1:
    companies = list(le.classes_)
    company = st.selectbox('Company', companies)
    
    ram = st.selectbox('RAM (GB)', [4, 8, 16, 32, 64])
    
    ssd = st.selectbox('SSD Storage (GB)', [0, 128, 256, 512, 1024, 2048])

with col2:
    processors = sorted(cpu_le.classes_)
    processor = st.selectbox('Processor', processors)
    
    gpus = sorted(gpu_le.classes_)
    gpu = st.selectbox('GPU Brand', gpus)
    
    hdd = st.selectbox('HDD Storage (GB)', [0, 500, 1024, 2048])

if st.button('Predict Price'):
    try:
        company_code = int(le.transform([company])[0])
        processor_code = int(cpu_le.transform([processor])[0])
        gpu_code = int(gpu_le.transform([gpu])[0])
    except Exception:
        st.error('Selected company, processor, or GPU not recognized by encoder')
        st.stop()

    # Features: ['Company_code', 'Ram', 'CPU_code', 'Gpu_code', 'SSD_GB', 'HDD_GB']
    features = np.array([[company_code, ram, processor_code, gpu_code, ssd, hdd]])
    prediction = model.predict(features)
    st.success(f'Estimated Laptop Price: ₹{prediction[0]:.2f} (INR)')
    
    brand_lower = company.lower()
    image_path = os.path.join(BASE_DIR, 'images', f'{brand_lower}.png')
    
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True, caption=f"Your Premium {company} Machine Awaits!")
    else:
        st.image(os.path.join(BASE_DIR, 'laptop_image.png'), use_container_width=True, caption="Your Premium Machine Awaits!")

st.markdown("---")
st.header("📊 Data Analysis")

@st.cache_data
def load_and_clean_data():
    csv_path = os.path.join(BASE_DIR, "laptop.csv")
    df = pd.read_csv(csv_path, encoding='latin1')
    rename_map = {'Manufacturer': 'Company', 'RAM': 'Ram', 'Price (Euros)': 'Price_euros'}
    df = df.rename(columns=rename_map)
    
    df['Ram'] = df['Ram'].astype(str).str.extract(r"(\d+)")[0]
    df['Ram'] = pd.to_numeric(df['Ram'], errors='coerce')
    df['CPU'] = df['CPU'].astype(str).str.strip()
    
    df['Price_euros'] = df['Price_euros'].astype(str).str.replace('"', '').str.replace("'", '').str.replace(',', '.', regex=False).str.replace(r'[^0-9\.]', '', regex=True)
    df['Price_euros'] = pd.to_numeric(df['Price_euros'], errors='coerce')
    
    df['Gpu_brand'] = df['GPU'].astype(str).apply(lambda x: x.split()[0] if len(x.split()) > 0 else 'Unknown')
    
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
    
    df = df.dropna(subset=['Company', 'Ram', 'CPU', 'Price_euros', 'Gpu_brand', 'SSD_GB', 'HDD_GB'])
    df['Price_inr'] = df['Price_euros'] * 88.0
    return df

df = load_and_clean_data()

analysis_type = st.radio("Select Analysis Type:", [
    "Laptop Brand Analysis", 
    "Processor Analysis", 
    "RAM Analysis",
    "GPU Brand Analysis",
    "Storage Analysis"
])

if analysis_type == "Laptop Brand Analysis":
    st.subheader("Average Price by Laptop Brand")
    brand_price = df.groupby('Company')['Price_inr'].mean().sort_values(ascending=False)
    st.bar_chart(brand_price)

elif analysis_type == "Processor Analysis":
    st.subheader("Average Price by Processor")
    df['CPU_Short'] = df['CPU'].apply(lambda x: " ".join(x.split()[:2]))
    cpu_price = df.groupby('CPU_Short')['Price_inr'].mean().sort_values(ascending=False).head(15)
    st.bar_chart(cpu_price)

elif analysis_type == "RAM Analysis":
    st.subheader("Average Price by RAM Capacity")
    ram_price = df.groupby('Ram')['Price_inr'].mean().sort_index()
    st.line_chart(ram_price)

elif analysis_type == "GPU Brand Analysis":
    st.subheader("Average Price by GPU Brand")
    gpu_price = df.groupby('Gpu_brand')['Price_inr'].mean().sort_values(ascending=False)
    st.bar_chart(gpu_price)

elif analysis_type == "Storage Analysis":
    st.subheader("Average Price by SSD Capacity")
    ssd_price = df[df['SSD_GB'] > 0].groupby('SSD_GB')['Price_inr'].mean().sort_index()
    st.line_chart(ssd_price)
    
    st.subheader("Average Price by HDD Capacity")
    hdd_price = df[df['HDD_GB'] > 0].groupby('HDD_GB')['Price_inr'].mean().sort_index()
    st.line_chart(hdd_price)
