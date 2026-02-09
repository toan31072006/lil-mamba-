import streamlit as st
import pystac_client
import planetary_computer
import odc.stac
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os
from datetime import datetime, timedelta

# --- 1. CẤU HÌNH THỦY VĂN (Kielder Water) ---
st.set_page_config(layout="wide", page_title="Kielder Water Twin (PNG Mask)", page_icon="🌊")

# Thông số hồ Kielder (Từ PDF báo cáo)
Z_MIN = 132.00      # Cao trình đáy (m)
Z_MAX = 184.00      # Cao trình dâng bình thường (m)
F_MAX = 10.90       # Diện tích mặt thoáng max (km2)
SHAPE_FACTOR = 2.0  # Hệ số n

# Diện tích 1 pixel (Sentinel-2 độ phân giải 10m)
PIXEL_AREA_M2 = 100 # 10m x 10m = 100m2
PIXEL_TO_KM2 = 1e-6 # Đổi m2 sang km2

# Cấu hình API & Thư mục
BBOX = [-2.6086054204386926, 55.158006025096086, -2.442220807822906, 55.224624142442934]
TIME_RANGE = "2017-01-01/2025-12-31"
MASK_DIR = "data/masks" # <-- Đọc từ folder masks chứa ảnh PNG

# --- 2. HÀM TÍNH TOÁN VẬT LÝ ---

def calculate_water_level_from_mask(mask_array):
    """
    Input: Ma trận ảnh mask (0 = nền, 255/1 = nước)
    Output: Diện tích (km2), Mực nước Z (m)
    """
    # Đếm số pixel nước (Giá trị > 0)
    water_pixels = np.count_nonzero(mask_array)
    
    # 1. Tính diện tích F (km2)
    area_km2 = water_pixels * PIXEL_AREA_M2 * PIXEL_TO_KM2
    
    # Clip (Không để vượt quá diện tích max của hồ)
    area_km2 = min(area_km2, F_MAX)
    
    # 2. Tính mực nước Z (m) theo công thức PDF
    if area_km2 <= 0:
        z = Z_MIN
    else:
        # Công thức: Z = Zmin + (Zmax - Zmin) * (F / Fmax)^(1/n)
        ratio = area_km2 / F_MAX
        z = Z_MIN + (Z_MAX - Z_MIN) * np.power(ratio, 1/SHAPE_FACTOR)
        
    return area_km2, z

def load_png_mask(png_path, target_size=None):
    """
    Đọc file PNG mask, chuyển về nhị phân (0-1).
    Nếu kích thước khác ảnh vệ tinh thì resize lại cho khớp.
    """
    try:
        mask = Image.open(png_path).convert('L') # Chuyển về ảnh xám
        
        if target_size and mask.size != target_size:
            mask = mask.resize(target_size, resample=Image.NEAREST)
            
        mask_arr = np.array(mask)
        
        # Ngưỡng hóa (Threshold): Đảm bảo chỉ có 0 và 1
        # Pixel > 127 coi là nước (1), còn lại là nền (0)
        binary_mask = np.where(mask_arr > 100, 1, 0).astype(np.uint8)
        
        return binary_mask
    except Exception as e:
        return None

def auto_generate_mask(pil_img):
    """Fallback: Tự tạo mask nếu không có file PNG"""
    arr = np.array(pil_img)
    # Thuật toán đơn giản: Nước thường tối và xanh
    mask = (arr[:,:,0] < 60) & (arr[:,:,2] > arr[:,:,0])
    return mask.astype(np.uint8)

# --- 3. HÀM API VỆ TINH ---

@st.cache_data(ttl=3600)
def fetch_metadata():
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=TIME_RANGE, query={"eo:cloud_cover": {"lt": 20}})
    items = list(search.item_collection())
    items.sort(key=lambda x: x.datetime)
    return items

@st.cache_data(show_spinner=False)
def download_satellite_image(item):
    ds = odc.stac.load([item], bands=["B04", "B03", "B02"], bbox=BBOX, resolution=10, chunks={})
    r = ds["B04"].values[0].astype(float)
    g = ds["B03"].values[0].astype(float)
    b = ds["B02"].values[0].astype(float)
    
    if item.datetime.strftime("%Y-%m-%d") >= "2022-01-25":
        r-=1000; g-=1000; b-=1000
        
    rgb = np.dstack((np.clip(r/2000,0,1), np.clip(g/2000,0,1), np.clip(b/2000,0,1)))
    return Image.fromarray((np.power(rgb, 0.6) * 255).astype(np.uint8))

# --- 4. GIAO DIỆN CHÍNH ---

st.title("🛰️ Kielder Digital Twin (PNG Mask Integration)")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("Tính toán mực nước từ **Mask PNG** và dữ liệu vệ tinh.")

# 1. Timeline
items = fetch_metadata()
dates = [i.datetime.date() for i in items]
selected_date = st.slider("Select Date:", min_value=dates[0], max_value=dates[-1], value=dates[-1], format="DD/MM/YYYY")

# Lấy item vệ tinh tương ứng
idx = np.searchsorted([d for d in dates], selected_date)
idx = min(idx, len(items)-1)
current_item = items[idx]
actual_date = current_item.datetime.date()

st.divider()

col_vis, col_stat = st.columns([1.5, 1], gap="large")

# --- CỘT TRÁI: HÌNH ẢNH ---
with col_vis:
    st.subheader(f"👁️ Satellite: {actual_date}")
    
    with st.spinner("Downloading from Microsoft API..."):
        sat_img = download_satellite_image(current_item)
    
    # --- LOGIC XỬ LÝ MASK PNG ---
    # Tìm file PNG trùng ngày trong folder
    png_filename = f"{actual_date.strftime('%Y-%m-%d')}.png"
    png_path = os.path.join(MASK_DIR, png_filename)
    
    mask_array = None
    mask_source = "N/A"
    
    if os.path.exists(png_path):
        # Nếu tìm thấy file PNG -> Load lên
        mask_array = load_png_mask(png_path, target_size=sat_img.size)
        mask_source = f"📂 Local PNG ({png_filename})"
    else:
        # Nếu không có -> Tự động tạo (Auto-threshold)
        mask_array = auto_generate_mask(sat_img)
        mask_source = "🤖 Auto-Generated (No PNG found)"
        
    # Tạo lớp phủ màu xanh để hiển thị
    blue_layer = np.zeros((sat_img.height, sat_img.width, 4), dtype=np.uint8)
    # Chỗ nào mask=1 thì tô màu xanh (0, 150, 255) với độ trong suốt 100/255
    blue_layer[mask_array == 1] = [0, 150, 255, 100]
    
    overlay_img = Image.alpha_composite(sat_img.convert("RGBA"), Image.fromarray(blue_layer))
    
    st.image(overlay_img, use_container_width=True)
    st.caption(f"Mask Source: **{mask_source}**")

# --- CỘT PHẢI: SỐ LIỆU & DỰ BÁO ---
with col_stat:
    st.subheader("📊 Water Level Analysis")
    
    # Tính toán
    area_km2, water_level_z = calculate_water_level_from_mask(mask_array)
    
    # Hiển thị Metric
    m1, m2 = st.columns(2)
    m1.metric("Surface Area", f"{area_km2:.2f} km²")
    m2.metric("Water Level (Z)", f"{water_level_z:.2f} m", help="Calculated using Reservoir Geometry Formula")
    
    st.divider()
    
    # --- DỰ BÁO HORIZON (2-4 THÁNG) ---
    st.markdown("#### 📅 Future Forecast (Horizon)")
    
    horizons = [60, 90, 120] # Ngày
    forecast_points = []
    
    # Điểm hiện tại
    forecast_points.append({"Time": "Now", "Date": actual_date, "Level": water_level_z})
    
    for days in horizons:
        f_date = actual_date + timedelta(days=days)
        
        # Tìm dữ liệu tương lai (Giả lập bằng cách tìm ảnh vệ tinh gần ngày đó nhất)
        f_idx = np.searchsorted([d for d in dates], f_date)
        
        if f_idx < len(items):
            f_item = items[f_idx]
            # Nếu tìm thấy ảnh tương lai (sai số < 20 ngày)
            if abs((f_item.datetime.date() - f_date).days) < 20:
                # Tải ảnh tương lai
                f_img_pil = download_satellite_image(f_item)
                # Vì tương lai chưa có file PNG label, ta dùng Auto-mask
                f_mask = auto_generate_mask(f_img_pil)
                _, f_z = calculate_water_level_from_mask(f_mask)
                
                forecast_points.append({
                    "Time": f"+{days//30} Months",
                    "Date": f_date,
                    "Level": f_z
                })
    
    # Vẽ biểu đồ
    if len(forecast_points) > 1:
        df_chart = pd.DataFrame(forecast_points)
        df_chart['DisplayDate'] = df_chart['Date'].apply(lambda x: x.strftime('%d/%m'))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_chart['Time'], y=df_chart['Level'],
            mode='lines+markers+text',
            text=df_chart['Level'].apply(lambda x: f"{x:.1f}m"),
            textposition="top center",
            line=dict(color='#00E5FF', width=3),
            marker=dict(size=8, color='white')
        ))
        
        fig.update_layout(
            title="Predicted Water Level Trend",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis_title="Elevation (m)",
            yaxis=dict(range=[Z_MIN, Z_MAX + 2], gridcolor='#333'),
            margin=dict(l=0,r=0,t=30,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough future data for forecast (Out of satellite range).")
