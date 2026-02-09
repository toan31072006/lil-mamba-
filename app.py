import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import re
import glob
from datetime import datetime, timedelta

# --- CẤU HÌNH ---
st.set_page_config(layout="wide", page_title="Lake Digital Twin", page_icon="🌊")

IMG_DIR = "data/images"  # Nơi chứa ảnh gốc
MASK_DIR = "data/masks"  # Nơi chứa ảnh mask

# --- HÀM XỬ LÝ ---
@st.cache_data
def load_file_index(folder_path):
    """Tạo index ngày tháng -> đường dẫn file"""
    if not os.path.exists(folder_path):
        return {}, []
    files = glob.glob(os.path.join(folder_path, "*.png")) # Hoặc .jpg tùy đuôi ảnh của bạn
    index = {}
    for f in files:
        # Tìm ngày dạng YYYY-MM-DD trong tên file
        match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(f))
        if match:
            dt = datetime.strptime(match.group(1), "%Y-%m-%d")
            index[dt] = f
    return index, sorted(index.keys())

def interpolate(target_date, date_prev, date_next, path_prev, path_next):
    """Nội suy tuyến tính giữa 2 ảnh"""
    img_prev = np.array(Image.open(path_prev).convert("RGB"), dtype=np.float32)
    img_next = np.array(Image.open(path_next).convert("RGB"), dtype=np.float32)
    
    if img_prev.shape != img_next.shape:
        img_next = np.array(Image.fromarray(img_next.astype('uint8')).resize((img_prev.shape[1], img_prev.shape[0])))

    total_seconds = (date_next - date_prev).total_seconds()
    if total_seconds == 0: return img_prev.astype(np.uint8)
    
    alpha = (target_date - date_prev).total_seconds() / total_seconds
    return (img_prev * (1 - alpha) + img_next * alpha).astype(np.uint8)

def get_image(target_date, index, dates, is_mask=False):
    """Lấy ảnh thật hoặc nội suy"""
    if not dates: return None, "No Data"
    
    # Case 1: Có ảnh thật
    if target_date in index:
        return Image.open(index[target_date]), "Real Data"
    
    # Case 2: Ngoài vùng dữ liệu
    if target_date < dates[0] or target_date > dates[-1]:
        return None, "Out of Range"
        
    # Case 3: Nội suy
    idx = np.searchsorted(dates, target_date)
    d_prev, d_next = dates[idx-1], dates[idx]
    img_arr = interpolate(target_date, d_prev, d_next, index[d_prev], index[d_next])
    
    # Nếu là mask thì threshold về 0-255 cho rõ nét
    if is_mask:
        img_arr = np.where(img_arr > 127, 255, 0).astype(np.uint8)
        
    return Image.fromarray(img_arr), "Interpolated"

# --- GIAO DIỆN ---
st.title("🛰️ Digital Twin: Water Level Monitoring")

# Load dữ liệu
img_idx, all_dates = load_file_index(IMG_DIR)
mask_idx, _ = load_file_index(MASK_DIR)

if not all_dates:
    st.error(f"Không tìm thấy ảnh trong thư mục '{IMG_DIR}'. Hãy giải nén dữ liệu vào folder data.")
    st.stop()

# Sidebar
st.sidebar.header("Controls")
selected_date = st.sidebar.slider(
    "Select Date:", 
    min_value=all_dates[0], 
    max_value=all_dates[-1], 
    value=all_dates[0],
    format="DD/MM/YYYY"
)
show_mask = st.sidebar.checkbox("Show Water Mask", True)

# Main View
col1, col2 = st.columns(2)

# Cột Trái: Hiện tại
with col1:
    st.subheader(f"📍 Current: {selected_date.strftime('%d/%m/%Y')}")
    img, status = get_image(selected_date, img_idx, all_dates)
    mask, _ = get_image(selected_date, mask_idx, all_dates, is_mask=True)
    
    if img:
        st.caption(f"Status: {status}")
        display = img.convert("RGBA")
        if show_mask and mask:
            # Tô màu xanh lên vùng mask
            mask_l = mask.convert("L")
            blue_layer = Image.new("RGBA", display.size, (0, 100, 255, 100))
            display = Image.composite(blue_layer, display, mask_l)
        st.image(display, use_container_width=True)

# Cột Phải: Horizon (1, 2, 3 tháng)
with col2:
    st.subheader("🔮 Forecast Horizons")
    horizons = [30, 60, 90] # ngày
    
    for days in horizons:
        h_date = selected_date + timedelta(days=days)
        h_img, h_status = get_image(h_date, img_idx, all_dates)
        h_mask, _ = get_image(h_date, mask_idx, all_dates, is_mask=True)
        
        st.markdown(f"**+{days//30} Month ({h_date.strftime('%d/%m/%Y')})**")
        if h_img:
            h_disp = h_img.convert("RGBA")
            if show_mask and h_mask:
                mask_l = h_mask.convert("L")
                blue_layer = Image.new("RGBA", h_disp.size, (0, 100, 255, 100))
                h_disp = Image.composite(blue_layer, h_disp, mask_l)
            st.image(h_disp, use_container_width=True)
        else:
            st.warning("No data available for this horizon.")
        st.divider()
