import streamlit as st
import pystac_client
import planetary_computer
import odc.stac
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os
from datetime import datetime, timedelta, time

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(layout="wide", page_title="Kielder Digital Twin (Interpolation)", page_icon="🌊")

# Thông số hồ Kielder
Z_MIN = 132.00
Z_MAX = 184.00
F_MAX = 10.90
SHAPE_FACTOR = 2.0
PIXEL_AREA_M2 = 100
PIXEL_TO_KM2 = 1e-6

# Cấu hình API
BBOX = [-2.6086054204386926, 55.158006025096086, -2.442220807822906, 55.224624142442934]
TIME_RANGE = "2017-01-01/2025-12-31"
MASK_DIR = "data/masks"
MAX_CLOUD_COVER = 10  # Tăng lên 10% để có nhiều điểm dữ liệu nội suy hơn

# --- 2. HÀM XỬ LÝ ẢNH & MASK ---

def calculate_water_level(mask_array):
    """Tính mực nước Z từ mask nhị phân"""
    water_pixels = np.count_nonzero(mask_array)
    area_km2 = min(water_pixels * PIXEL_AREA_M2 * PIXEL_TO_KM2, F_MAX)
    
    if area_km2 <= 0: return area_km2, Z_MIN
    
    ratio = area_km2 / F_MAX
    z = Z_MIN + (Z_MAX - Z_MIN) * np.power(ratio, 1/SHAPE_FACTOR)
    return area_km2, z

def load_mask_for_date(date_obj, ref_shape):
    """
    Tìm file mask PNG cho ngày cụ thể. 
    Nếu không có -> Trả về None (để sau này Auto-gen).
    """
    filename = f"img_{date_obj.strftime('%Y-%m-%d')}.png"
    path = os.path.join(MASK_DIR, filename)
    
    if os.path.exists(path):
        try:
            mask = Image.open(path).convert('L')
            if mask.size != ref_shape:
                mask = mask.resize(ref_shape, resample=Image.NEAREST)
            # Threshold về 0-1
            return np.where(np.array(mask) > 100, 1, 0).astype(np.uint8)
        except:
            return None
    return None

def auto_generate_mask(img_array):
    """Tự tạo mask từ ảnh vệ tinh (cho những ngày thiếu file label)"""
    # Heuristic: Nước tối (Red < 60) và Blue > Red
    return ((img_array[:,:,0] < 60) & (img_array[:,:,2] > img_array[:,:,0])).astype(np.uint8)

# --- 3. KẾT NỐI API & NỘI SUY (CORE LOGIC) ---

@st.cache_data(ttl=3600)
def fetch_metadata():
    """Lấy danh sách các ngày có ảnh sạch"""
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"], 
        bbox=BBOX, 
        datetime=TIME_RANGE, 
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}
    )
    items = list(search.item_collection())
    items.sort(key=lambda x: x.datetime)
    return items

def download_image_raw(item):
    """Tải ảnh thô, không cache ở đây để tránh lỗi hash"""
    ds = odc.stac.load([item], bands=["B04", "B03", "B02"], bbox=BBOX, resolution=10, chunks={})
    r = ds["B04"].values[0].astype(float)
    g = ds["B03"].values[0].astype(float)
    b = ds["B02"].values[0].astype(float)
    
    if item.datetime.strftime("%Y-%m-%d") >= "2022-01-25":
        r-=1000; g-=1000; b-=1000
    
    return np.dstack((np.clip(r/2000,0,1), np.clip(g/2000,0,1), np.clip(b/2000,0,1)))

def get_interpolated_data(target_date, items):
    """
    Hàm quan trọng nhất:
    - Nếu trúng ngày -> Lấy ảnh thật.
    - Nếu lệch ngày -> Lấy ảnh trước & sau rồi trộn (Blend) pixel + mask.
    """
    # 1. Chuyển target_date về dạng datetime so sánh được
    t_dt = datetime.combine(target_date, time(12,0)).astimezone()
    dates = [i.datetime for i in items]
    
    # 2. Tìm vị trí chèn
    idx = np.searchsorted(dates, t_dt)
    
    # Xử lý biên (đầu/cuối chuỗi)
    if idx == 0: idx = 1
    if idx >= len(dates): idx = len(dates) - 1
    
    item_prev = items[idx-1]
    item_next = items[idx]
    
    # 3. Tính trọng số thời gian (Alpha)
    t_prev = item_prev.datetime
    t_next = item_next.datetime
    total_sec = (t_next - t_prev).total_seconds()
    curr_sec = (t_dt - t_prev).total_seconds()
    
    alpha = np.clip(curr_sec / total_sec, 0, 1) if total_sec > 0 else 0
    
    # 4. Tải dữ liệu 2 đầu
    with st.spinner(f"Interpolating: {t_prev.date()} ⟷ {t_next.date()} (α={alpha:.2f})..."):
        arr_prev = download_image_raw(item_prev)
        arr_next = download_image_raw(item_next)
        
        # Resize nếu lệch pixel (do cắt BBOX đôi khi lệch 1px)
        if arr_prev.shape != arr_next.shape:
             h, w, c = arr_prev.shape
             # Dùng openCV hoặc PIL resize, ở đây dùng PIL cho đơn giản
             img_next_pil = Image.fromarray((arr_next*255).astype(np.uint8)).resize((w, h))
             arr_next = np.array(img_next_pil) / 255.0

        # --- A. TRỘN ẢNH VỆ TINH ---
        arr_interp = arr_prev * (1 - alpha) + arr_next * alpha
        img_final = Image.fromarray((np.power(arr_interp, 0.6) * 255).astype(np.uint8)) # Gamma correction 0.6
        
        # --- B. TRỘN MASK (Quan trọng) ---
        # Lấy mask gốc (từ file PNG hoặc Auto)
        w, h = img_final.size
        mask_p = load_mask_for_date(t_prev, (w, h))
        if mask_p is None: mask_p = auto_generate_mask(arr_prev*255)
            
        mask_n = load_mask_for_date(t_next, (w, h))
        if mask_n is None: mask_n = auto_generate_mask(arr_next*255)
            
        # Blend mask (ra ảnh xám)
        mask_blend = mask_p * (1 - alpha) + mask_n * alpha
        # Threshold: > 0.5 thì tính là nước (để về lại nhị phân)
        mask_final = np.where(mask_blend > 0.5, 1, 0).astype(np.uint8)
        
        return img_final, mask_final, f"Interpolated ({alpha:.1%})"

# --- 4. GIAO DIỆN CHÍNH ---

st.title(f"🛰️ Kielder Digital Twin: Auto-Interpolation")
col_info, col_cloud = st.columns([3, 1])
col_info.markdown("Tự động nội suy ảnh và mực nước cho **mọi ngày bất kỳ**.")
col_cloud.metric("Max Cloud", f"{MAX_CLOUD_COVER}%")

# Load Metadata
items = fetch_metadata()
if not items:
    st.error("Không có dữ liệu. Hãy kiểm tra kết nối API.")
    st.stop()

min_d, max_d = items[0].datetime.date(), items[-1].datetime.date()

# SLIDER CHỌN NGÀY (Cho phép chọn từng ngày một)
selected_date = st.slider("Timeline Control:", min_value=min_d, max_value=max_d, value=max_d, format="DD/MM/YYYY")

st.divider()

col_vis, col_stat = st.columns([1.6, 1], gap="large")

# === CỘT TRÁI: HIỂN THỊ ẢNH ===
with col_vis:
    st.subheader(f"👁️ View: {selected_date.strftime('%d/%m/%Y')}")
    
    # Gọi hàm nội suy
    img, mask, status = get_interpolated_data(selected_date, items)
    
    # Hiển thị
    blue_layer = np.zeros((img.height, img.width, 4), dtype=np.uint8)
    blue_layer[mask == 1] = [0, 150, 255, 120] # Màu xanh nước
    
    overlay = Image.alpha_composite(img.convert("RGBA"), Image.fromarray(blue_layer))
    st.image(overlay, use_container_width=True)
    st.caption(f"Status: **{status}**")

# === CỘT PHẢI: SỐ LIỆU & BIỂU ĐỒ ===
with col_stat:
    st.subheader("📊 Interpolated Analytics")
    
    # Tính toán từ Mask nội suy
    area, level = calculate_water_level(mask)
    
    c1, c2 = st.columns(2)
    c1.metric("Area (F)", f"{area:.2f} km²")
    c2.metric("Level (Z)", f"{level:.2f} m", delta="Calculated")
    
    st.write("---")
    st.markdown("#### 📅 3-Month Forecast (Interpolated)")
    
    # Dự báo tương lai (Cũng dùng nội suy)
    horizons = [30, 60, 90]
    points = [{"Date": selected_date, "Level": level, "Type": "Current"}]
    
    for days in horizons:
        f_date = selected_date + timedelta(days=days)
        if f_date <= max_d:
            # Tái sử dụng hàm nội suy cho tương lai
            # Lưu ý: Demo thì gọi lại hàm này, thực tế nên cache nếu gọi nhiều
            _, f_mask, _ = get_interpolated_data(f_date, items)
            _, f_z = calculate_water_level(f_mask)
            points.append({"Date": f_date, "Level": f_z, "Type": "Forecast"})
    
    # Vẽ biểu đồ
    df = pd.DataFrame(points)
    fig = go.Figure()
    
    # Đường nối
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Level"], mode='lines', 
        line=dict(color='gray', width=1, dash='dot'), showlegend=False
    ))
    
    # Điểm dữ liệu
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Level"], mode='markers+text',
        text=df["Level"].apply(lambda x: f"{x:.1f}m"), textposition="top center",
        marker=dict(size=12, color=['#00E5FF' if t=="Current" else '#FFD740' for t in df["Type"]])
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(range=[Z_MIN, Z_MAX+2], title="Elevation (m)")
    )
    st.plotly_chart(fig, use_container_width=True)
