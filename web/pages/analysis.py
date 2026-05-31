# web/pages/analysis.py

# web/pages/analysis.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
from streamlit_autorefresh import st_autorefresh
from config.settings import settings

st.set_page_config(page_title="Real-time Analysis", page_icon="📈", layout="wide")

# Tự động refresh trang mỗi 3 giây (3000 ms)
count = st_autorefresh(interval=3000, limit=None, key="realtime_refresh")

st.markdown("""
<style>
    .metric-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #2E7D32;
    }
    .metric-alert { border-left: 5px solid #D32F2F; background-color: #ffeaea;}
    .metric-title { font-size: 1.1rem; color: #555; font-weight: bold;}
    .metric-value { font-size: 2rem; color: #111; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("🔴 Real-time Behavior Analytics")
st.markdown("Live monitoring of anomalous behaviors detected by the CCTV network.")

# --- ĐỌC DỮ LIỆU LOG TRONG NGÀY ---
today_str = datetime.now().strftime('%Y%m%d')
log_file = settings.data_dir / "logs" / f"realtime_log_{today_str}.csv"

if not log_file.exists():
    st.info("🕒 Hệ thống đang chờ dữ liệu... Hãy bật Camera Demo lên nhé!", icon="ℹ️")
    st.stop()

# Đọc file CSV
df = pd.read_csv(log_file)
if df.empty:
    st.info("✅ Mọi thứ đều bình thường. Chưa có hành vi bất thường nào trong ngày hôm nay.")
    st.stop()

# Ép kiểu datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp', ascending=False) # Mới nhất lên đầu

# --- THỐNG KÊ NHANH (METRICS) ---
st.subheader("Báo cáo trong ngày", divider="green")
col1, col2, col3, col4 = st.columns(4)

total_anomalies = len(df)
count_fighting = len(df[df['Behavior'] == 'fighting'])
count_falling = len(df[df['Behavior'] == 'falling'])
count_loitering = len(df[df['Behavior'] == 'loitering'])

# Kiểm tra xem có cảnh báo mới trong vòng 1 phút qua không
one_min_ago = datetime.now() - timedelta(minutes=1)
recent_alerts = df[df['Timestamp'] > one_min_ago]

with col1:
    bg_class = "metric-alert" if len(recent_alerts) > 0 else "metric-container"
    st.markdown(f"""
        <div class="{bg_class}">
            <div class="metric-title">Tổng Sự Cố Nay</div>
            <div class="metric-value">{total_anomalies}</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-container"><div class="metric-title">Bạo lực (Fighting)</div><div class="metric-value" style="color:#D32F2F;">{count_fighting}</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-container"><div class="metric-title">Té ngã (Falling)</div><div class="metric-value" style="color:#F57C00;">{count_falling}</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-container"><div class="metric-title">Lảng vảng (Loitering)</div><div class="metric-value" style="color:#1976D2;">{count_loitering}</div></div>""", unsafe_allow_html=True)

st.write("") # Margin

# --- VẼ BIỂU ĐỒ (CHARTS) ---
chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    st.subheader("Dòng thời gian sự cố (Timeline)")
    # Nhóm theo từng phút để vẽ biểu đồ
    df_grouped = df.copy()
    df_grouped['Minute'] = df_grouped['Timestamp'].dt.floor('Min')
    time_chart_data = df_grouped.groupby(['Minute', 'Behavior']).size().reset_index(name='Count')
    
    color_map = {'fighting': '#D32F2F', 'falling': '#F57C00', 'loitering': '#1976D2'}
    
    fig_time = px.bar(time_chart_data, x='Minute', y='Count', color='Behavior',
                      color_discrete_map=color_map,
                      title="Tần suất hành vi bất thường theo thời gian",
                      labels={'Minute': 'Thời gian', 'Count': 'Số lượng phát hiện'})
    fig_time.update_layout(xaxis_title="", yaxis_title="Số lần", height=400, plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_time, use_container_width=True)

with chart_col2:
    st.subheader("Tỷ trọng (Distribution)")
    pie_data = df['Behavior'].value_counts().reset_index()
    pie_data.columns = ['Behavior', 'Count']
    
    fig_pie = px.pie(pie_data, values='Count', names='Behavior', 
                     color='Behavior', color_discrete_map=color_map,
                     hole=0.4, title="Phân bổ các loại sự cố")
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- BẢNG DỮ LIỆU LOG ---
st.subheader("Nhật ký chi tiết", divider="green")
# Highlight các dòng dữ liệu
def highlight_behavior(val):
    if val == 'fighting': return 'color: #D32F2F; font-weight: bold'
    elif val == 'falling': return 'color: #F57C00; font-weight: bold'
    elif val == 'loitering': return 'color: #1976D2; font-weight: bold'
    return ''

st.dataframe(df.head(50).style.applymap(highlight_behavior, subset=['Behavior']), 
             use_container_width=True, hide_index=True)