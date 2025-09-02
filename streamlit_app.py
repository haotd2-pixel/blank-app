import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

@st.cache_data
def load_data(uploaded_file):
    # Adjust columns/types if needed based on your real CSV header
    dtypes = {
        'id': np.int32,
        'time': np.float32,
        'xloc_kf': np.float32,
        'yloc_kf': np.float32,
        'lane_kf': 'category',
        'speed_kf': np.float32,
        'acceleration_kf': np.float32,
        'length_smoothed': np.float32,
        'width_smoothed': np.float32,
        'type_most_common': 'category',
        'av': 'category',
        'run_index': np.int8,
    }
    # Only use the cols present in your file!
    usecols = [
        'id', 'time', 'xloc_kf', 'yloc_kf', 'lane_kf', 'speed_kf',
        'acceleration_kf', 'length_smoothed', 'width_smoothed',
        'type_most_common', 'av', 'run_index'
    ]
    df = pd.read_csv(uploaded_file, dtype=dtypes, usecols=usecols)
    return df

st.title("TGSIM I-90/I-94 Trajectory Data Visualization & Analysis")
st.warning("Note: File upload limit is 200MB per file. Upload one chunk at a time.")

uploaded_file = st.file_uploader("Upload TGSIM Trajectory CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)

    # Lane selection
    lanes = sorted(df['lane_kf'].unique())
    selected_lane = st.selectbox("Select Lane", lanes)
    filtered_df = df[df['lane_kf'] == selected_lane]

    # Vehicle selection
    vehicle_ids = filtered_df['id'].unique()
    selected_vehicles = st.multiselect("Select Vehicles", vehicle_ids, default=list(vehicle_ids[:10]))
    plot_df = filtered_df[filtered_df['id'].isin(selected_vehicles)].copy()

    # Time zoom
    min_time = float(plot_df['time'].min())
    max_time = float(plot_df['time'].max())
    time_range = st.slider("Time Range", min_time, max_time, (min_time, max_time))
    plot_df = plot_df[(plot_df['time'] >= time_range) & (plot_df['time'] <= time_range[1])].copy()

    # Trajectory Visualization
    fig = px.line(
        plot_df,
        x="xloc_kf", y="yloc_kf", color="id",
        labels={"xloc_kf": "X Location", "yloc_kf": "Y Location"},
        title="Vehicle Trajectories"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.header("Traffic Flow Metrics")

    # --- Headways ---
    st.subheader("Headways (Temporal & Spatial)")
    sorted_df = plot_df.sort_values(['lane_kf', 'time', 'xloc_kf'])
    headways_time = sorted_df.groupby('id')['time'].diff().dropna()
    headways_space = sorted_df.groupby('id')['xloc_kf'].diff().dropna()
    fig1, ax1 = plt.subplots()
    ax1.hist(headways_time, bins=30, color='skyblue')
    ax1.set_title("Temporal Headways")
    ax1.set_xlabel("Headway (s)")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots()
    ax2.hist(headways_space, bins=30, color='orange')
    ax2.set_title("Spatial Headways")
    ax2.set_xlabel("Headway (m)")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    # --- Speed Distribution ---
    st.subheader("Speed Distribution")
    fig_spd = px.histogram(plot_df, x="speed_kf", nbins=40, title="Vehicle Speeds")
    st.plotly_chart(fig_spd)

    # --- Space-Mean Speed ---
    st.subheader("Space-Mean Speeds (per 10s segments)")
    plot_df['segment'] = pd.cut(plot_df['time'], bins=np.arange(min_time, max_time+10, 10))
    sms = plot_df.groupby('segment')['speed_kf'].mean()
    fig_sms = px.line(
        x=sms.index.astype(str), y=sms.values,
        labels={'x': 'Time Segment', 'y': 'Space-Mean Speed'},
        title="Space-Mean Speeds by Segment"
    )
    st.plotly_chart(fig_sms)

    # --- Flow (veh/s over time) ---
    st.subheader("Traffic Flow")
    flow = plot_df.groupby('time')['id'].nunique()
    fig_flow = px.line(
        x=flow.index, y=flow.values,
        labels={'x': 'Time (s)', 'y': 'Flow (veh/s)'},
        title="Traffic Flow Over Time"
    )
    st.plotly_chart(fig_flow)

    # --- Density (veh/100m bins) ---
    st.subheader("Density Distribution")
    plot_df['distance_bin'] = pd.cut(plot_df['xloc_kf'], bins=np.arange(plot_df['xloc_kf'].min(), plot_df['xloc_kf'].max()+100, 100))
    density = plot_df.groupby('distance_bin')['id'].nunique()
    fig_density = px.bar(
        x=density.index.astype(str), y=density.values,
        labels={'x': 'Position Bin (m)', 'y': 'Density (veh/100m)'},
        title="Density by Position Bin"
    )
    st.plotly_chart(fig_density)

    # --- Speed-Density-Flow Relationships ---
    st.subheader("Speed-Density-Flow (Fundamental Diagram)")
    # aggregate bins by segment/time if aligned, or align with sms index for visuals
    sum_df = pd.DataFrame({
        'space_mean_speed': sms.values,
        'flow': flow.reindex(sms.index, fill_value=0).values,
        'density': density.reindex(sms.index, fill_value=0).values
    })
    fig_sdf = px.scatter(
        sum_df, x="density", y="space_mean_speed", size="flow",
        title="Speed-Density-Flow Relationship"
    )
    st.plotly_chart(fig_sdf)

    st.write("Interact with metrics by lane, vehicle, and zoom region. Each plot can be filtered for more detail. For a full description, see the README.")
else:
    st.info("Please upload the TGSIM CSV file to begin.")
