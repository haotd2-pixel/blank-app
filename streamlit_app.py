import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# --- Data Preprocessing ---
@st.cache_data
def load_data(uploaded_file):
    # Efficient reading; only needed columns
    dtypes = {
        'id': np.int32,
        'time': np.float32,
        'xloc_kf': np.float32,
        'yloc_kf': np.float32,
        'lane_kf': 'category',
        'speed_kf': np.float32,
        'acceleration': np.float32,
        'length_sm': np.float32,
        'width_sm': np.float32,
        'type_mostav': 'category',
        'run_index': np.int8
    }
    df = pd.read_csv(uploaded_file, dtype=dtypes)
    return df

# --- Streamlit UI ---
st.title("TGSIM I-90/I-94 Trajectory Data Visualization & Analysis")

uploaded_file = st.file_uploader("Upload TGSIM Trajectory CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    
    # Lane selection menu
    lanes = sorted(df['lane_kf'].unique())
    selected_lane = st.selectbox("Select Lane", lanes)
    filtered_df = df[df['lane_kf'] == selected_lane]

    # Vehicle selection (optional, scalable for large data)
    vehicle_ids = filtered_df['id'].unique()
    selected_vehicles = st.multiselect("Vehicles to display", vehicle_ids, default=vehicle_ids[:10])

    plot_df = filtered_df[filtered_df['id'].isin(selected_vehicles)].copy()

    # Region / zoom selection
    min_time, max_time = plot_df['time'].min(), plot_df['time'].max()
    time_range = st.slider("Time Range", float(min_time), float(max_time), (float(min_time), float(max_time)))
    plot_df = plot_df[(plot_df['time'] >= time_range) & (plot_df['time'] <= time_range[1])].copy()

    # --- Trajectory Visualization (Plotly for interactive zoom) ---
    fig = px.line(
        plot_df,
        x="xloc_kf", y="yloc_kf", color="id",
        labels={"xloc_kf": "X Position", "yloc_kf": "Y Position"},
        title="Vehicle Trajectories (Zoom and Pan enabled)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Traffic Flow Analysis ---
    st.header("Traffic Flow Metrics")

    # Time and Space Headways
    st.subheader("Headways")
    plot_headways = plot_df.sort_values(['lane_kf', 'time', 'xloc_kf'])
    headways_time = plot_headways.groupby('lane_kf').apply(
        lambda g: g.sort_values('time').groupby('id')['time'].diff().dropna()
    ).explode().dropna()
    headways_space = plot_headways.groupby('lane_kf').apply(
        lambda g: g.sort_values('xloc_kf').groupby('id')['xloc_kf'].diff().dropna()
    ).explode().dropna()
    fig_ht = plt.figure()
    plt.hist(headways_time, bins=30, color='skyblue')
    plt.xlabel('Temporal Headway (s)')
    plt.ylabel('Count')
    plt.title('Distribution of Temporal Headways')
    st.pyplot(fig_ht)

    fig_hs = plt.figure()
    plt.hist(headways_space, bins=30, color='orange')
    plt.xlabel('Spatial Headway (m)')
    plt.ylabel('Count')
    plt.title('Distribution of Spatial Headways')
    st.pyplot(fig_hs)

    # Individual Speeds
    st.subheader("Speeds")
    fig_spd = px.histogram(plot_df, x="speed_kf", nbins=40, title="Distribution of Individual Vehicle Speeds")
    st.plotly_chart(fig_spd)

    # Space-Mean Speed (by segments, e.g. per 10s window)
    st.subheader("Space-Mean Speeds")
    plot_df['segment'] = pd.cut(plot_df['time'], bins=np.arange(min_time, max_time+10, 10))
    sms = plot_df.groupby('segment')['speed_kf'].mean()
    fig_sms = px.line(x=sms.index.astype(str), y=sms.values, labels={'x':'Time Segment', 'y':'Space-Mean Speed'}, title="Space-Mean Speeds Over Time")
    st.plotly_chart(fig_sms)

    # Flow (vehicles per time)
    st.subheader("Flow")
    flow = plot_df.groupby('time')['id'].nunique()
    fig_flow = px.line(x=flow.index, y=flow.values, labels={'x':'Time (s)', 'y':'Flow (veh/s)'}, title="Traffic Flow Over Time")
    st.plotly_chart(fig_flow)

    # Density (vehicles per distance)
    st.subheader("Density")
    # Example: per 100m segment along xloc_kf
    plot_df['distance_bin'] = pd.cut(plot_df['xloc_kf'], bins=np.arange(plot_df['xloc_kf'].min(), plot_df['xloc_kf'].max()+100, 100))
    density = plot_df.groupby('distance_bin')['id'].nunique()
    fig_density = px.bar(x=density.index.astype(str), y=density.values, labels={'x':'Position Bin', 'y':'Density (veh/100m)'}, title="Density Distribution")
    st.plotly_chart(fig_density)

    # Speed-Density-Flow Relationships
    st.subheader("Speed-Density-Flow Relationships")
    sum_df = pd.DataFrame({
        'space_mean_speed': sms.values,
        'flow': flow.reindex(sms.index, fill_value=0).values,
        'density': density.reindex(sms.index, fill_value=0).values
    })
    fig_sdf = px.scatter(sum_df, x="density", y="space_mean_speed", size="flow", title="Speed-Density-Flow Scatter")
    st.plotly_chart(fig_sdf)

    st.write("Explore and interpret the metrics above to understand traffic behavior for selected lanes and vehicles.")

else:
    st.info("Please upload the TGSIM CSV file to begin.")