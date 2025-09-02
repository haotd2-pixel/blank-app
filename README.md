# TGSIM I-90/I-94 Interactive Traffic Visualization

This Streamlit app visualizes and analyzes Third-Generation Simulation (TGSIM) vehicle trajectory data with intuitive interface, lane selection, zoom, and traffic flow analytics.

## How to Run

1. Install requirements:
  
  pip install streamlit pandas numpy matplotlib plotly


2. Launch the app:
  
  streamlit run streamlit_app.py


3. Upload your TGSIM CSV file when prompted.

## Features

- Lane selection menu and vehicle filtering
- Time range (zoom/region) adjustment
- Interactive trajectory plot (Plotly)
- Analytical plots: headways (time, space), speeds, space-mean speeds, flow, density
- Speed-density-flow scatter plot

## Dataset Preprocessing

- Data types optimized for memory
- Filtering and grouping for scalable performance
- Uses efficient vectorized pandas operations

## Metric Interpretation

- **Headways:** Spacing and following time between vehicles
- **Speeds:** Individual and aggregated
- **Flow:** Vehicles per second (per lane or region)
- **Density:** Vehicles per length segment
- **Speed-Density-Flow:** Relationships between main traffic metrics

