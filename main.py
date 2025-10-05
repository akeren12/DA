import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import io

# Page config
st.set_page_config(page_title="Crime Pattern Analyzer", layout="wide", page_icon="🚔")

st.title("🚔 Crime Pattern Analysis with K-Means Clustering")
st.markdown("""
This interactive app uses K-Means clustering to identify crime hotspots, temporal trends, and patterns from your data. 
Upload a CSV with features like latitude, longitude, crime_type (0=Theft,1=Assault,2=Fraud), hour (0-23), and frequency.
Discover insights for resource allocation and preventive policing!
""")

# Sidebar for inputs
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Crime Data (CSV)", type="csv")
use_sample = st.sidebar.checkbox("Use Sample Data", value=True) if not uploaded_file else False

if use_sample and not uploaded_file:
    # Load synthetic data
    try:
        df = pd.read_csv("synthetic_data.csv")
        st.sidebar.success("Using synthetic data (1000 records).")
    except FileNotFoundError:
        st.sidebar.error("Sample data not found. Generate it or upload your own.")
        st.stop()
else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df)} records from upload.")
    else:
        st.sidebar.warning("Upload data or check 'Use Sample Data'.")
        st.stop()

# Display raw data
st.subheader("1. Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.dataframe(df.head(), use_container_width=True)
with col2:
    st.write(f"**Dataset Shape:** {df.shape}")
    st.write("**Columns:**", list(df.columns))
    st.write("**Missing Values:**")
    st.dataframe(df.isnull().sum())

# Preprocessing section
st.subheader("2. Data Preprocessing")
features = st.multiselect("Select Features for Clustering", df.columns.tolist(), 
                          default=['latitude', 'longitude', 'crime_type', 'hour', 'frequency'])
scale_features = st.checkbox("Scale Features (Recommended for K-Means)", value=True)

if st.button("Preprocess Data"):
    # Cleaning
    df_clean = df[features].dropna().drop_duplicates()
    X = df_clean.values
    
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.success("Data cleaned and scaled!")
    else:
        X_scaled = X
        st.success("Data cleaned (no scaling).")
    
    # Store in session state
    st.session_state.X_scaled = X_scaled
    st.session_state.df_clean = df_clean
    st.session_state.scaler = scaler if scale_features else None
    st.session_state.features = features

# Clustering section (only if preprocessed)
if 'X_scaled' in st.session_state:
    st.subheader("3. Apply K-Means Clustering")
    
    # Parameter tuning
    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)
    with col2:
        max_iter = st.slider("Max Iterations", min_value=50, max_value=500, value=300)
    
    if st.button("Run K-Means"):
        X_scaled = st.session_state.X_scaled
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=max_iter, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add to DataFrame
        df_results = st.session_state.df_clean.copy()
        df_results['cluster'] = cluster_labels
        
        # Metrics
        sil_score = silhouette_score(X_scaled, cluster_labels)
        st.session_state.df_results = df_results
        st.session_state.kmeans = kmeans
        st.session_state.sil_score = sil_score
        
        st.success(f"Clustering complete! Silhouette Score: {sil_score:.3f} (Higher is better; >0.5 = good separation).")
        
        # Under-the-hood explanation (your original steps)
        with st.expander("🔍 What Happens Under the Hood?"):
            st.markdown("""
            1. **Choose K**: You selected {k} clusters (e.g., 3 hotspots).
            2. **Initialize Centroids**: Randomly pick {k} starting points in the feature space.
            3. **Assign Points**: Each crime record goes to the nearest centroid using Euclidean distance:  
               $$ d(\\mathbf{{x}}_i, \\mathbf{{c}}_j) = \\sqrt{{\\sum_{{m=1}}^{{p}} (x_{{i,m}} - c_{{j,m}})^2}} $$  
               (p = number of features).
            4. **Update Centroids**: Recalculate means: $$ \\mathbf{{c}}_j = \\frac{{1}}{{n_j}} \\sum_{{\\mathbf{{x}}_i \\in C_j}} \\mathbf{{x}}_i $$.
            5. **Repeat**: Until convergence (assignments stabilize).
            6. **Output**: {k} clusters with centroids as pattern centers (e.g., average location/time).
            
            In simple terms: Similar crimes (location, type, time) group into natural patterns like "night thefts downtown."
            """.format(k=k))

    # Results section (only if clustered)
    if 'df_results' in st.session_state:
        df_results = st.session_state.df_results
        kmeans = st.session_state.kmeans
        sil_score = st.session_state.sil_score
        
        st.subheader("4. Cluster Insights & Visualizations")
        
        # Cluster Summary Table
        cluster_summary = df_results.groupby('cluster').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'crime_type': lambda x: x.mode()[0] if not x.empty else None,
            'hour': 'mean',
            'frequency': 'mean',
            'cluster': 'size'
        }).round(2)
        cluster_summary.columns = ['Avg Latitude', 'Avg Longitude', 'Dominant Crime Type', 'Avg Hour', 'Avg Frequency', 'Size']
        cluster_summary['Dominant Crime Type'] = cluster_summary['Dominant Crime Type'].map({0: 'Theft', 1: 'Assault', 2: 'Fraud'})
        st.dataframe(cluster_summary, use_container_width=True)
        
        # Example Insights
        for i in range(k):
            row = cluster_summary.iloc[i]
            st.write(f"**Cluster {i}**: {row['Dominant Crime Type']} hotspot at ({row['Avg Latitude']:.4f}, {row['Avg Longitude']:.4f}), "
                     f"avg. {row['Avg Hour']:.1f} hour, frequency {row['Avg Frequency']:.1f}. Size: {row['Size']} crimes.")
        
        # Visualizations (Interactive with Plotly)
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geographical Hotspots", "⏰ Temporal Patterns", "📊 Crime Types", "📈 Elbow/Silhouette"])
        
        with tab1:
            # Interactive Map with Folium (clusters as markers)
            m = folium.Map(location=[df_results['latitude'].mean(), df_results['longitude'].mean()], zoom_start=12)
            for cluster in range(k):
                cluster_data = df_results[df_results['cluster'] == cluster]
                for idx, row in cluster_data.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3, popup=f"Cluster {cluster}: {row['crime_type']}",  # Crime type popup
                        color=plt.cm.viridis(cluster / k), fill=True, fillOpacity=0.7
                    ).add_to(m)
            folium_static(m, width=700, height=500)
        
        with tab2:
            # Histogram: Hours by cluster
            fig = px.histogram(df_results, x='hour', color='cluster', nbins=24, 
                               title="Crime Occurrences by Hour and Cluster", 
                               labels={'hour': 'Hour of Day', 'count': 'Occurrences'})
            fig.update_layout(bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Pie charts for each cluster
            fig = make_subplots(rows=1, cols=k, specs=[[{'type': 'pie'}]*k], subplot_titles=[f'Cluster {i}' for i in range(k)])
            for i in range(k):
                cluster_data = df_results[df_results['cluster'] == i]['crime_type'].value_counts()
                labels = ['Theft', 'Assault', 'Fraud']
                values = [cluster_data.get(0, 0), cluster_data.get(1, 0), cluster_data.get(2, 0)]
                fig.add_trace(go.Pie(labels=labels, values=values, name=f'Cluster {i}'), row=1, col=i+1)
            fig.update_layout(title_text="Crime Type Distribution by Cluster", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Elbow and Silhouette (pre-computed for range)
            if st.button("Compute Elbow/Silhouette Curves"):
                X_scaled = st.session_state.X_scaled
                inertias, sil_scores = [], []
                for ck in range(2, 11):
                    km = KMeans(n_clusters=ck, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)
                    sil_scores.append(silhouette_score(X_scaled, km.labels_))
                
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(x=list(range(2,11)), y=inertias, mode='lines+markers', name='Inertia'))
                fig_elbow.update_layout(title='Elbow Method', xaxis_title='K', yaxis_title='Inertia')
                st.plotly_chart(fig_elbow, use_container_width=True)
                
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(x=list(range(2,11)), y=sil_scores, mode='lines+markers', name='Silhouette Score'))
                fig_sil.update_layout(title='Silhouette Score', xaxis_title='K', yaxis_title='Score')
                st.plotly_chart(fig_sil, use_container_width=True)
        
        # Export
        st.subheader("5. Export Results")
        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        st.download_button("Download Clustered Data (CSV)", csv_buffer.getvalue(), "clustered_crimes.csv", "text/csv")
        
        st.info("💡 **Next Steps**: Use clusters for predictive policing—e.g., increase patrols in high-frequency hotspots at peak hours.")

else:
    st.warning("Preprocess data first to enable clustering.")
