import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sqlite3
import os

# Scikit-learn imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- CONSTANTS ---
DATA_PATH = '../data/Driver_Behavior.csv'
DB_PATH = '../database/drivers_data.db'
COLS_TO_NORMALIZE = [
    'speed_kmph', 'accel_x', 'accel_y', 'brake_pressure',
    'steering_angle', 'throttle', 'lane_deviation',
    'headway_distance', 'reaction_time'
]


def load_and_prepare_db(csv_path, db_path):
    """
    Loads CSV data and creates a SQLite database.
    Acts as the ETL (Extract, Transform, Load) step.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Extract
    df = pd.read_csv(csv_path)

    # Load to SQL (Using a context manager ensures the connection closes)
    # We use one database. Raw data goes to 'drivers', normalized can be calculated on the fly or stored.
    with sqlite3.connect(db_path) as conn:
        df.to_sql('drivers', conn, if_exists='replace', index=False)
        print(f"Data successfully loaded into {db_path} (Table: drivers)")

    return df


def perform_sql_analysis(db_path):
    """
    Executes SQL queries to extract insights from the database.
    """
    with sqlite3.connect(db_path) as conn:
        print("\n--- SQL Analysis ---")

        # 1. Phone Usage Impact
        # Note: We use read_sql directly. No need for .max() on a single-row result.
        q_phone = """
        SELECT 
            AVG(reaction_time) as avg_reaction, 
            AVG(lane_deviation) as avg_deviation
        FROM drivers
        GROUP BY phone_usage
        """
        df_phone_stats = pd.read_sql(q_phone, conn)
        print("Stats by Phone Usage (0=No, 1=Yes):")
        print(df_phone_stats)

        # 2. Potential Accidents (Tailgating + Speeding)
        q_accident = """
        SELECT COUNT(*) as count
        FROM drivers
        WHERE headway_distance < 10 AND speed_kmph > 60
        """
        accidents = pd.read_sql(q_accident, conn).iloc[0]['count']
        print(f"\nPotential accident situations: {accidents}")

        # 3. Aggressive Braking
        q_aggressive = """
        SELECT COUNT(*) as count
        FROM drivers
        WHERE brake_pressure > 80
        """
        aggressive = pd.read_sql(q_aggressive, conn).iloc[0]['count']
        print(f"Aggressive braking incidents: {aggressive}")


def calculate_risk_score(df):
    """
    Feature Engineering: Calculates a custom Risk Score based on physics and behavior.
    """
    # Vectorized calculations using numpy for performance
    # Breaking down the formula for readability

    term_speed = np.maximum(0, df['speed_kmph'] - 90) / 40
    term_phone = df['phone_usage'] * 1.5
    term_lane = np.abs(df['lane_deviation'])
    term_accel_x = np.maximum(0, np.abs(df['accel_x'])) / 3
    term_accel_y = np.maximum(0, np.abs(df['accel_y']) - 2) / 3
    term_brake = np.maximum(0, df['brake_pressure'] - 60) / 40
    term_throttle = np.maximum(0, df['throttle'] - 70) / 30
    term_headway = np.maximum(0, 20 - df['headway_distance']) / 20
    term_reaction = np.maximum(0, df['reaction_time'] - 1.2) / 1.5

    df['risk_score'] = (
            term_speed + term_phone + term_lane +
            term_accel_x + term_accel_y + term_brake +
            term_throttle + term_headway + term_reaction
    )

    print(f"\nMax Risk Score calculated: {df['risk_score'].max():.2f}")
    return df


def perform_clustering(df, cols):
    """
    Performs K-Means clustering.
    Uses StandardScaler because K-Means is distance-based.
    """
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols])

    # Fit K-Means
    kmeans = KMeans(n_clusters=3, random_state=18, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    print("\nClustering completed. Cluster counts:")
    print(df['Cluster'].value_counts())

    return df


def plot_radar_charts(df, cols):
    """
    Generates Radar Charts using Plotly.
    Uses MinMaxScaler to normalize data to [0, 1] range for visualization.
    """
    # Normalize data specifically for the plot (0 to 1 range)
    # We use a separate scaler here because visuals need 0-1, but ML needed Mean-Std.
    min_max_scaler = MinMaxScaler()
    df_plot = df.copy()
    df_plot[cols] = min_max_scaler.fit_transform(df[cols])

    # 1. Cluster Radar Chart
    radar_data = df_plot.groupby('Cluster')[cols].mean()

    fig = go.Figure()
    for cluster_id in radar_data.index:
        fig.add_trace(go.Scatterpolar(
            r=radar_data.loc[cluster_id],
            theta=radar_data.columns,
            fill='toself',
            name=f'Cluster {cluster_id}'
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Driver Profiles by Cluster (Normalized)"
    )
    fig.show()

    # 2. Phone Usage Radar Chart
    radar_phone = df_plot.groupby('phone_usage')[cols].mean()

    fig2 = go.Figure()
    for usage in radar_phone.index:
        label = "With Phone" if usage == 1 else "No Phone"
        fig2.add_trace(go.Scatterpolar(
            r=radar_phone.loc[usage],
            theta=radar_phone.columns,
            fill='toself',
            name=label
        ))

    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Impact of Phone Usage on Driving (Normalized)"
    )
    fig2.show()


def plot_static_graphs(df):
    """
    Generates static Seaborn/Matplotlib graphs.
    """
    # Boxplot: Reaction Time vs Phone
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='phone_usage', y='reaction_time', data=df,palette='rocket')
    plt.title('Reaction Time vs Phone Usage')
    plt.xlabel('Phone Usage (0=No, 1=Yes)')
    plt.ylabel('Reaction Time (s)')
    plt.show()

    # Scatter: Speed vs Risk
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='speed_kmph', y='risk_score', hue='Cluster', palette='viridis', alpha=0.6)
    plt.grid(True, alpha=0.3)
    plt.title('Driver Clusters: Speed vs Risk Score')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Risk Score')
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_data = df[['accel_x', 'brake_pressure']].corr()
    sns.heatmap(corr_data, annot=True, cmap='GnBu')
    plt.title('Correlation: Acceleration vs Braking')
    plt.show()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. ETL
    df_drivers = load_and_prepare_db(DATA_PATH, DB_PATH)

    # 2. SQL Analysis
    perform_sql_analysis(DB_PATH)

    # 3. Feature Engineering
    df_drivers = calculate_risk_score(df_drivers)

    # 4. Machine Learning
    df_drivers = perform_clustering(df_drivers, COLS_TO_NORMALIZE)

    # 5. Visualization
    plot_static_graphs(df_drivers)
    plot_radar_charts(df_drivers, COLS_TO_NORMALIZE)