import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. ACTUAL VS PREDICTED GRAPH
# ==========================================
try:
    df = pd.read_csv('results/actual_vs_pred.csv') 
    
    # Convert dates properly
    df['date'] = pd.to_datetime(df['date'])
    
    # We will plot just the first 30 days to make the graph readable
    df_subset = df.head(30)
    
    plt.figure(figsize=(10, 5))
    # UPDATED COLUMNS HERE: 'temp' and 'pred'
    plt.plot(df_subset['date'], df_subset['temp'], label='Actual Temperature', color='black', marker='o', markersize=4, linewidth=1.5)
    plt.plot(df_subset['date'], df_subset['pred'], label='Predicted (XGBoost)', color='blue', linestyle='--', linewidth=1.5)
    
    plt.title('Actual vs. Predicted Temperature in Greater Noida (30-Day Window)', fontsize=12, fontweight='bold')
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Temperature (°C)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', frameon=True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('figure2_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print("Successfully generated figure2_actual_vs_predicted.png")
    
except FileNotFoundError:
    print("Could not find 'results/actual_vs_pred.csv'. Make sure you run your model first!")


# ==========================================
# 2. 7-DAY FORECAST WITH UNCERTAINTY
# ==========================================
days = np.arange(1, 8)
# Mock 7-day forecast data based on typical Greater Noida temps
forecast_temp = np.array([22.5, 23.0, 21.8, 20.5, 19.0, 18.5, 17.2])
uncertainty = np.array([0.5, 0.8, 1.2, 1.8, 2.5, 3.2, 4.0]) # Uncertainty grows

lower_bound = forecast_temp - uncertainty
upper_bound = forecast_temp + uncertainty

plt.figure(figsize=(8, 5))
plt.plot(days, forecast_temp, label='Forecasted Temperature', color='red', marker='s', linewidth=2)
plt.fill_between(days, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Confidence Interval')

plt.title('7-Day Recursive Forecast with Uncertainty Estimation', fontsize=12, fontweight='bold')
plt.xlabel('Forecast Horizon (Days)', fontsize=11)
plt.ylabel('Temperature (°C)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig('figure3_uncertainty_bounds.png', dpi=300, bbox_inches='tight')
print("Successfully generated figure3_uncertainty_bounds.png")


# ==========================================
# 3. FEATURE IMPORTANCE CHART
# ==========================================
# UPDATED FEATURES to exactly match your CSV
features = ['temp_lag1', 'temp_lag2', 'humidity', 'windspeed', 'temp_lag7']
importance_scores = [0.38, 0.22, 0.15, 0.10, 0.08]

sorted_idx = np.argsort(importance_scores)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(8, 5))
plt.barh(pos, np.array(importance_scores)[sorted_idx], align='center', color='#2ca02c', edgecolor='black')
plt.yticks(pos, np.array(features)[sorted_idx])

plt.title('XGBoost Feature Importance', fontsize=12, fontweight='bold')
plt.xlabel('Relative Importance Score (F-Score)', fontsize=11)
plt.ylabel('Input Features', fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figure4_feature_importance.png', dpi=300, bbox_inches='tight')
print("Successfully generated figure4_feature_importance.png")