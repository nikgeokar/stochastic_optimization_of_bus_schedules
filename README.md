# Robust Stochastic Optimization for Urban Bus Scheduling Using Graph Neural Networks

## 📌 Project Overview

This project presents an advanced framework for optimizing urban bus schedules by combining stochastic optimization with state-of-the-art Graph Neural Network (GNN)-based traffic forecasts. The developed methodology effectively reduces passenger waiting times by addressing uncertainties in traffic conditions and passenger demand, ensuring more robust and reliable public transportation.

---

## 🚀 Features

- **Probabilistic Traffic Forecasting:** Graph Neural Networks (GNNs) accurately forecast traffic conditions.
- **Realistic Passenger Demand Simulation:** Spatially- and temporally-aware synthetic passenger demand generation.
- **Stochastic Schedule Optimization:** Minimizes passenger waiting gaps, accounting for uncertainties.
- **Operational Constraints Management:** Respects real-world operational constraints like bus availability, driver schedules, stop importance, and easily accommodates custom organizational constraints.

---

## 📂 Project Structure
├── Network_Structure.ipynb # Road network visualization <br>
├── Traffic_Prediction_GNN.ipynb # GNN-based traffic forecasting<br>
├── Passengers_Demand.ipynb # Passenger demand simulation<br>
├── Bus_Stop_Importance.ipynb # Calculation of stop importance weights<br>
├── Optimizer.ipynb # Stochastic optimization framework<br>
├── Passengers_Waiting_Time_Calculation.ipynb # Pre-computation of passenger waiting gaps<br>
├── Comparison_popular_ml_algo.ipynb # Comparative ML model analysis (XGBoost, LSTM, RF)<br>
├── GNN_Results_Analysis.ipynb # GNN performance analysis and results<br>
└── Graph_Data_Handler.py # Utility for graph data handling<br>


## 🔧 Technical Stack

- **Python**, **Jupyter Notebooks**
- **IBM DOcplex (CPLEX)** optimization library
- **PyTorch Geometric** (GNN Implementation)
- **NumPy, Pandas, Matplotlib** (Data Analysis and Visualization)

---

## 🛠️ Methodology

Our methodology involves these core steps:

1. **Graph-based Traffic Forecasting:** Leveraging GNNs for probabilistic forecasting of road-segment speeds.
2. **Passenger Demand Modeling:** Synthetic data simulation based on spatial and temporal influences.
3. **Pre-computation of Passenger Waiting Gaps:** Enhancing computational efficiency.
4. **Stochastic Optimization:** Minimizing passenger waiting times while complying with various operational and custom organizational constraints.

---

## 📈 Evaluation & Results

- Our stochastic optimization approach significantly outperformed heuristic and random scheduling baselines.
- Demonstrated exceptional robustness under simulated traffic anomalies, confirming its adaptability in real-world scenarios.

---

## 🚧 Future Work

- Incorporate real passenger data and develop dedicated GNNs for demand prediction.
- Expand the set of operational constraints (driver shifts, vehicle maintenance).
- Introduce handling for special events, holidays, and dynamic road network changes.
- Explore potential for real-time adaptive scheduling strategies.

---

## 📖 Detailed Documentation

Refer to the detailed thesis documentation provided in the repository for comprehensive theoretical and experimental insights
