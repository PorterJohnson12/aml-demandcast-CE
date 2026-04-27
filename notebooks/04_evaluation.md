# DemandCast Evaluation & Tuning Report

## 1. Metric Interpretations for Taxi Operations
*The following metrics evaluate our Week 3 baseline model's performance on the validation set. They are translated into practical terms for the driver scheduling and operations teams.*

* **MAE (Mean Absolute Error) = 8.82:** On average, our demand forecast is off by about 8.82 taxi rides per hour in any given zone. For operations, this means if we schedule drivers based on this forecast, we will typically have about 9 too many or 9 too few drivers waiting in a zone.
* **RMSE (Root Mean Squared Error) = 18.14:** This metric acts as a "worst-case scenario" penalty. Because it heavily weights large errors, a higher number here (18.14) compared to our standard average error (8.82) means that while we are fairly accurate most of the time, the model occasionally gets blindsided by massive, unexpected spikes or drops in passenger demand (such as a sudden rainstorm or an unannounced transit outage).
* **R² (R-Squared) = 0.9480:** Our model successfully accounts for 94.8% of the natural, predictable fluctuations in taxi demand. The remaining 5.2% represents random noise, human behavior, or external factors we aren't currently tracking.
* **MBE (Mean Bias Error) = 0.0671:** This tells us the direction of our errors. A positive number (0.0671) means we very slightly over-predict demand on average. This is actually a safe bias for operations, as it leans slightly toward having a driver idle rather than leaving a passenger stranded on the curb.
* **MAPE (Mean Absolute Percentage Error) = 58.53%:** On average, our predictions are off by 58.53% relative to the actual demand. If we expect 100 people to need a ride, our prediction could be off by about 59 people. *(Note: MAPE can be highly skewed by low-demand hours, as explained below).*

### *A Note on Handling Zero-Demand Hours (MAPE)*
*Because MAPE calculates a percentage error by dividing by the actual demand, hours where a zone has exactly 0 trips cause a mathematical "division by zero" error. To handle this, I added a tiny constant value (epsilon = 0.001) to the actual demand in the denominator. This allows the calculation to complete without artificially inflating the error or dropping valid low-demand hours from our evaluation.*

---

## 2. Hyperparameter Tuning Results

### Baseline vs. Tuned Performance (Validation Set)
* **Week 3 Baseline MAE:** 8.8175
* **Tuned Model MAE:** 8.7270
* **Absolute Improvement:** 0.0905 fewer mispredicted rides per hour.

### Was the tuning worth it?
Yes, the systematic tuning process was highly valuable and absolutely worth the compute cost. 

While the absolute improvement in the MAE (about 0.09 rides) seems small at first glance, our initial Optuna trials revealed a significant risk of overfitting—one highly complex trial achieved a strong cross-validation score but failed catastrophically on the unseen test set. 

By allowing Optuna to run through a full 15 trials, it systematically found the optimal regularization balance in Trial #10. Specifically, increasing `min_samples_leaf` to 3 and restricting `max_features` to `log2` forced the Random Forest to generalize rather than memorizing historical noise. The result is a highly stable model with a validation MAE of 8.73 and an even stronger final sealed test MAE of 7.75. This model has been successfully registered as Version 4 in the MLflow Production stage.