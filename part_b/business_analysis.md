# Part B — Business Case Analysis
## Scenario: Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target Variable:** `items_sold` — the number of items sold at a store in a given month under a specific promotion.

**Candidate Input Features:**
- Store attributes: `store_size`, `location_type`, `monthly_footfall`, `competition_density`
- Promotion type: `flat_discount`, `bogo`, `free_gift`, `category_offer`, `loyalty_points`
- Calendar features: `month`, `is_weekend`, `is_festival`, `season`
- Customer demographics: age distribution, income band of local area
- Historical performance: average items sold in past 3 months, promotion history

**Type of ML Problem:** This is a **supervised regression problem**. The goal is to predict a continuous numeric outcome (items sold) for each store-month-promotion combination, and then select the promotion that yields the highest predicted items sold. It is not a classification problem because we are not predicting a category — we are predicting a quantity. It is supervised because we have labelled historical data where we know what promotion was run and how many items were sold.

---

### B1(b) — Why Items Sold is Better Than Revenue

Using total sales revenue as the target variable introduces a confounding factor: revenue is directly influenced by item price, which varies across product categories and discount levels. A Flat Discount promotion may generate lower revenue per unit sold but drive significantly higher volume — and the business goal is to maximise footfall, basket size, and repeat visits, not just short-term revenue.

Items sold (sales volume) is a more reliable target because it isolates the behavioural response to the promotion from pricing effects. A high revenue figure could simply reflect expensive items being sold in small quantities, not promotion effectiveness.

**Broader principle:** This illustrates the importance of choosing a target variable that directly represents the business outcome you want to change. Proxy metrics that conflate multiple effects (like revenue conflating price and volume) can mislead the model into learning spurious patterns and produce recommendations that optimise the wrong thing.

---

### B1(c) — Alternative to a Single Global Model

A single global model across all 50 stores would force the model to find one set of relationships between features and items sold that works for all stores simultaneously. However, urban stores with high footfall and strong competition respond very differently to promotions than rural stores with loyal, lower-frequency customers.

**Proposed strategy — Cluster-then-Model (Segmented Modelling):**
1. First, cluster stores into groups based on their attributes (location type, size, footfall, competition density) using K-Means or hierarchical clustering.
2. Train a separate regression model for each cluster of stores.

This approach allows each sub-model to learn promotion-response patterns specific to stores with similar characteristics. An alternative is a **mixed-effects model** or **hierarchical model** that includes store-level random effects, capturing both shared patterns and individual store behaviour within a single framework. Both approaches are superior to a single global model because they respect the structural heterogeneity in the data.

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables

The four source tables are: `transactions`, `store_attributes`, `promotion_details`, and `calendar`.

**Join strategy:**
1. Start with the `transactions` table as the base (one row per transaction).
2. Left-join `store_attributes` on `store_id` to bring in store size, location type, footfall, and competition density.
3. Left-join `promotion_details` on `promotion_id` to bring in the type and mechanics of the promotion active at each store on each date.
4. Left-join `calendar` on `date` to bring in weekend flags, festival flags, and month/season indicators.

**Grain of the final dataset:** One row = one store on one day (store-day grain). This gives the model daily granularity. Alternatively, if data is sparse, aggregate to store-month grain (one row = one store in one month), computing total items sold and the dominant promotion type for that month.

**Aggregations before modelling:** Compute `total_items_sold` per store per month, `avg_basket_size`, `promotion_days_active`, and `footfall_in_month`. These aggregations reduce noise from day-to-day fluctuations and create stable monthly signals.

---

### B2(b) — EDA Strategy

**Analysis 1 — Items Sold Distribution by Promotion Type (Box Plot):**
Plot the distribution of `items_sold` for each of the 5 promotion types. This reveals which promotions have higher median sales, and whether any promotion types have high variance (inconsistent performance). If BOGO shows a bimodal distribution, it may perform very well in some store types and poorly in others — flagging the need for interaction features.

**Analysis 2 — Promotion Performance by Location Type (Grouped Bar Chart):**
Compare average items sold per promotion type, broken down by `location_type` (urban, semi-urban, rural). This directly tests whether promotion effectiveness varies by location — if Flat Discount drives more sales in urban stores but Loyalty Points works better in rural stores, this justifies the segmented modelling strategy and the inclusion of promotion × location interaction features.

**Analysis 3 — Temporal Trend Analysis (Line Plot):**
Plot total items sold over time aggregated by month. This reveals seasonality patterns (e.g., peaks in festive months, dips in off-season months), trends (growing or declining overall sales), and any anomalies. Findings directly influence feature engineering — strong seasonality would justify including `month` and `is_festival` as features, and might suggest a time-series decomposition approach.

**Analysis 4 — Correlation Heatmap of Numeric Features:**
Compute correlations between `items_sold` and all numeric features: `competition_density`, `store_size` (encoded), `footfall`, `is_weekend`, `is_festival`. High correlation with `is_festival` would confirm its importance as a feature. Multicollinearity between predictors (e.g., `footfall` and `store_size`) would inform feature selection to avoid redundant features in linear models.

---

### B2(c) — Handling the 80% No-Promotion Imbalance

If 80% of transactions occurred without any promotion, a naive model may learn to ignore promotion type entirely — defaulting to predicting the baseline sales level and achieving reasonable accuracy by doing so. This means the model captures little of the promotion-driven signal that matters most to the business.

**Steps to address this:**
1. **Stratified sampling:** Ensure the train-test split preserves the proportion of promotion vs no-promotion records in both sets, so evaluation metrics reflect performance on both groups.
2. **Separate modelling:** Train one model specifically on promoted transactions to learn the differential effect of each promotion type, and use the no-promotion records to establish a baseline. The business question is the uplift — how much more does each promotion sell vs the baseline — not the absolute level.
3. **Feature engineering:** Create an `is_promoted` binary feature and include it explicitly so the model can learn the baseline vs promotional states separately.
4. **Reweighting:** Apply higher sample weights to promoted transactions during training so the model pays more attention to the minority group that carries the decision-relevant signal.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split Strategy and Evaluation Metrics

**Split strategy:** With 3 years of monthly data across 50 stores, a temporal split is essential. Use the first 2.5 years (30 months) as training data and the final 6 months as the test set. This ensures the model is never evaluated on data from a time period it was trained on, accurately reflecting real deployment conditions.

**Why random split is inappropriate:** Monthly store data has strong temporal dependencies — sales in December are influenced by November trends, festive seasons repeat annually, and promotions are often run in sequences. A random split would scatter future records into the training set, creating data leakage. The model would appear to predict well in evaluation but fail in production because it implicitly used future information during training.

**Evaluation metrics:**
- **RMSE (Root Mean Squared Error):** Measures average prediction error in the same units as items sold. Penalises large errors more heavily — useful because a large prediction miss on a high-footfall store in December could lead to stockout or overstock, both costly.
- **MAE (Mean Absolute Error):** Average absolute deviation between predicted and actual items sold. More interpretable to business stakeholders — "our model's predictions are off by X items on average."
- **MAPE (Mean Absolute Percentage Error):** Percentage error relative to actual sales. Useful for comparing performance across stores of very different sizes — a 50-item error is large for a small rural store but trivial for a large urban outlet.

---

### B3(b) — Investigating Different Recommendations via Feature Importance

To investigate why the model recommends Loyalty Points for Store 12 in December and Flat Discount in March, extract and compare the model's input features for both predictions:

1. **Pull the feature vectors** for Store 12 in December and March. The key differences will likely be `month` (12 vs 3), `is_festival` (December likely has festival flags), `footfall` (typically higher in December), and `competition_density`.

2. **Use feature importance from the Random Forest** to identify which features most influence the prediction. If `is_festival` has high importance and December has `is_festival=1`, the model has learned that loyalty programs work well during festive periods — perhaps customers respond to points accumulation when they are already in a gift-buying mindset.

3. **Communicate to marketing** using a SHAP (SHapley Additive exPlanations) waterfall chart for each prediction, showing how each feature pushed the prediction up or down from the baseline. This translates the model's logic into business language: "In December, the high festive activity and increased footfall make Loyalty Points the strongest driver of items sold. In March, which is a low-footfall off-season month, a Flat Discount is predicted to better overcome purchase hesitancy."

---

### B3(c) — End-to-End Deployment Process

**Saving the model:**
Use `joblib.dump(pipeline, 'promotion_model.pkl')` to serialise the entire scikit-learn pipeline (including the preprocessor and model) to disk. Store this file in a versioned model registry (e.g., MLflow or a cloud storage bucket with version tags) so previous model versions can be rolled back if needed.

**Preparing and feeding new monthly data:**
At the start of each month, an automated data pipeline (e.g., an Airflow DAG or a scheduled cloud function) collects the previous month's transaction data, joins it with the store attributes and calendar tables using the same logic as training, engineers the same date and contextual features, and feeds the prepared feature matrix into the loaded pipeline using `pipeline.predict(X_new)`. The output is one recommended promotion per store for the upcoming month.

**Monitoring for model degradation:**
1. **Prediction distribution monitoring:** Track the distribution of predicted items sold each month. A sudden shift in the distribution (e.g., all stores predicted to have much higher or lower sales) may indicate data drift — changes in the input data distribution that the model was not trained on.
2. **Performance tracking:** After each month, once actual sales are recorded, compute RMSE and MAE for that month and plot them over time. A rising error trend signals model degradation.
3. **Concept drift detection:** Use statistical tests (e.g., Population Stability Index on key features like `promotion_type` frequency or `footfall`) to detect when the relationship between features and sales has changed — for example, if a new competitor enters a market and permanently changes shopping behaviour.
4. **Retraining trigger:** Define a threshold (e.g., RMSE exceeds 1.5× the training baseline for two consecutive months) that automatically flags the model for retraining on the most recent 30 months of data using the same pipeline and hyperparameters.
