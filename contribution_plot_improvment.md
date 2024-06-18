# Enhancing Shapash Contribution Plots with Jittering, Smart Selection, and Volume Representation

Shapash is a cutting-edge library for **machine learning interpretability**, offering diverse tools to facilitate understanding of model predictions. Three significant improvements in Shapash's **contribution plots** include **jittering points** on violin plots, **smart selection** for diverse class representation, and **volume representation** via curves and bars. These enhancements greatly enhance the **clarity** and **interpretability** of the plots, enabling users to derive deeper **insights** from their data.

## 1. Jittering Points on Violin Plot

**Violin plots** in Shapash display detailed distributions of **feature contributions**. However, overlapping points can obscure individual values. **Jittering** introduces controlled random noise to disperse points along the x-axis, preventing overlap and ensuring clearer visibility.

Here's the key snippet illustrating jittering:

```python
# Binning data into intervals and calculating the percentage of points in each interval
intervals = pd.cut(data, bins, duplicates="drop")
points_per_interval = intervals.value_counts()
total_points = len(data)
percentage_per_interval = (points_per_interval / total_points).sort_index().to_dict()

# Mapping those percentages to the original data points
percentage_series = intervals.map(percentage_per_interval).to_numpy()

# Creating jittered points
jitter = np.random.normal(mean, std, len(percentage_series))
if np.isnan(percentage_series).any():
    percentage_series.fill(1)

if side in ["negative", "positive"]:
    jitter = np.abs(jitter)

jitter = np.clip(jitter, clip_min, clip_max)

if side == "negative":
    jitter *= -1

jittered_points = numerical_features + np.clip(jitter * percentage_series, -0.5, 0.5)
```

### How It Works

1. **Binning Data and Calculating Percentages**: The data is segmented into intervals, and the percentage of points in each interval is computed.

2. **Mapping Percentages to Data Points**: The calculated percentages are mapped back to the original data points, creating `percentage_series`.

3. **Generating Jitter**: Random noise (`jitter`) is added to the data points to disperse them and avoid overlap.

4. **Handling Class Distinctions**: Depending on whether the side is **"negative"** or **"positive"**, adjustments ensure clear separation between classes.

### Summary

Jittering enhances the clarity of violin plots by dispersing points and making individual contributions more distinguishable. In classification tasks, it helps differentiate predictions for **class 0** (left side, negative values) and **class 1** (right side, positive values).

<div style="display: flex; justify-content: space-between;">
  <img src="img_medium/jittering1.png" alt="Violin Plot Without Jittering" style="width: 45%;"/>
  <img src="img_medium/jittering2.png" alt="Violin Plot With Jittering" style="width: 45%;"/>
</div>

### Value Derived from Jittering

- **Enhanced Visibility**: Points are clearly separated, improving interpretability.
- **Clear Class Differentiation**: Facilitates understanding of class-specific contributions.
- **Visual Appeal**: Reduces clutter, enhancing aesthetic appeal of the plot.

## 2. Smart Sampling for Diverse Class Representation

Shapash utilizes a **smart sampling strategy** to ensure a balanced representation of classes within the dataset. This approach involves clustering data points and sampling from each cluster, thereby avoiding biases towards specific classes and ensuring the selected points reflect the overall data distribution.

Here's the function handling smart selection:

```python
def _intelligent_sampling(self, data, max_points, col_value_count, random_seed):
    """
    Performs intelligent sampling based on the distribution of values in the specified column.
    """
    rng = np.random.default_rng(seed=random_seed)

    # Check if data is numerical data
    is_col_str = True
    if data.dtype.kind in "fc":
        is_col_str = False

    if (col_value_count < len(data) / 20) or is_col_str:
        cluster_labels = data
        cluster_counts = cluster_labels.value_counts()
    else:
        n_clusters = min(100, len(data) // 20)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto")
        cluster_labels = pd.Series(kmeans.fit_predict(data.values.reshape(-1, 1)))
        cluster_counts = cluster_labels.value_counts()

    weights = cluster_counts.apply(lambda x: (x ** 0.5) / x).to_dict()
    selection_weights = cluster_labels.apply(lambda x: weights[x])
    selection_weights /= selection_weights.sum()
    selected_indices = rng.choice(
        data.index.tolist(), max_points, p=selection_weights, replace=False
    )
    return selected_indices
```

### How It Works

The `_intelligent_sampling` function selects a subset of data based on the distribution of values in a specified column. Here’s how it operates:

1. **Data Type Handling**:
   - It checks if the data contains numerical (`float` or `int`) or categorical (`object` or `category`) data.

2. **Condition Check**:
   - If the number of unique values (`col_value_count`) is less than 5% of the total rows in the dataset (`len(data) / 20`) or if the column contains string data, it uses the original column values without clustering (`is_col_str`).

3. **Clustering Approach**:
   - **Numeric Data**: For numeric columns with more than 5% unique values, the function performs KMeans clustering. It determines the number of clusters (`n_clusters`) based on either 100 clusters or a fraction of the dataset size (`len(data[col]) // 20`).
   - **Categorical Data**: No clustering is applied to categorical data; it directly uses the original values.

4. **Cluster Weight Calculation**:
   - If clustering is applied, weights for each cluster are calculated based on the square root of cluster counts (`(x ** 0.5) / x`). This ensures a balanced representation of clusters in the sampling process.

5. **Selection Process**:
   - The function normalizes the calculated weights (`selection_weights`) so that they sum to 1, ensuring proportional selection probabilities.

6. **Random Selection**:
   - Using a random number generator (`rng.choice`), the function selects `max_points` indices from the dataset based on the normalized weights (`selection_weights`). This strategy ensures the selected subset reflects the original data’s distribution.

### Summary

This approach ensures that the selected sample for plotting features represents the dataset's diversity, enhancing the reliability and interpretability of Shapash's visualizations.

<div style="display: flex; justify-content: space-between;">
  <img src="img_medium/smart_subset1.png" alt="Violin Plot Without Smart Selection" style="width: 45%;"/>
  <img src="img_medium/smart_subset2.png" alt="Violin Plot With Smart Selection" style="width: 45%;"/>
</div>

### Value Derived from Smart Selection

- **Balanced Class Representation**: Prevents dominance of any single class in visualizations.
- **Robust Interpretations**: Ensures insights drawn are representative of overall data trends.
- **Scalable Sampling**: Efficiently handles large datasets while maintaining sampling integrity.

## 3. Volume Representation via Curves and Bars

Shapash represents the **volume of data points** using **curves for continuous variables** and **bars for discrete variables**. This dual approach provides an **accurate visual summary** of how features are distributed across their range and how they contribute to the model's predictions.

Here's the code snippet illustrating volume representation:

```python
if feature_values.iloc[:, 0].dtype.kind in "biufc":
    from sklearn.neighbors import KernelDensity

    # Using Kernel Density Estimation for continuous variables
    kde = KernelDensity(
        bandwidth=(feature_values_array.max() - feature_values_array.min()) / 100,
        kernel="epanechnikov",
    ).fit(feature_values_array[:, None])
    xs = np.linspace(min(feature_values_array), max(feature_values_array), 1000)
    log_dens = kde.score_samples(xs[:, None])
    y_upper = np.exp(log_dens) * h / (np.max(np.exp(log_dens)) * 3) + contributions_min
    y_lower = np.full_like(y_upper, contributions_min)
else:
    # Counting values for discrete variables
    feature_values_counts = feature_values.value_counts()
    xs = feature_values_counts.index.get_level_values(0).sort_values()
    y_upper = (
        feature_values_counts.loc[xs] / feature_values_counts.sum()
    ).values.flatten() / 3 + contributions_min
    y_lower = np.full_like(y_upper, contributions_min)

# Creating the plot with either curve or bars
density_plot = go.Scatter(
    x=np.concatenate([pd.Series(xs), pd.Series(xs)[::-1]]),
    y=pd.concat([pd.Series(y_upper), pd.Series(y_lower)[::-1]]),
    fill="toself",
    hoverinfo="none",
    showlegend=False,
    line={"color": self._style_dict["contrib_distribution"]},
)
```

### How It Works

- **Continuous Variables**: Uses **Kernel Density Estimation (KDE)** to create smooth curves representing data distributions. Bandwidth is dynamically adjusted based on feature range.

- **Discrete Variables**: Constructs bars showing frequency distribution of discrete values, normalized to highlight relative proportions.

### Summary

This approach visually represents feature distributions comprehensively, aiding in understanding data volumes and their impacts on model predictions.

<div style="display: flex; justify-content: space-between;">
  <img src="img_medium/density_curve1.png" alt="Scatter Plot Without Density Curve" style="width: 45%;"/>
  <img src="img_medium/density_curve2.png" alt="Scatter Plot With Density Curve" style="width: 45%;"/>
</div>
<div style="display: flex; justify-content: space-between;">
  <img src="img_medium/smart_subset1.png" alt="Violin Plot Without Smart Selection" style="width: 45%;"/>
  <img src="img_medium/smart_subset2.png" alt="Violin Plot With Smart Selection" style="width: 45%;"/>
</div>

### Value Derived from Volume Representation

- **Clear Data Distribution**: Provides intuitive insights into feature distributions.
- **Impact Analysis**: Visualizes how features contribute to model predictions.
- **Enhanced Clarity**: Simplifies interpretation of complex data patterns.

## Conclusion

Enhancements in Shapash's contribution plots—jittering points, smart selection for diverse class representation, and volume representation via curves and bars—significantly improve the interpretability and usability of visualizations. These advancements empower users to gain deeper insights into model behaviors, facilitating informed decision-making in machine learning applications. By enhancing visualization clarity and representativeness, Shapash continues to be an invaluable tool for data scientists and analysts.
