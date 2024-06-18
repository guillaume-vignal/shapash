# Enhancing Shapash Feature/Contribution Plots: Jittering, Smart Selection, and Volume Representation

Shapash is an innovative library for **machine learning interpretability**, offering various tools to help understand model predictions. Three notable enhancements in Shapash's **feature/contribution plots** include **jittering points** on violin plots, **smart selection** for diverse class representation, and **volume representation** via curves and bars. These features significantly improve the **clarity** and **interpretability** of the plots, making it easier for users to draw **insights** from the data.

## 1. Jittering Points on Violin Plot

**Violin plots** in Shapash provide a detailed distribution of **feature contributions**, but overlapping points can obscure individual values. **Jittering**, or adding random noise to the points, disperses them slightly along the x-axis. This prevents overlap and makes each point distinctly visible.

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

1. **Binning Data and Calculating Percentages**: The data is binned into intervals, and the percentage of points in each interval is calculated. This involves:
   - Using `pd.cut` to bin the data.
   - Counting the number of points in each interval with `value_counts`.
   - Calculating the percentage of total points in each interval and sorting these percentages.

2. **Mapping Percentages to Original Data Points**: The calculated percentages are mapped back to the original data points using the bin intervals, resulting in a `percentage_series`.

3. **Generate Jitter**: Random noise (jitter) is generated using a normal distribution. The mean and standard deviation of this noise can be adjusted to control the spread of the jitter.

4. **Handle NaNs**: If any NaNs are found in the `percentage_series`, they are replaced with 1 to ensure they do not interfere with the jittering process.

5. **Adjust Jitter for Positive/Negative Sides**:
   - If the side is either **"negative"** or **"positive"**, the jitter is made non-negative using the `np.abs` function.
   - Jitter values are clipped to a specified range using `np.clip`.

6. **Apply Directional Adjustment**:
   - If the side is **"negative"**, the jitter values are multiplied by -1 to ensure they are on the left side (class 0) in a classification context.
   - For the **positive** side (class 1), the jitter remains as is.

7. **Combine Jitter with Features**: Finally, the jittered points are calculated by adding the jitter (scaled by `percentage_series` and clipped to [-0.5, 0.5]) to the original numerical feature values. This disperses the points along the x-axis, making each point more distinct.

### Summary

The purpose of jittering in this context is to **enhance the visual clarity** of violin plots by preventing points from overlapping. This is achieved by **adding controlled random noise** to the data points, ensuring that they are spread out and more easily distinguishable. In classification tasks, this method helps to **clearly separate points representing different classes**, with **negative** values indicating points with a prediction of **class 0** on the left and **positive** values indicating points with a prediction of **class 1** on the right.

<div style="display: flex; justify-content: space-between;">
  <img src="img_medium/jittering1.png" alt="Violin Plot Without Jittering" style="width: 45%;"/>
  <img src="img_medium/jittering2.png" alt="Violin Plot With Jittering" style="width: 45%;"/>
</div>

### Value Derived from Jittering

- **Enhanced Visibility:** Points are more spread out, making it easier to distinguish individual contributions.
- **Better Interpretation:** Users can accurately see the distribution and density of data points without overlap.
- **Improved Aesthetics:** The plot looks less cluttered and more visually appealing.

## 2. Smart Selection for Diverse Class Representation

Shapash employs a **smart sampling strategy** to ensure a **diverse representation of classes** in the dataset. This approach involves **clustering the data** and sampling points from each cluster. By doing so, it avoids **over-representation** of any particular class and ensures that the selected points represent the **overall distribution** of the data.

Here's the function that handles smart selection:

```python
def _subset_sampling(
    self, selection=None, max_points=2000, col=None, col_value_count=0
):
    if col_value_count > 10:
        from sklearn.cluster import MiniBatchKMeans

        # Clustering data using MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=10, random_state=0)
        kmeans.fit(data[[col]] if col else data)
        data["group"] = kmeans.predict(data)
    else:
        # Grouping data based on index or column value
        data["group"] = (
            data.index % 10 if col is None else data[col].apply(lambda x: int(x % 10))
        )

    idx_list = []
    for group in data["group"].unique():
        data_group = data[data["group"] == group]
        sample_size = min(len(data_group), max_points // 10)
        idx_list += data_group.sample(n=sample_size, random_state=0).index.to_list()
    return idx_list
```

### How It Works

The smart selection process begins by evaluating the **number of unique values** in a specified column (`col_value_count`). If this number is greater than 10, the data is clustered using the **MiniBatchKMeans** algorithm from the `sklearn` library. The algorithm creates 10 clusters, and each data point is assigned to one of these clusters.

If the number of unique values is 10 or fewer, a simpler approach is used: data points are grouped based on their index or a specific column value.

1. **Clustering with MiniBatchKMeans**:
- If there are more than 10 unique values, `MiniBatchKMeans` clusters the data into 10 groups.
- Each data point is assigned a cluster label stored in the "group" column.

2. **Grouping without Clustering**:
- If there are 10 or fewer unique values, data points are assigned to groups based on their index or a specific column value.

After grouping, the function samples points from each group to ensure that the final selection is diverse and representative of the entire dataset.

### Summary

This code ensures a balanced representation of different classes in the sampled data, enhancing the interpretability and reliability of Shapash's feature/contribution plots.

<div style="display: flex; justify-content: space-between;">
  <img src="img_medium/smart_subset1.png" alt="Violin Plot Without Smart Selection" style="width: 45%;"/>
  <img src="img_medium/smart_subset2.png" alt="Violin Plot With Smart Selection" style="width: 45%;"/>
</div>

### Value Derived from Smart Selection

- **Balanced Representation:** Ensures that different classes are fairly represented, leading to more reliable interpretations.
- **Avoids Bias:** Prevents the plot from being dominated by majority classes, highlighting contributions from minority classes.
- **Efficient Sampling:** Even with large datasets, this method effectively samples a manageable number of points without losing representativeness.

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
fig.add_trace(density_plot)
```

### How It Works

For **continuous variables**, the code uses **Kernel Density Estimation (KDE)** to create a smooth curve representing the data distribution. The KDE bandwidth is calculated based on the range of feature values, ensuring a balance between smoothness and detail. The y-values are scaled to fit within the plot.

For **discrete variables**, the code generates bars to show the frequency of each unique value. These bars are normalized to the total count of the feature values, providing a proportionate representation.

### Summary

This code provides a comprehensive visualization of the feature distributions, helping users to better understand the volume and impact of different features on the model's predictions.

<div style="display: flex; justify-content: space-between;">
  <img src="img_medium/density_curve1.png" alt="Scatter Plot Without Density Curve" style="width: 45%;"/>
  <img src="img_medium/density_curve2.png" alt="Scatter Plot With Density Curve" style="width: 45%;"/>
</div>

## Value Derived from Volume Representation

- **Intuitive Understanding:** Continuous variables' distributions are shown smoothly, while discrete variables are clearly delineated.
- **Data Density Insights:** Users can quickly grasp where data points are concentrated and how they contribute to predictions.
- **Improved Readability:** The use of curves and bars simplifies the visual interpretation, making it easy to identify key patterns and anomalies.

## Conclusion

The enhancements in Shapash's feature/contribution plots—jittering points, smart selection for diverse class representation, and volume representation via curves and bars—significantly boost the interpretability and usability of the visualizations. These features allow users to gain deeper insights into their models' behavior, leading to more informed decision-making and better understanding of machine learning models. By making these visualizations clearer and more representative, Shapash continues to be a valuable tool for data scientists and analysts.
