import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

data = pd.read_csv('utility.csv')

data['total_cost'] = (data['water_liter'] * data['water_rate'] +
                       data['electricity_kwh'] * data['electricity_rate'] +
                       data['gas_m3'] * data['gas_rate'])

data['water_per_animal'] = data['water_liter'] / data['household_size']

print(f"total_cost: {data['total_cost'].describe()}")
print(f"\nwater_per_animal: {data['water_per_animal'].describe()}")

def mean(data):
    return sum(data) / len(data)

def median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        return sorted_data[n//2]

def mode(data):
    counts = Counter(data)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    return modes[0] if len(modes) == 1 else modes

def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)

def std_dev(data):
    return variance(data) ** 0.5

def data_range(data):
    return max(data) - min(data)

def quartiles(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    q1 = median(sorted_data[:n//2])
    q2 = median(sorted_data)
    q3 = median(sorted_data[(n+1)//2:])
    return q1, q2, q3

def iqr(data):
    q1, _, q3 = quartiles(data)
    return q3 - q1

def skewness(data):
    m = mean(data)
    s = std_dev(data)
    n = len(data)
    return (n / ((n-1) * (n-2))) * sum(((x - m) / s) ** 3 for x in data)

def kurtosis(data):
    m = mean(data)
    s = std_dev(data)
    n = len(data)
    return (n * (n+1) / ((n-1) * (n-2) * (n-3))) * sum(((x - m) / s) ** 4 for x in data) - (3 * (n-1)**2 / ((n-2) * (n-3)))

def describe_numerical(data, column_name):
    values = [x for x in data[column_name] if pd.notna(x)]
    q1, q2, q3 = quartiles(values)

    stats_dict = {
        'Mean': mean(values),
        'Median': median(values),
        'Mode': mode(values),
        'Std Dev': std_dev(values),
        'Variance': variance(values),
        'Range': data_range(values),
        'Min': min(values),
        'Max': max(values),
        'Q1': q1,
        'Q2': q2,
        'Q3': q3,
        'IQR': iqr(values),
        'Skewness': skewness(values),
        'Kurtosis': kurtosis(values)
    }
    return stats_dict

print("\nQUESTION 1 - GENERAL\n")

numerical_cols = ['avg_temperature_c', 'household_size', 'working_days',
                  'water_liter', 'electricity_kwh', 'gas_m3',
                  'water_rate', 'electricity_rate', 'gas_rate',
                  'total_cost', 'water_per_animal']

print("NUMERICAL COLUMNS - Custom:")
for col in numerical_cols:
    print(f"\n{col.upper()}:")
    stats_result = describe_numerical(data, col)
    for key, value in stats_result.items():
        print(f"  {key}: {value}")

print("\nCATEGORICAL COLUMNS:")
categorical_cols = ['billing_month', 'season', 'ownership_status', 'energy_efficiency_rating']

for col in categorical_cols:
    values = data[col].dropna()
    unique_vals = list(set(values))

    freq = {}
    for val in unique_vals:
        freq[val] = sum(1 for x in values if x == val)

    total = len(values)
    percentages = {k: (v / total) * 100 for k, v in freq.items()}

    print(f"\n{col.upper()}:")
    print(f"  Unique values: {unique_vals}")
    print(f"  Frequencies:")
    for val in sorted(freq.keys(), key=str):
        print(f"    {val}: {freq[val]} ({percentages[val]:.2f}%)")

print("\nLIBRARY:")
print(data[numerical_cols].describe())
print(f"\nSkewness:\n{data[numerical_cols].skew()}")
print(f"\nKurtosis:\n{data[numerical_cols].kurtosis()}")

print("\nQUESTION 2 - GENERAL\n")

def detect_outliers_iqr(data, column):
    q1, _, q3 = quartiles([x for x in data[column] if pd.notna(x)])
    iqr_val = q3 - q1
    lower_bound = q1 - 1.5 * iqr_val
    upper_bound = q3 + 1.5 * iqr_val
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

for col in numerical_cols:
    outliers, lower, upper = detect_outliers_iqr(data, col)
    print(f"{col}: {len(outliers)} outliers (bounds: [{lower:.2f}, {upper:.2f}])")

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].boxplot(data[col].dropna())
    axes[i].set_title(col)
    axes[i].set_ylabel('Value')

axes[-1].axis('off')
plt.tight_layout()
plt.savefig('outlier_boxplots.png')
plt.close()

print("\nQUESTION 3 - GENERAL\n")

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].hist(data[col].dropna(), bins=30, edgecolor='black')
    axes[i].set_title(col)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

axes[-1].axis('off')
plt.tight_layout()
plt.savefig('numerical_histograms.png')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    value_counts = data[col].value_counts()
    axes[i].bar(value_counts.index, value_counts.values)
    axes[i].set_title(col)
    axes[i].set_xlabel('Category')
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('categorical_barcharts.png')
plt.close()

print("\nQUESTION 4 - GENERAL\n")

print("Significance level = 0.05\n")

for col in numerical_cols:
    sample_data = data[col].dropna().sample(min(5000, len(data[col].dropna())))
    stat, p_value = stats.shapiro(sample_data)
    is_normal = "Normal" if p_value > 0.05 else "Tidak Normal"
    print(f"{col}:")
    print(f"  Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    print(f"  Kesimpulan: {is_normal}\n")

print("\nQUESTION 5.1 - SPECIFIC\n")

water_per_animal = data['water_per_animal'].dropna()
n = len(water_per_animal)
sample_mean = water_per_animal.mean()
sample_std = water_per_animal.std()
mu_0 = 3000

print("1. HIPOTESIS:")
print("   H0: mean = 3000")
print("   H1: mean < 3000")

alpha = 0.05
print(f"\n2. TINGKAT SIGNIFIKANSI: alpha = {alpha}")

print(f"\n3. UJI STATISTIK:")
print(f"   t-test")
print(f"   Sample size n = {n}")
print(f"   Sample mean = {sample_mean:.2f}")
print(f"   Sample std = {sample_std:.2f}")

t_critical = stats.t.ppf(alpha, df=n-1)
print(f"   Critical value: {t_critical:.4f}")
print(f"   Daerah kritis: t < {t_critical:.4f}")

print(f"\n4. NILAI UJI STATISTIK:")
t_stat = (sample_mean - mu_0) / (sample_std / np.sqrt(n))
print(f"   t-statistic = {t_stat:.4f}")

p_value = stats.t.cdf(t_stat, df=n-1)
print(f"   p-value = {p_value:.4f}")

print(f"\n5. KEPUTUSAN:")
if p_value < alpha:
    print(f"   TOLAK H0")
    print(f"   Kesimpulan: Rata-rata penggunaan air per hewan kurang dari 3000 liter per bulan")
else:
    print(f"   GAGAL TOLAK H0")
    print(f"   Kesimpulan: Tidak cukup bukti bahwa rata-rata < 3000")

print("\nQUESTION 5.3 - SPECIFIC\n")

ratings = data['energy_efficiency_rating'].dropna()
n = len(ratings)
ab_count = sum((ratings == 'A') | (ratings == 'B'))
p_hat = ab_count / n

print(f"Sample size n = {n}")
print(f"Houses with rating A or B = {ab_count}")
print(f"Sample proportion = {p_hat:.4f}\n")

print("TEST 1: Proporsi > 50%")
print("\n1. HIPOTESIS:")
print("   H0: p = 0.5")
print("   H1: p > 0.5")

alpha = 0.05
p_0 = 0.5
print(f"\n2. TINGKAT SIGNIFIKANSI: alpha = {alpha}")

print(f"\n3. UJI STATISTIK:")
print(f"   Z-test")

z_critical = stats.norm.ppf(1 - alpha)
print(f"   Critical value: {z_critical:.4f}")
print(f"   Daerah kritis: z > {z_critical:.4f}")

print(f"\n4. NILAI UJI STATISTIK:")
z_stat = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
print(f"   z-statistic = {z_stat:.4f}")

p_value = 1 - stats.norm.cdf(z_stat)
print(f"   p-value = {p_value:.4f}")

print(f"\n5. KEPUTUSAN:")
if p_value < alpha:
    print(f"   TOLAK H0")
    print(f"   Kesimpulan: Proporsi rumah dengan rating A/B lebih dari 50%")
else:
    print(f"   GAGAL TOLAK H0")
    print(f"   Kesimpulan: Tidak cukup bukti bahwa proporsi > 50%")

print("\nTEST 2: Proporsi > 60%")

print("\n1. HIPOTESIS:")
print("   H0: p = 0.6")
print("   H1: p > 0.6")

p_0 = 0.6
print(f"\n2. TINGKAT SIGNIFIKANSI: alpha = {alpha}")

print(f"\n3. UJI STATISTIK:")
print(f"   Z-test")

z_critical = stats.norm.ppf(1 - alpha)
print(f"   Critical value: {z_critical:.4f}")
print(f"   Daerah kritis: z > {z_critical:.4f}")

print(f"\n4. NILAI UJI STATISTIK:")
z_stat = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
print(f"   z-statistic = {z_stat:.4f}")

p_value = 1 - stats.norm.cdf(z_stat)
print(f"   p-value = {p_value:.4f}")

print(f"\n5. KEPUTUSAN:")
if p_value < alpha:
    print(f"   TOLAK H0")
    print(f"   Kesimpulan: Proporsi rumah dengan rating A/B lebih dari 60%")
else:
    print(f"   GAGAL TOLAK H0")
    print(f"   Kesimpulan: Tidak cukup bukti bahwa proporsi > 60%")

print("\nQUESTION 6.2 - SPECIFIC\n")

winter_cost = data[data['season'] == 'Winter']['total_cost'].dropna()
summer_cost = data[data['season'] == 'Summer']['total_cost'].dropna()

n_winter = len(winter_cost)
n_summer = len(summer_cost)
mean_winter = winter_cost.mean()
mean_summer = summer_cost.mean()
std_winter = winter_cost.std()
std_summer = summer_cost.std()

print("Data:")
print(f"   Winter: n = {n_winter}, mean = {mean_winter:.2f}, std = {std_winter:.2f}")
print(f"   Summer: n = {n_summer}, mean = {mean_summer:.2f}, std = {std_summer:.2f}")

print("\n1. HIPOTESIS:")
print("   H0: mean_winter = mean_summer")
print("   H1: mean_winter > mean_summer")

alpha = 0.05
print(f"\n2. TINGKAT SIGNIFIKANSI: alpha = {alpha}")

print(f"\n3. UJI STATISTIK:")
print(f"   t-test")

df = n_winter + n_summer - 2
t_critical = stats.t.ppf(1 - alpha, df=df)
print(f"   Critical value: {t_critical:.4f}")
print(f"   Daerah kritis: t > {t_critical:.4f}")

print(f"\n4. NILAI UJI STATISTIK:")
pooled_std = np.sqrt(((n_winter - 1) * std_winter**2 + (n_summer - 1) * std_summer**2) / df)
t_stat = (mean_winter - mean_summer) / (pooled_std * np.sqrt(1/n_winter + 1/n_summer))
print(f"   t-statistic = {t_stat:.4f}")

p_value = 1 - stats.t.cdf(t_stat, df=df)
print(f"   p-value = {p_value:.4f}")

print(f"\n5. KEPUTUSAN:")
if p_value < alpha:
    print(f"   TOLAK H0")
    print(f"   Kesimpulan: Rata-rata biaya Winter lebih tinggi dari Summer")
else:
    print(f"   GAGAL TOLAK H0")
    print(f"   Kesimpulan: Tidak cukup bukti bahwa rata-rata biaya Winter > Summer")

print("\nQUESTION 6.4 - SPECIFIC\n")

winter_gas = data[data['season'] == 'Winter']['gas_m3'].dropna()
summer_gas = data[data['season'] == 'Summer']['gas_m3'].dropna()

n_winter = len(winter_gas)
n_summer = len(summer_gas)
var_winter = winter_gas.var()
var_summer = summer_gas.var()
std_winter = winter_gas.std()
std_summer = summer_gas.std()

print("Data:")
print(f"   Winter: n = {n_winter}, variance = {var_winter:.2f}, std = {std_winter:.2f}")
print(f"   Summer: n = {n_summer}, variance = {var_summer:.2f}, std = {std_summer:.2f}")

print("\n1. HIPOTESIS:")
print("   H0: var_winter = var_summer")
print("   H1: var_winter > var_summer")

alpha = 0.05
print(f"\n2. TINGKAT SIGNIFIKANSI: alpha = {alpha}")

print(f"\n3. UJI STATISTIK:")
print(f"   F-test")

df1 = n_winter - 1
df2 = n_summer - 1
f_critical = stats.f.ppf(1 - alpha, df1, df2)
print(f"   Critical value: {f_critical:.4f}")
print(f"   Daerah kritis: F > {f_critical:.4f}")

print(f"\n4. NILAI UJI STATISTIK:")
f_stat = var_winter / var_summer
print(f"   F-statistic = {f_stat:.4f}")
print(f"   df1 = {df1}, df2 = {df2}")

p_value = 1 - stats.f.cdf(f_stat, df1, df2)
print(f"   p-value = {p_value:.4f}")

print(f"\n5. KEPUTUSAN:")
if p_value < alpha:
    print(f"   TOLAK H0")
    print(f"   Kesimpulan: Variansi gas Winter lebih tinggi dari Summer")
else:
    print(f"   GAGAL TOLAK H0")
    print(f"   Kesimpulan: Tidak cukup bukti bahwa variansi gas Winter > Summer")
