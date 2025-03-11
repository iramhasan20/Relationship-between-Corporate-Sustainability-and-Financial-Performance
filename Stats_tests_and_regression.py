import pandas as pd

# Load dataset
file_name = 'data.csv'
df = pd.read_csv(file_name)

# Display basic descriptive statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Diagnostic checks

# Select relevant variables
X = df[['SP_INDEX', 'FP1', 'FP2', 'FP3', 'FP4', 'SIZE', 'VOLATILITY', 'YEARS']]

# Add constant for regression
X = sm.add_constant(X)

# Compute VIF for each variable
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF values
print(vif_data)

# Sub-categories classification

# Categorizing firms by age
df['Years_Category'] = pd.qcut(df['YEARS'], 3, labels=['Young', 'Mid', 'Old'])

# Categorizing firms by market size
df['Size_Category'] = pd.qcut(df['SIZE'], 3, labels=['Small', 'Medium', 'Large'])

# Count firms in each category
print(df['Years_Category'].value_counts())
print(df['Size_Category'].value_counts())

# Compute quartiles for SIZE
Q1 = df['SIZE'].quantile(0.25)
Q2 = df['SIZE'].quantile(0.50)  # Median
Q3 = df['SIZE'].quantile(0.75)

print(f"Q1: {Q1}, Q2: {Q2}, Q3: {Q3}")

# t-Test for Mean Differences
sub1 = df[df['YEARS'] < df['YEARS'].median()]['FP1']
sub2 = df[df['YEARS'] >= df['YEARS'].median()]['FP1']
t_stat, p_value = stats.ttest_ind(sub1, sub2, equal_var=False)
print(f't-test: t-stat={t_stat}, p-value={p_value}')

# F-Test for Variance Differences
f_stat = np.var(sub1, ddof=1) / np.var(sub2, ddof=1)
print(f'F-test: F-stat={f_stat}')

# Define Dependent and Independent Variables
y = df['FP1']
X = df[['SP_INDEX', 'SIZE', 'VOLATILITY', 'YEARS']]
X = sm.add_constant(X)

# OLS Regression
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

# Fixed Effects Model
fixed_model = PanelOLS(y, X, entity_effects=True).fit()
print(fixed_model.summary)

# Random Effects Model
random_model = RandomEffects(y, X).fit()
print(random_model.summary)

# F-Test for Fixed Effects
f_test = fixed_model.f_statistic.pval
print(f'Fixed Effects Model F-test p-value: {f_test}')

# Hausman Test for Model Selection
diff = fixed_model.params - random_model.params
cov_diff = fixed_model.cov - random_model.cov
hausman_stat = diff.T @ np.linalg.inv(cov_diff) @ diff
p_value_hausman = stats.chi2.sf(hausman_stat, len(diff))
print(f'Hausman Test Statistic: {hausman_stat}, p-value: {p_value_hausman}')

# Define dependent and independent variables
y = df['FP']  # Replace with chosen financial performance variable
X = df[['SP_INDEX', 'Control1', 'Control2', 'Control3']]  # Replace with sustainability measure and control variables
X = sm.add_constant(X)

# OLS Model with PCSE
ols_model = sm.OLS(y, X)
ols_results = ols_model.fit(cov_type='cluster')  # Clustered standard errors
print("OLS Model with PCSE:")
print(ols_results.summary())

# Fixed Effects Model with PCSE
fe_model = PanelOLS(y, X, entity_effects=True)
fe_results = fe_model.fit(cov_type='clustered', cluster_entity=True)
print("Fixed Effects Model with PCSE:")
print(fe_results.summary)

# Random Effects Model with PCSE
re_model = RandomEffects(y, X)
re_results = re_model.fit(cov_type='clustered', cluster_entity=True)
print("Random Effects Model with PCSE:")
print(re_results.summary)
