import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DoWhy imports
import dowhy
from dowhy import CausalModel

# EconML imports
from econml.dml import LinearDML

# --- Step 1: Generate synthetic data ---

np.random.seed(42)
n = 1000

# Confounders
age = np.random.normal(40, 10, n)
income = np.random.normal(50000, 15000, n)

# Treatment assignment influenced by confounders
treatment = (age + np.random.normal(0, 1, n) > 40).astype(int)

# Outcome influenced by treatment and confounders + noise
outcome = 5 * treatment + 0.1 * age + 0.0001 * income + np.random.normal(0, 1, n)

# Put into DataFrame
data = pd.DataFrame({'age': age, 'income': income, 'treatment': treatment, 'outcome': outcome})

print("First 5 rows of the data:\n", data.head())

# --- Step 2: Causal inference with DoWhy ---

print("\n--- DoWhy Causal Inference ---")

model = CausalModel(
    data=data,
    treatment='treatment',
    outcome='outcome',
    common_causes=['age', 'income']
)

# Visualize the causal graph (optional)
# This requires graphviz installed; uncomment if you want to see the graph
# model.view_model()

identified_estimand = model.identify_effect()
print("Identified estimand:", identified_estimand)

estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression")
print("DoWhy estimate of average treatment effect (ATE):", estimate.value)

# Refute estimate with placebo treatment test (optional)
refute = model.refute_estimate(identified_estimand, estimate,
                               method_name="placebo_treatment_refuter", placebo_type="permute")
print(refute)

# --- Step 3: Causal inference with EconML ---

print("\n--- EconML Causal Inference ---")

est = LinearDML(model_y='linear', model_t='linear', discrete_treatment=True, random_state=42)
est.fit(Y=data['outcome'], T=data['treatment'], X=data[['age', 'income']])

# Estimate treatment effect for each sample
te_pred = est.effect(X=data[['age', 'income']])

print("EconML average treatment effect (ATE):", np.mean(te_pred))

# Optional: plot treatment effects by age
plt.scatter(data['age'], te_pred, alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Estimated Treatment Effect')
plt.title('EconML Estimated Treatment Effect by Age')
plt.show()
