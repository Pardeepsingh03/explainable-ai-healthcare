import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dowhy import CausalModel
from econml.dml import LinearDML
from causalml.inference.meta import BaseTRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Step 1: Generate synthetic data
np.random.seed(42)
n = 1000
age = np.random.normal(40, 10, n)
income = np.random.normal(50000, 15000, n)
treatment = (age + np.random.normal(0, 1, n) > 40).astype(int)
outcome = 5 * treatment + 0.1 * age + 0.0001 * income + np.random.normal(0, 1, n)

data = pd.DataFrame({'age': age, 'income': income, 'treatment': treatment, 'outcome': outcome})

# Step 2: DoWhy
print("\n--- DoWhy ---")
model = CausalModel(
    data=data,
    treatment='treatment',
    outcome='outcome',
    common_causes=['age', 'income']
)
identified_estimand = model.identify_effect()
dowhy_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
print("DoWhy ATE:", dowhy_estimate.value)

# Step 3: EconML
print("\n--- EconML ---")
econml_model = LinearDML(model_y='linear', model_t='linear', discrete_treatment=True, random_state=42)
econml_model.fit(Y=data['outcome'], T=data['treatment'], X=data[['age', 'income']])
econml_te_pred = econml_model.effect(X=data[['age', 'income']])
print("EconML ATE:", np.mean(econml_te_pred))

# Plot EconML effect by age
plt.scatter(data['age'], econml_te_pred, alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Estimated Treatment Effect")
plt.title("EconML: Treatment Effect by Age")
plt.show()

# Step 4: CausalML
print("\n--- CausalML ---")
X = data[['age', 'income']]
y = data['outcome']
t = data['treatment']

X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(X, t, y, test_size=0.3, random_state=42)

causalml_model = BaseTRegressor(learner=RandomForestRegressor(n_estimators=100, random_state=42))
causalml_model.fit(X=X_train, treatment=t_train, y=y_train)
causalml_te_pred = causalml_model.predict(X=X_test)
print("CausalML ATE:", np.mean(causalml_te_pred))
