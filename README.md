# L2D_SVM
The L2D-SVM (Learning to Defer SVM) is an implementation of a classifier and rejector approach designed to defer decisions to an expert when the classifier's confidence is comparatively low. This implementation uses a dual formulation to optimize both a classifier and a rejector function, determining when to defer based on the certainty of the classifier versus the expert.

# Requirements
You can install the necessary libraries using pip:
```bash
pip install numpy cvxpy
```

# Usage
To use the L2D-SVM model, you need to import it into your Python script, initialize it with the desired configuration, train it on your data, and then use it to make predictions. Here’s a basic example in the main:
```bash
if __name__ == '__main__':
    random.seed(11)
    X_train = np.random.randn(2, 100)
    y_train = 2*np.random.randint(0, 2, 100)-1
    m_train = 2*np.random.randint(0, 2, 100)-1
    SVM_expert = L2D_SVM(C1=1.0, L0=1.0, L1=1.0, kernel_f='RBF', kernel_r='RBF', var=0.1, solver_name='ECOS', solver_opts=None)
    SVM_expert.fit(X_train, y_train, m_train)
    X_test = np.random.randn(2, 20)
    y_test = 2*np.random.randint(0, 2, 20)-1
    m_test = 2*np.random.randint(0, 2, 20)-1
    y_pred = SVM_expert.predict(X_test, m_test)
    metrics = SVM_expert.metrics(y_pred, y_test, m_test)
```
# Methods
fit(X, y, m): Trains the model using training data and expert predictions.

predict(X_pred, m_pred): Predicts the outcomes using the system {classifier, expert}.

metrics(y_pred, y_true, m_pred): Calculates various performance metrics {defer ratio, accuracies}.
