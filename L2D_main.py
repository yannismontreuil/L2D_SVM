import numpy as np
import random

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
    print('System Accuracy: ', metrics['system_accuracy'])
    print('Expert Accuracy: ', metrics['expert_accuracy'])
    print('Classifier Accuracy: ', metrics['classifier_accuracy'])
    print('Defer Ratio: ', metrics['defer_ratio'])
