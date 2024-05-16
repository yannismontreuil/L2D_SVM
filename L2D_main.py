import numpy as np
import cvxpy as cp
import random

class L2D_SVM:
    '''
    L2D_SVM is a class that implements the L2D-SVM algorithm with predictor and rejector approach.
    We aim to defer a decision to the expert when the classifier is uncertain, and at the same time, the expert is certain.
    A rejector $r$ is trained to defer the decision to the expert when the classifier $f$ is not sufficiently confident comparing with the expert $m$.
    The following implementation is the dual formulation of the L2D-SVM algorithm.
    '''
    def __init__(self, *, C1=1.0, L0=1.0, L1=1.0, kernel_f = 'RBF', kernel_r = 'RBF', var=0.1, solver_name='ECOS', solver_opts=None):
        '''
        :param C1: Errors Scaling hyperparameter
        :param L0: Penlization factor for the classifier
        :param L1: Penlization factor for the rejector
        :param kernel_f: Kernel matrix for the classifier
        :param kernel_r: Kernel matrix for the rejector
        :param var: Variance of the kernel
        :param solver_name: Solver type {ECOS, MOSEK, etc.}
        :param solver_opts: Solver Options used in CXVPY
        '''
        super().__init__()
        self.C1 = C1
        self.L0 = L0
        self.L1 = L1
        self.kernel_f = kernel_f
        self.kernel_r = kernel_r
        self.var = var
        self.solver_name = solver_name
        self.solver_opts = solver_opts

    def kernel(self, x, x_pred, var, kernel):
        '''
        :param x: Training Inputs
        :param x_pred: Prediction Inputs
        :param var: Variance of the kernel
        :param kernel: Type of kernel {RBF, linear}
        :return: Kernel Matrix {dim(x), dim(x_pred)}
        '''
        if kernel == 'RBF':
            X_norm = np.sum(x.T ** 2, axis=-1)
            Y_norm = np.sum(x_pred.T ** 2, axis=-1)
            K = np.exp(-var * (X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(x.T, x_pred)))
        if kernel == 'linear':
            K = x.T @ x_pred
        return K
    def M_function(self, alpha, y, tau):
        '''
        :param alpha: Calibration coefficient
        :param y: True Labels
        :param tau: Lagrangian Parameter
        :return: M function for the classifier
        '''
        return cp.multiply(alpha*tau,y)
    def P_function(self, mu, beta, alpha, m, tau):
        '''
        :param mu: Lagrangian Parameter
        :param beta: Calibration coefficient
        :param alpha: Calibration coefficient
        :param m: Expert predictions
        :param tau: Lagrangian Parameter
        :return: P function for the rejector
        '''
        t1 = cp.multiply(beta*m, mu)
        return alpha*tau - t1
    def rejector(self, L1, Kr, br, P):
        '''
        Rejector function
        '''
        return 1/L1 * P.value @ Kr + br

    def classifier(self, L0, Kf, bf, M):
        '''
        Classifier function
        '''
        return 1/L0 * M.value @ Kf + bf
    def fit(self, X, y, m):
        '''
        Training step for the L2D-SVM
        :param X: Training Inputs
        :param y: True Labels
        :param m: Expert predictions
        :return: None
        '''
        N = len(y)
        self.X_tr = X
        self.y_tr = y
        self.m_tr = m
        self.tau = cp.Variable(N)
        self.mu = cp.Variable(N)
        self.m = m
        self.alpha = 1
        self.expert_errors = np.where(m != y, 1,0)
        # self.beta = self.alpha/(1-2*self.expert_errors)
        self.beta = 1
        self.P = self.P_function(self.mu, self.beta, self.alpha, self.m, self.tau)
        self.M = self.M_function(self.alpha, y, self.tau)
        self.K_f = self.kernel(X, X, self.var, self.kernel_f)
        self.K_r = self.kernel(X, X, self.var, self.kernel_r)

        K_F = cp.atoms.affine.wraps.psd_wrap(self.K_f )
        K_R = cp.atoms.affine.wraps.psd_wrap(self.K_r)
        constraint = [cp.sum(self.M) == 0, cp.sum(self.P) == 0, self.tau + self.mu <= self.C1, self.tau >= 0, self.mu >= 0]
        objective = cp.Minimize(
            + 1 / self.L0 * cp.quad_form(self.M, K_F) + 1 / self.L1 * cp.quad_form(self.P, K_R) -
            cp.sum(self.tau) - cp.sum(cp.multiply(self.mu, self.expert_errors)))
        prob = cp.Problem(objective, constraint)
        prob.solve(solver=self.solver_name, **(self.solver_opts or {}))
        print('Optimisation finished')
        self.bias_r = np.mean(1/self.beta - self.rejector(self.L1, self.K_r, 0, self.P))
        self.rejector_r = self.rejector(self.L1, self.K_r, self.bias_r, self.P)
        self.bias_f = np.mean(y*(1/self.alpha - self.rejector_r) - self.classifier(self.L0, self.K_f, 0, self.M))
        self.classifier_f = self.classifier(self.L0, self.K_f, self.bias_f, self.M)


    def predict(self, X_pred, m_pred):
        '''
        Prediction step for the L2D-SVM
        :param X_pred: Inputs val/test
        :param m_pred: Expert predictions val/test
        :return: prediction from the system {classifier, expert}
        '''
        self.Kr_pred = self.kernel(self.X_tr, X_pred, self.var, self.kernel_r)
        self.Kf_pred = self.kernel(self.X_tr, X_pred, self.var, self.kernel_f)
        self.rejector_pred = self.rejector(self.L1, self.Kr_pred, self.bias_r, self.P)
        self.classifier_pred = self.classifier(self.L0, self.Kf_pred, self.bias_f, self.M)
        self.prediction = np.where(self.rejector_pred > 0, self.classifier_pred, m_pred)
        self.defer_indices = np.where(self.rejector_pred<=0)
        return self.prediction

    def metrics(self, y_pred, y_true, m_pred):
        '''
        :param y_pred: Predictions from the system
        :param y_true: True labels
        :param m_pred: Expert predictions
        :return: Metrics {System Accuracy, Expert Accuracy, Classifier Accuracy, Defer Ratio}
        '''
        self.system_accuracy = np.mean(np.where(y_pred*y_true>0, 1, 0))
        self.expert_accuracy = np.mean(np.where(m_pred*y_true>0, 1, 0))
        self.classifier_accuracy = np.mean(np.where(self.classifier_pred*y_true>0, 1, 0))
        self.defer_ratio = len(self.defer_indices)/len(y_pred)
        dict = {'system_accuracy': self.system_accuracy, 'expert_accuracy': self.expert_accuracy,
                'classifier_accuracy': self.classifier_accuracy, 'defer_ratio': self.defer_ratio}
        return dict
    def get_params(self):
        return self.bias_f, self.bias_r, self.classifier_f, self.rejector_r



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
