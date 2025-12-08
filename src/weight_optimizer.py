from .solver_lp import run_lp
from .solver_cd import run_coordinate_descent
from .solver_gd import run_gradient_descent


class WeightOptimizer:

    def __init__(self):
        pass

    def linear_programming(self, G, Y, root, *args, **kwargs):
        Y = normalize_dict(Y)
        theta, Y_hat = run_lp(G, Y, root, *args, **kwargs)
        self.theta = theta
        self.Y_hat = Y_hat
        return theta

    def coordinate_descent(self, G, Y, root, *args, **kwargs):
        Y = normalize_dict(Y)
        theta_dict, Yhat_dict, loss_history, theta_history = run_coordinate_descent(
            G, Y, root, *args, **kwargs
        )
        self.cd_theta_dict = theta_dict
        self.cd_Yhat_dict = Yhat_dict
        self.cd_loss_history = loss_history
        self.cd_theta_history = theta_history
        return theta_history[-1]

    def gradient_descent(self, G, Y, root, seed=None, *args, **kwargs):
        Y = normalize_dict(Y)
        theta_dict, Yhat_dict, loss_history, theta_history = run_gradient_descent(
            G, Y, root, seed=seed, *args, **kwargs
        )
        self.gd_theta_dict = theta_dict
        self.gd_Yhat_dict = Yhat_dict
        self.gd_loss_history = loss_history
        self.gd_theta_history = theta_history
        return theta_history[-1]


def normalize_dict(P_M: dict):
    s = sum(P_M.values())
    return {k: v / s for k, v in P_M.items()}
