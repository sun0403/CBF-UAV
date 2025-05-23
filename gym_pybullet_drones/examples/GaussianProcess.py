import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class GaussianProcessCBF(nn.Module):
    def __init__(self, input_dim, noise_variance=1e-2, max_points=50, debug=False):
        super(GaussianProcessCBF, self).__init__()
        self.noise_variance = noise_variance
        self.kernel_variance = 1.4
        self.kernel_lengthscale = 0.5
        self.X_train = torch.empty((0, input_dim), dtype=torch.float32)
        self.Y_train = torch.empty((0, 1), dtype=torch.float32)
        self.max_points = max_points
        self.device = torch.device("cpu")
        self.X_all = []
        self.debug = debug
        print(f"[INFO] GP model using device: {self.device}")

    def rbf_kernel(self, X1, X2):
        dists_sq = torch.sum((X1.unsqueeze(1) - X2.unsqueeze(0))**2, dim=-1)
        return self.kernel_variance * torch.exp(-0.5 * dists_sq / self.kernel_lengthscale ** 2)

    def predict(self, x_test):
        if self.X_train.size(0) == 0:
            mean = torch.zeros((x_test.size(0), 1), dtype=torch.float32)
            var = torch.ones((x_test.size(0), 1), dtype=torch.float32)
            return mean, var

        K = self.rbf_kernel(self.X_train, self.X_train) + \
            self.noise_variance * torch.eye(self.X_train.size(0))
        K_s = self.rbf_kernel(self.X_train, x_test)
        K_ss = self.rbf_kernel(x_test, x_test) + self.noise_variance * torch.eye(x_test.size(0))

        alpha = torch.linalg.solve(K, self.Y_train)
        mu = torch.matmul(K_s.T, alpha)
        v = torch.linalg.solve(K, K_s)
        cov = K_ss - torch.matmul(K_s.T, v)

        return mu, cov

    def forward(self, x):
        return self.predict(x)

    def add_sample(self, x, y):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([[y]], dtype=torch.float32)
        self.X_train = torch.cat([self.X_train, x], dim=0)
        self.Y_train = torch.cat([self.Y_train, y], dim=0)
        self.X_all.append(x.cpu().numpy().squeeze())
        if self.X_train.size(0) > self.max_points:
            self.X_train = self.X_train[-self.max_points:]
            self.Y_train = self.Y_train[-self.max_points:]

    def predict_variance(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        _, cov = self.predict(x)
        return cov.item()

    def compute_variance_gradient_and_hessian(self, x_input):
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float32)
        x_input = x_input.clone().detach().requires_grad_(True).unsqueeze(0)
        _, cov = self.predict(x_input)
        sigma2 = cov.squeeze()
        grad = torch.autograd.grad(sigma2, x_input, create_graph=True)[0]

        hessian_rows = []
        for i in range(x_input.shape[1]):
            grad_i = grad[0, i]
            grad2 = torch.autograd.grad(grad_i, x_input, retain_graph=True)[0]
            hessian_rows.append(grad2)
        hessian = torch.cat(hessian_rows, dim=0)

        return grad.detach().numpy(), hessian.detach().numpy()

    def is_far_enough(self, x_candidate, tau=0.01):
        if self.X_train.size(0) == 0:
            return True
        dists = torch.norm(self.X_train - torch.tensor(x_candidate, dtype=torch.float32), dim=1)
        return torch.min(dists) >= tau

    def update(self, cur_pos, cur_vel, u_opt, target_pos, obstacle_positions, GRAVITY,
               safe_distance, lambda1, lambda2,
               epsilon=0.5, num_samples=10,
               min_step=0.1, max_step=0.5):

        added = 0
        trials = 0
        max_trials = num_samples * 20

        direction = target_pos - cur_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            print("[GP] Warning: direction vector too small")
            return
        direction /= direction_norm

        while added < num_samples and trials < max_trials:
            trials += 1
            step = np.random.uniform(min_step, max_step)
            pos_sample = cur_pos + direction * step + np.random.normal(0, 0.05, size=3)
            vel_sample = cur_vel + np.random.normal(0, 0.05, size=3)
            x_sample = np.concatenate([pos_sample, vel_sample])
            h = np.min([np.linalg.norm(pos_sample - obs) for obs in obstacle_positions]) - safe_distance
            sigma2 = self.predict_variance(x_sample)
            hs = h - sigma2

            sigma2_threshold = 0.2
            if sigma2 > sigma2_threshold and self.is_far_enough(x_sample):
                self.add_sample(x_sample, sigma2)
                added += 1
                print(f"[GP Sample Added] #{added}: h={h:.3f}, σ²={sigma2:.3f}, hs={hs:.3f}")

        print(f"[GP] Sample update: added {added}/{num_samples}, total: {self.X_train.size(0)}")

        
    
