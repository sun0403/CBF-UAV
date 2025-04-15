import torch
import torch.nn as nn
import numpy as np

class GaussianProcessCBF(nn.Module):
    def __init__(self, input_dim, noise_variance=1e-3, max_points=150):
        super(GaussianProcessCBF, self).__init__()
        self.noise_variance = noise_variance
        self.kernel_variance = nn.Parameter(torch.tensor(1.0))
        self.kernel_lengthscale = nn.Parameter(torch.tensor(1.0))
        self.X_train = torch.empty((0, input_dim), dtype=torch.float32)
        self.Y_train = torch.empty((0, 1), dtype=torch.float32)
        self.max_points = max_points

        
    def rbf_kernel(self, X1, X2):
        dists_sq = torch.sum((X1.unsqueeze(1) - X2.unsqueeze(0))**2, dim=-1)  # (N, M) 矩阵，N 是样本数，M 是维度
        return self.kernel_variance * torch.exp(-0.5 * dists_sq / self.kernel_lengthscale ** 2)
    def predict(self, x_test):
        if self.X_train.size(0) == 0:
            mean = torch.zeros((x_test.size(0), 1), dtype=torch.float32)
            var = torch.ones((x_test.size(0), 1), dtype=torch.float32)
            return mean, var

        K = self.rbf_kernel(self.X_train, self.X_train) + self.noise_variance * torch.eye(self.X_train.size(0))
        K_s = self.rbf_kernel(self.X_train, x_test)  # 计算 K_s
        K_ss = self.rbf_kernel(x_test, x_test) + self.noise_variance * torch.eye(x_test.size(0))

        # 计算 alpha，这里 alpha 的形状应该是 (N, 1)
        alpha = torch.linalg.solve(K, self.Y_train)

        # 调整矩阵乘法，确保 K_s 和 alpha 的维度匹配
        mu = torch.matmul(K_s.T, alpha)  # 计算 mu

        # 计算协方差
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
        if self.X_train.size(0) > self.max_points:
            self.X_train = self.X_train[-self.max_points:]
            self.Y_train = self.Y_train[-self.max_points:]

    def predict_variance(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        _, cov = self.predict(x)
        return cov.item()

    def compute_variance_gradient(self, x_input):

        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float32)
        
        x_input = x_input.flatten().unsqueeze(0).detach().requires_grad_(True)

        
        K = self.rbf_kernel(self.X_train, self.X_train) + self.noise_variance * torch.eye(self.X_train.size(0))
        K_inv = torch.linalg.inv(K)

        
        k_xq = self.rbf_kernel(self.X_train, x_input)
        x_input_expanded = x_input.expand_as(self.X_train)  
        grad_k_xq = (self.X_train - x_input_expanded) * k_xq / self.kernel_lengthscale ** 2  
        

        
        grad_sigma2 = -2 * torch.matmul(K_inv, grad_k_xq)  

        return grad_sigma2.detach().numpy()



    def update(self, cur_pos, cur_vel, u_opt, obstacle_positions, GRAVITY,
           safe_distance=0.2, lambda1=10.0, lambda2=10.0,
           epsilon=0.01, num_samples=50):

        added = 0
        for obs in obstacle_positions:
            direction = cur_pos - obs
            distance = np.linalg.norm(direction) + 1e-6
            direction /= distance
            
            pos_sample = cur_pos + direction * np.random.uniform(0.0, 0.1)
            vel_sample = cur_vel+ direction*np.random.uniform(0.0, 0.1, size=3)
            x_sample = np.concatenate([pos_sample, vel_sample])

            h = distance - safe_distance
            h_dot = np.dot(pos_sample - obs, vel_sample)
            acc = (u_opt / GRAVITY) - np.array([0, 0, 9.8])
            h_ddot = np.dot(pos_sample - obs, acc)

            sigma2 = self.predict_variance(x_sample)
            hs = h - sigma2
        
            if abs(hs) < epsilon:
                added += 1
                self.add_sample(x_sample,hs)
                

            

        if added < num_samples:

            print(f"[⚠️] Only added {added}/{num_samples} exploratory samples.")
