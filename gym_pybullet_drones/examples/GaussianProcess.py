import torch
import torch.nn as nn
import numpy as np

class GaussianProcessCBF(nn.Module):
    def __init__(self, input_dim, noise_variance=1e-3, max_points=80):
        super(GaussianProcessCBF, self).__init__()
        self.noise_variance = noise_variance
        self.kernel_variance = 0.3
        self.kernel_lengthscale = 0.5
        self.X_train = torch.empty((0, input_dim), dtype=torch.float32)
        self.Y_train = torch.empty((0, 1), dtype=torch.float32)
        self.max_points = max_points
        self.x_b = None
        self.device = torch.device("cuda")
        
    def rbf_kernel(self, X1, X2):
        dists_sq = torch.sum((X1.unsqueeze(1) - X2.unsqueeze(0))**2, dim=-1)  
        return self.kernel_variance * torch.exp(-0.5 * dists_sq / self.kernel_lengthscale ** 2)
    def predict(self, x_test):
        if self.X_train.size(0) == 0:
            mean = torch.zeros((x_test.size(0), 1), dtype=torch.float32)
            var = torch.ones((x_test.size(0), 1), dtype=torch.float32)
            return mean, var

        K = self.rbf_kernel(self.X_train, self.X_train) + self.noise_variance * torch.eye(self.X_train.size(0))
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
           safe_distance, lambda1, lambda2,
           epsilon=0.2, num_samples=50):

        added = 0
        pos_center = cur_pos
        vel_center = cur_vel

        for _ in range(num_samples * 5):  # 允许更多尝试
            
            pos_sample = pos_center + np.random.normal(0, 0.05, size=3)
            vel_sample = vel_center + np.random.normal(0, 0.05, size=3)
            x_sample = np.concatenate([pos_sample, vel_sample])

            
            h = np.min([np.linalg.norm(pos_sample - obs) - safe_distance for obs in obstacle_positions])
            sigma2 = self.predict_variance(x_sample)
            hs = h - sigma2

            #
            if abs(hs) < epsilon and self.is_far_enough(x_sample):
                self.add_sample(x_sample, hs)
                added += 1
                print(f"h={h:.4f}, sigma²={sigma2:.4f}, hs={hs:.4f}")
                if added >= num_samples:
                    break

        if added < num_samples:
            print(f"只添加了 {added}/{num_samples} 个样本")
        else:
            print(f"成功添加 {added} 个新样本")

        print(f"当前 GP 训练样本总数：{self.X_train.size(0)}")

    
    

    def save(self, path):
        torch.save({
            "X_train": self.X_train,
            "Y_train": self.Y_train,
            "kernel_variance": self.kernel_variance,
            "kernel_lengthscale": self.kernel_lengthscale
            }, path)
        print(f" GP model saved to {path}")

    @staticmethod
    def load(path, input_dim, noise_variance=1e-3, max_points=150):
        """加载保存的 GP 模型"""
        model = GaussianProcessCBF(input_dim=input_dim, noise_variance=noise_variance, max_points=max_points)
        checkpoint = torch.load(path, map_location='cpu')  # 使用 CPU 加载
        model.X_train = checkpoint["X_train"]
        model.Y_train = checkpoint["Y_train"]
        model.kernel_variance = checkpoint["kernel_variance"]
        model.kernel_lengthscale = checkpoint["kernel_lengthscale"]
        print(f" GP model loaded from {path}")
        return model
    
    def compute_variance_gradient_and_hessian(self, x_input):
        
        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=torch.float32)
        x_input = x_input.clone().detach().requires_grad_(True).unsqueeze(0)  # (1, D)

        # 获取预测协方差（sigma² 是 scalar）
        _, cov = self.predict(x_input)  # cov.shape: (1, 1)
        sigma2 = cov.squeeze()  # scalar

        # 一阶导数：
        grad = torch.autograd.grad(sigma2, x_input, create_graph=True)[0]  # shape: (1, D)

        # 二阶导数（Hessian）
        hessian_rows = []
        for i in range(x_input.shape[1]):
            grad_i = grad[0, i]
            grad2 = torch.autograd.grad(grad_i, x_input, retain_graph=True)[0]  # shape: (1, D)
            hessian_rows.append(grad2)

        hessian = torch.cat(hessian_rows, dim=0)  # shape: (D, D)

        return grad.detach().numpy(), hessian.detach().numpy()
    

    def is_far_enough(self, x_candidate, tau=0.01):
        if self.X_train.size(0) == 0:
            return True
        dists = torch.norm(self.X_train - torch.tensor(x_candidate, dtype=torch.float32), dim=1)
        return torch.min(dists) >= tau

