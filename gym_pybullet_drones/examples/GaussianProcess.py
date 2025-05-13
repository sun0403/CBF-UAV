import torch
import torch.nn as nn
import numpy as np


class GaussianProcessCBF(nn.Module):
    def __init__(self, input_dim, noise_variance=1e-3, max_points=100):
        super(GaussianProcessCBF, self).__init__()
        self.noise_variance = noise_variance
        self.kernel_variance = 0.5
        self.kernel_lengthscale = 0.5
        self.X_train = torch.empty((0, input_dim), dtype=torch.float32)
        self.Y_train = torch.empty((0, 1), dtype=torch.float32)
        self.max_points = max_points
        self.device = torch.device("cpu") 
        self.X_all = []
        print(f"[INFO] GP model using device: {self.device}")
        

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
        self.X_all.append(x.cpu().numpy().squeeze())
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

    # def update(self, cur_pos, cur_vel, u_opt, obstacle_positions, GRAVITY,
    #            safe_distance, lambda1, lambda2,
    #            epsilon=0.1, num_samples=50):

    #     added = 0
    #     for _ in range(num_samples * 5):
    #         pos_sample = cur_pos + np.random.normal(0, 0.2, size=3)
    #         vel_sample = cur_vel + np.random.normal(0, 0.2, size=3)
    #         x_sample = np.concatenate([pos_sample, vel_sample])

    #         h = np.min([np.linalg.norm(pos_sample - obs) - safe_distance for obs in obstacle_positions])
    #         sigma2 = self.predict_variance(x_sample)
    #         hs = h - sigma2
    #         adaptive_epsilon = epsilon + 0.6 * sigma2
    #         if abs(hs) < adaptive_epsilon and self.is_far_enough(x_sample):
    #             self.add_sample(x_sample, hs)
    #             added += 1
    #             print(f"[Sample Added] h={h:.4f}, sigma²={sigma2:.4f}, hs={hs:.4f}")
    #             if added >= num_samples:
    #                 break

    #     if added < num_samples:
    #         print(f"[Warning] 只添加了 {added}/{num_samples} 个样本")
    #     else:
    #         print(f"[Success] 成功添加 {added} 个新样本")

    #     print(f"[GP] 当前训练样本总数：{self.X_train.size(0)}")

    
    # def update(self, cur_pos, cur_vel, u_opt, obstacle_positions, GRAVITY,
    #        safe_distance, lambda1, lambda2,
    #        epsilon=0.05, num_samples=50):

    #     added = 0
    #     candidate_points = []
    #     candidate_h = []
    #     candidate_sigma2 = []

    #     num_candidates = num_samples * 10  # 候选池数量

    #     for _ in range(num_candidates):
    #         # 在当前状态周围生成扰动
    #         pos_sample = cur_pos + np.random.normal(0, 0.1, size=3)
    #         vel_sample = cur_vel + np.random.normal(0, 0.1, size=3)
    #         x_sample = np.concatenate([pos_sample, vel_sample])

    #         # 计算最小安全距离（h）
    #         h = np.min([np.linalg.norm(pos_sample - obs) - safe_distance for obs in obstacle_positions])
    #         sigma2 = self.predict_variance(x_sample)

    #         candidate_points.append(x_sample)
    #         candidate_h.append(h)
    #         candidate_sigma2.append(sigma2)

    #     candidate_points = np.array(candidate_points)
    #     candidate_h = np.array(candidate_h)
    #     candidate_sigma2 = np.array(candidate_sigma2)

    #     # 优先选择最大 sigma² 的点（即不确定性高的）
    #     best_indices = np.argsort(-candidate_sigma2)

    #     for idx in best_indices:
    #         x_sample = candidate_points[idx]
    #         h = candidate_h[idx]
    #         sigma2 = candidate_sigma2[idx]
    #         hs = h - 5.0*sigma2
    #         adaptive_epsilon = epsilon + 0.8 * sigma2

    #         if abs(hs) < adaptive_epsilon and self.is_far_enough(x_sample):
    #             self.add_sample(x_sample, hs)
    #             added += 1
    #             print(f"[Sample Added] h={h:.4f}, sigma²={sigma2:.4f}, hs={hs:.4f}")
    #             if added >= num_samples:
    #                 break

    #     if added < num_samples:
    #         print(f"[Warning] 只添加了 {added}/{num_samples} 个样本")
    #     else:
    #         print(f"[Success] 成功添加 {added} 个新样本")

    #     print(f"[GP] 当前训练样本总数：{self.X_train.size(0)}")


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
    def update(self, cur_pos, cur_vel, u_opt, obstacle_positions, GRAVITY,
           safe_distance, lambda1, lambda2,
           epsilon=0.1, num_samples=40):

        added = 0
        candidate_points = []
        candidate_sigma2 = []

        num_candidates = num_samples * 10  # 候选池数量

        for _ in range(num_candidates):
            # 沿当前路径进行扰动采样
            pos_sample = cur_pos + np.random.normal(0, 0.15, size=3)
            vel_sample = cur_vel + np.random.normal(0, 0.15, size=3)
            x_sample = np.concatenate([pos_sample, vel_sample])

            sigma2 = self.predict_variance(x_sample)

            if self.is_far_enough(x_sample) and sigma2 > 0.05:
                candidate_points.append(x_sample)
                candidate_sigma2.append(sigma2)

        if len(candidate_points) == 0:
            print("[GP] No high uncertainty samples found this round.")
            return

        # 按 σ² 降序排序
        sorted_indices = np.argsort(-np.array(candidate_sigma2))

        for idx in sorted_indices:
            x_sample = candidate_points[idx]
            sigma2 = candidate_sigma2[idx]
            self.add_sample(x_sample, sigma2)
            added += 1
            print(f"[GP Sample Added] σ²={sigma2:.4f}")
            if added >= num_samples:
                break

        print(f"[GP] Added {added}/{num_samples} samples, total: {self.X_train.size(0)}")


    def is_far_enough(self, x_candidate, tau=0.01):
        if self.X_train.size(0) == 0:
            return True
        dists = torch.norm(self.X_train - torch.tensor(x_candidate, dtype=torch.float32), dim=1)
        return torch.min(dists) >= tau
