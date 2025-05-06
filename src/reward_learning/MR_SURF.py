import csv
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

from reward_learning.MR import MR, BradleyTerryLoss, LinearLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MR_SURF(MR):
    def __init__(self, config, path, linear_loss=False, threshold=0.999):
        super(MR_SURF, self).__init__(config, path, linear_loss)
        self.threshold = threshold  # 확신 임계값 τ

    @staticmethod
    def initialize(config, path=None, allow_existing=True, linear_loss=False):
        obs_dim = config.get("obs_dim")
        act_dim = config.get("act_dim")
        hidden_size = config.get("hidden_size", 256)
        lr = config.get("lr", 0.001)

        model = MR_SURF(
            config={"obs_dim": obs_dim, "act_dim": act_dim, "hidden_size": hidden_size},
            path=path,
            linear_loss=linear_loss,
        )

        if path is not None:
            if os.path.isfile(path):
                if not allow_existing:
                    print("Skipping model initialization because already exists")
                    return None, None
                model.load_state_dict(
                    torch.load(path, weights_only=True, map_location=device)
                )
                print(f"Model loaded from {path}")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return model, optimizer

    def surf_loss(self, rewards_s0, rewards_s1, mask0, mask1, mu=None):
        """무라벨 데이터용 SURF 로스"""
        reward_s0_sum = torch.sum(rewards_s0 * (1 - mask0), dim=1)
        reward_s1_sum = torch.sum(rewards_s1 * (1 - mask1), dim=1)

        if self.linear_loss:
            pred_probs = reward_s1_sum / (reward_s1_sum + reward_s0_sum + 1e-6)
        else:
            pred_probs = torch.sigmoid(reward_s1_sum - reward_s0_sum)

        # 예측 레이블 (0 or 1)
        pseudo_mu = (pred_probs > 0.5).float()

        # 확신 임계값 이상인 경우만 선택
        confident_mask = ((pred_probs - 0.5).abs() > (self.threshold - 0.5)).float()
        confident_ratio = confident_mask.mean().item()  # 확신 샘플 비율

        if confident_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred_probs.device), 0.0

        # 서브셋에 대해 BT or Linear Loss 적용
        confident_pred_probs = pred_probs[confident_mask.bool()]
        confident_pseudo_mu = pseudo_mu[confident_mask.bool()]

        loss = F.binary_cross_entropy(confident_pred_probs, confident_pseudo_mu)

        return loss, confident_ratio

    def train_model_with_surf(self, optimizer, labeled_loader, unlabeled_loader, val_loader=None, num_epochs=100, lambda_u=1.0):
        """SURF 방식 학습"""
        loss_fn = LinearLoss() if self.linear_loss else BradleyTerryLoss()

        print("[Train started] MR-SURF path:", self.path)

        with open(self.log_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Labeled Loss", "Unlabeled Loss", "Total Loss", "Validation Loss", "Confident Ratio"])

        best_loss = float("inf")

        for epoch in tqdm(range(num_epochs), desc="Training MR-SURF"):
            self.train()
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)

            total_labeled_loss = 0.0
            total_unlabeled_loss = 0.0
            total_confident_ratio = 0.0
            num_unlabeled_batches = 0

            for _ in range(len(labeled_loader)):
                # Labeled batch
                batch_l = next(labeled_iter, None)
                if batch_l is None:
                    break

                s0_obs_l, s0_act_l, s1_obs_l, s1_act_l, mu_l, mask0_l, mask1_l = [x.to(device) for x in batch_l]

                rewards_s0_l = self(s0_obs_l, s0_act_l)
                rewards_s1_l = self(s1_obs_l, s1_act_l)
                loss_labeled = loss_fn(rewards_s0_l, rewards_s1_l, mu_l, mask0_l, mask1_l)

                # Unlabeled batch
                batch_u = next(unlabeled_iter, None)
                if batch_u is not None:
                    s0_obs_u, s0_act_u, s1_obs_u, s1_act_u, _, mask0_u, mask1_u = [x.to(device) for x in batch_u]

                    rewards_s0_u = self(s0_obs_u, s0_act_u)
                    rewards_s1_u = self(s1_obs_u, s1_act_u)

                    loss_unlabeled, confident_ratio = self.surf_loss(rewards_s0_u, rewards_s1_u, mask0_u, mask1_u)
                    total_confident_ratio += confident_ratio
                    num_unlabeled_batches += 1
                else:
                    loss_unlabeled = torch.tensor(0.0, requires_grad=True, device=device)

                # Combine and optimize
                total_loss = loss_labeled + lambda_u * loss_unlabeled

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_labeled_loss += loss_labeled.item()
                total_unlabeled_loss += loss_unlabeled.item()

            # Epoch 평균 계산
            avg_labeled_loss = total_labeled_loss / len(labeled_loader)
            avg_unlabeled_loss = total_unlabeled_loss / len(labeled_loader)
            avg_confident_ratio = total_confident_ratio / max(1, num_unlabeled_batches)
            total_combined_loss = avg_labeled_loss + lambda_u * avg_unlabeled_loss

            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(data_loader=val_loader, loss_fn=loss_fn)
            else:
                val_loss = 0.0

            # CSV 기록
            with open(self.log_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch + 1,
                    avg_labeled_loss,
                    avg_unlabeled_loss,
                    total_combined_loss,
                    val_loss,
                    avg_confident_ratio
                ])

            # Best 모델 저장
            if total_combined_loss < best_loss:
                best_loss = total_combined_loss
                torch.save(self.state_dict(), self.path)

        print("MR-SURF training completed.")