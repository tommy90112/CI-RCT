# CI-RCT Experiments Log

紀錄各次實驗的參數設定與結果，方便後續比較與論文撰寫。

---

## 資料集說明

| 項目 | 說明 |
|---|---|
| Dataset | Elliptic++ (KDD 2023) |
| Node types | `transaction` (~203K), `wallet` (~900K connected) |
| Edge types | wallet→tx, tx→wallet, tx→tx, wallet→wallet |
| Features | transaction: 110-dim (93 local + 17 stats), wallet: 54-dim |
| Labels | transaction only；illicit=1, licit=0, unknown=excluded |
| Train/Val/Test split | 70 / 15 / 15 (stratified on labeled transactions) |
| Evaluation target | `transaction` nodes only |

---

## 預設訓練參數（未特別標註即使用此值）

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--epochs` | 200 | 訓練總 epoch 數 |
| `--lr` | 1e-3 | Adam 學習率 |
| `--hidden_dim` | 128 | HGT 隱藏層維度 |
| `--num_hgt_layers` | 3 | HGT 層數 |
| `--num_heads` | 4 | HGT multi-head 數 |
| `--dropout` | 0.3 | Dropout 比率 |
| `--type_emb_dim` | 16 | Node type embedding 維度 |
| `--max_hops` | 5 | Root cause tracing 最大 hop |
| `--ce_threshold` | 0.1 | Causal effect 閾值 |
| `--node_limit` | 500 | Causal subgraph 最大節點數 |
| `--lambda_adversarial` | 0.1 | λ₁：WGAN-GP adversarial loss 權重 |
| `--lambda_stability` | 0.5 | λ₂：Causal Shapley stability loss 權重 |
| `--n_critic` | 5 | Discriminator steps per Generator step |
| `--gp_weight` | 10.0 | WGAN-GP gradient penalty 係數 |
| `--noise_std` | 0.05 | Generator Gaussian noise std |
| `--seed` | 42 | 隨機種子 |
| `--device` | cpu | 計算裝置 |
| `--eval_every` | 10 | 每幾個 epoch 評估一次 |

---

## 實驗紀錄

---

### EXP-001 ～ EXP-003：subsample_tx 大小對效能的影響

**目的：** 探討在記憶體限制下，subsample transaction 節點數對偵測效果的影響。

**Subsample 策略：** 保留所有 fraud 節點 + 隨機抽樣 licit 節點（stratified）。

**共同設定：**

| 參數 | 值 |
|---|---|
| `--dataset` | elliptic++ |
| `--use_gan` | true（Phase 2, GAN enabled）|
| `--epochs` | 200 |
| 其餘參數 | 預設值 |

**結果：**

| Exp | `--subsample_tx` | 約 fraud 節點 | 約 licit 節點 | fraud:licit 比 | Test F1 | Test AUC |
|---|---|---|---|---|---|---|
| EXP-001 | 20,000 | ~5K | ~15K | ≈1:3 | **0.8870** | 0.9341 |
| EXP-002 | 30,000 | ~5K | ~25K | ≈1:5 | 0.8758 | 0.9320 |
| EXP-003 | 40,000 | ~5K | ~35K | ≈1:7 | 0.8601 | **0.9361** |

**觀察與分析：**
- F1 隨 subsample 增大而下降：licit 節點越多，class imbalance 越嚴重，決策邊界偏移，F1 越難提升。
- AUC 維持穩定（0.932～0.936）：模型的排名能力（ranking quality）不受 subsample 大小影響，模型有學到有效的 fraud 表示。
- Class weight 補正（`neg/pos` 比值）雖已啟用，但 20K 時 imbalance 最輕，綜合效果最佳。
- **暫定 `subsample_tx=20000` 作為後續實驗的基準設定。**

---

## 待做實驗

- [ ] 不同 `--lambda_adversarial` 值（0.01, 0.1, 0.5）對結果的影響
- [ ] 不同 `--num_hgt_layers`（2, 3, 4）對結果的影響
- [ ] 關閉 GAN（`--use_gan false`）作為 ablation baseline
- [ ] 確認 Elliptic++ 是否有 wallet-level label 以進行 wallet-level evaluation（對標 SAGE-FIN）
- [ ] 與 SAGE-FIN transaction-level 結果正式比較

---

## 與 SAGE-FIN 比較說明

SAGE-FIN 原論文分別報告：
1. **Transaction-level detection**
2. **Wallet/Address-level detection**

目前 CI-RCT 僅對 `transaction` 節點有標籤與評估，可對標 SAGE-FIN 的 **transaction detection** 欄位。

Wallet-level evaluation 需確認 Elliptic++ 原始資料是否附帶 `wallets_classes.csv`（wallet illicit labels）。若有，需在 `elliptic_plus_loader.py` 加入 `data["wallet"].y` 及對應 mask。

---

*最後更新：2026-04-08*
