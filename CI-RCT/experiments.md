# CI-RCT 實驗記錄

## 資料集

**Elliptic++（KDD 2023）**

| 節點類型 | 數量 |
|---------|------|
| transaction | ~203,000 |
| wallet（connected） | ~900,000 |

| 邊類型 | 數量 |
|--------|------|
| wallet → transaction (sends) | — |
| transaction → wallet (pays) | — |
| transaction → transaction (flows_to) | — |
| wallet → wallet (connects / addr→addr) | ~2,870,000 |

標籤分布（transaction 節點）：
- illicit（詐欺）：~10%
- licit（正常）：~90%（僅有標籤的 ~46K 節點參與訓練）

Train / Val / Test 切分：70 / 15 / 15（stratified）

---

## 固定參數（所有實驗共用）

| 參數 | 值 |
|------|-----|
| dataset | elliptic++ |
| data_root | data |
| epochs | 200 |
| lr | 1e-3 |
| num_hgt_layers | 3 |
| num_heads | 4 |
| dropout | 0.3 |
| type_emb_dim | 16 |
| max_hops | 5 |
| ce_threshold | 0.1 |
| node_limit | 500 |
| lambda_adversarial | 0.1 |
| n_critic | 5 |
| gp_weight | 10.0 |
| noise_std | 0.05 |
| eval_every | 10 |
| device | cuda |
| seed | 42 |

---

## 實驗結果

### Exp-01｜基準模型（最佳結果）

**目的：** 移除 addr→addr 邊以降低 GPU 記憶體壓力，full-batch 訓練。

| 參數 | 值 |
|------|-----|
| hidden_dim | 128 |
| include_addr_addr | false |
| labeled_only | false |
| use_gan | true |
| lambda_stability | 0.5 |

**指令：**
```bash
python train.py \
  --dataset elliptic++ \
  --data_root data \
  --epochs 200 \
  --use_gan true \
  --lambda_stability 0.5 \
  --eval_every 10 \
  --device cuda
```

**結果：**

| 指標 | 數值 |
|------|------|
| Test F1 | **0.7728** |
| Test AUC | **0.9630** |

**備註：** 目前最佳結果，超過 SAGE-FIN baseline（~0.75）。

---

### Exp-02｜加回 addr→addr（OOM）

**目的：** 測試加入 wallet→wallet 邊是否能提升效能。

| 參數 | 值 |
|------|-----|
| hidden_dim | 128 |
| include_addr_addr | **true** |
| use_gan | true |
| lambda_stability | 0.5 |

**結果：** ❌ CUDA Out of Memory

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 620.00 MiB.
GPU 0 total: 47.40 GiB / PyTorch allocated: 45.60 GiB
```

**結論：** 47GB GPU 在 full-batch + addr→addr 的情況下記憶體不足，無法訓練。

---

### Exp-03｜加回 addr→addr + 縮小 hidden_dim

**目的：** 縮小模型容量以便加入 addr→addr 邊。

| 參數 | 值 |
|------|-----|
| hidden_dim | **64** |
| include_addr_addr | **true** |
| use_gan | true |
| lambda_stability | 0.5 |

**指令：**
```bash
python train.py \
  --dataset elliptic++ \
  --data_root data \
  --include_addr_addr true \
  --hidden_dim 64 \
  --epochs 200 \
  --use_gan true \
  --lambda_stability 0.5 \
  --eval_every 10 \
  --device cuda
```

**結果：**

| 指標 | 數值 |
|------|------|
| Test F1 | 0.7290 |
| Test AUC | 0.9611 |

**備註：** 相較 Exp-01，F1 下降 0.044。hidden_dim 縮小造成表達能力下降，addr→addr 邊可能也引入雜訊，兩者共同導致效能降低。

---

### Exp-04｜fraud_subgraph（詐欺中心子圖 + addr→addr）

**目的：** 以有標籤 tx 為中心，BFS 2 hop 收集周圍 wallet 子圖並保留 addr→addr 邊，在記憶體可行的前提下引入 wallet→wallet 結構訊號。

**程式改動：**
- `elliptic_plus_loader.py` 新增 `fraud_subgraph` 參數與 `_fraud_subgraph_wallets()` helper
- `train.py` 新增 `--fraud_subgraph`、`--fraud_subgraph_hops` CLI 參數
- wallet 節點從全量 ~900K 縮減至以標籤 tx 為錨點的 2-hop 子集（約數萬），addr→addr 邊只保留子集內部的連結

**子圖建構邏輯：**
1. 找出所有有標籤 tx（class 1 or 2）
2. Hop 1：收集直接與這些 tx 相連的 wallet（via AddrTx / TxAddr）
3. Hop 2：從 hop-1 wallet 再往外一步，透過 addr→addr 收集鄰居 wallet
4. 只保留 hop-1 + hop-2 wallet 及它們之間的 addr→addr 邊

| 參數 | 值 |
|------|-----|
| hidden_dim | 128 |
| include_addr_addr | false（由 fraud_subgraph 內部控制） |
| fraud_subgraph | **true** |
| fraud_subgraph_hops | **2** |
| use_gan | true |
| lambda_stability | 0.5 |
| GPU | CUDA_VISIBLE_DEVICES=1（A6000, 49GB） |

**指令：**
```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset elliptic++ \
  --data_root data \
  --epochs 200 \
  --use_gan true \
  --lambda_stability 0.5 \
  --fraud_subgraph true \
  --fraud_subgraph_hops 2 \
  --eval_every 10 \
  --device cuda
```

**結果：**

| 指標 | 數值 |
|------|------|
| Test F1 | 0.7626 |
| Test AUC | **0.9669** |

**備註：** AUC 比 Exp-01 提升 0.004（0.9630 → 0.9669），顯示 addr→addr 子圖確實提供了有用的排序訊號。F1 略低（0.7728 → 0.7626），可能因為子圖過濾改變了 wallet 節點的訊息傳遞範圍。整體而言 addr→addr 對模型有正面貢獻。

---

## 綜合比較

| 實驗 | hidden_dim | addr→addr 策略 | use_gan | F1 | AUC |
|------|-----------|---------------|---------|-----|-----|
| Exp-01（F1 最佳） | 128 | 無 | ✓ | **0.7728** | 0.9630 |
| Exp-02 | 128 | 全量（OOM） | ✓ | OOM | — |
| Exp-03 | 64 | 全量 | ✓ | 0.7290 | 0.9611 |
| Exp-04（AUC 最佳） | 128 | fraud_subgraph hops=2 | ✓ | 0.7626 | **0.9669** |

---

## 與 Baseline 對比

| 方法 | F1（illicit） |
|------|-------------|
| GCN | ~0.65 |
| GraphSAGE | ~0.70 |
| SAGE-FIN | ~0.75 |
| **CI-RCT Exp-01** | **0.7728** |
| **CI-RCT Exp-04** | 0.7626 |

---

## 待跑實驗（消融實驗）

| 實驗 | 用途 | 狀態 |
|------|------|------|
| Exp-05：use_gan=false | 驗證 GAN 模組的貢獻 | ⬜ 待跑 |
| Exp-06：lambda_stability=0.0 | 驗證 stability loss 的貢獻 | ⬜ 待跑 |
| Exp-07：fraud_subgraph hops=1 | 比較 1-hop vs 2-hop wallet 子圖效果 | ⬜ 待跑 |
