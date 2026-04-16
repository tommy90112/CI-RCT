# CI-RCT：基於因果干預之異質圖神經網路根因追溯的可解釋性研究
## 論文提案文件（Thesis Proposal）

> **Version:** Draft v0.4
> **Status:** 提案階段（Proposal Stage）
> **Last Updated:** 2026-04-01
> **Author:** Tommy（淡江大學統計學研究所）
> **定位策略:** 獨立全新框架，以 Pearl SCM、NCM、Asymmetric Causal Shapley 為理論基礎，從頭設計

---

## 目錄

1. [研究概覽](#1-研究概覽)
2. [研究背景與動機](#2-研究背景與動機)
3. [核心研究問題與目標](#3-核心研究問題與目標)
4. [相關文獻定位](#4-相關文獻定位)
5. [模型架構設計](#5-模型架構設計)
6. [理論基礎說明](#6-理論基礎說明)
7. [實驗規劃](#7-實驗規劃)
8. [論文章節結構](#8-論文章節結構)
9. [開發路線圖](#9-開發路線圖)
10. [口試問答預備](#10-口試問答預備)
11. [參考文獻核心清單](#11-參考文獻核心清單)

---

## 1. 研究概覽

| 項目 | 內容 |
|------|------|
| **論文題目（英）** | CI-RCT: A Causal Intervention Framework for Root Cause Tracing and Explainability on Heterogeneous Graphs |
| **論文題目（中）** | CI-RCT：基於因果干預之異質圖神經網路根因追溯可解釋性研究 |
| **研究類型** | 方法論研究（全新框架提出 + 多資料集實驗驗證） |
| **核心技術** | Pearl SCM、do-calculus、Neural Causal Model（NCM）、Asymmetric Causal Shapley、Heterogeneous GNN（HGT）、Root Cause Tracing、Causal Adversarial GAN |
| **應用場景** | 金融詐騙偵測（以 Elliptic++ Bitcoin 交易異質圖為主要驗證場景） |
| **理論工具來源** | Pearl (2009)、Heskes et al. (NeurIPS 2020)、Frye et al. (NeurIPS 2020)、Xia et al. (2021)、Behnam & Wang (ECCV 2024) |
| **框架定位** | 獨立全新設計，CXGNN 等先行工作為 Related Work 比較對象 |
| **Dataset-agnostic** | 是，任何可表示為有向異質圖的資料集均可適用 |

### 1.1 核心論文 Claim

> CI-RCT 是首個針對**異質圖**場景，**從頭設計**，同時解決下列三個問題的端到端統一框架：
>
> 1. **可解釋根因追溯**：以 Pearl SCM 和 do-calculus 為基礎，設計類型感知的 Neural Causal Model，採用 **Asymmetric Causal Shapley**（Frye et al. & Heskes et al., NeurIPS 2020）計算跨節點類型的因果貢獻，並透過反向因果追溯演算法找出詐騙行為的資金源頭節點，輸出完整的可解釋因果路徑
>
> 2. **類別不平衡 + 偽裝對抗魯棒性**：設計因果約束 GAN（Causal Adversarial GAN），以 DAG 拓撲排序為約束條件生成偽裝詐騙節點，透過對抗博弈讓 GNN 學會識破偽裝，同時解決詐騙節點極度稀少的類別不平衡問題
>
> 3. **因果解釋穩定性保證**：在聯合損失函數中引入 Causal Consistency Loss，確保對抗訓練過程中 Causal Shapley φ 值不因圖結構擾動而崩潰，保障可解釋性在對抗強化後依然穩定

### 1.2 與現有方法的核心區別聲明

CI-RCT **不是**對任何現有方法的擴展（extension）。它以 Pearl SCM、Neural Causal Model（NCM）、Asymmetric Causal Shapley Value 等成熟理論工具為基礎，針對「**異質圖上的詐騙根因追溯**」這個現有工作尚未同時解決的問題，從頭設計了一個統一的四模組框架。CXGNN、CaT-GNN 等先行工作是本研究的 Related Work 比較對象，而非繼承基礎。

---

## 2. 研究背景與動機

### 2.1 三個核心 Research Gaps

現有 GNN 詐騙偵測方法存在三個尚未被同時解決的核心缺口：

**Gap 1 — 缺乏可解釋的根因追溯能力**

現有 GNN 可解釋方法（GNNExplainer、PGExplainer）主要基於統計**相關性**（association），只能回答「哪個子圖對預測結果統計上最相關」，無法回答「詐騙資金從哪個節點起源、沿何條路徑流動」。這在金融監管場景中是根本性的缺陷——執法單位需要可追訴的資金鏈路徑與根因節點，而非統計顯著的子圖遮罩。

此外，現有因果 GNN 解釋器（如 CXGNN、OrphicX）均針對**同質圖**設計，直接應用到異質圖（user、wallet、transaction 等多類型節點共存）時，會喪失節點類型與邊類型的語義資訊，因果效應的計算失去類型區分，導致解釋結果不可靠。

**Gap 2 — 極度類別不平衡**

詐騙資料天然呈現嚴重的類別不平衡，詐騙節點通常只佔全體節點的 1–5%。在 GNN 的 message passing 過程中，少數類節點的特徵訊號容易被多數類鄰居「稀釋」，導致模型對詐騙特徵的學習不足，召回率偏低。

**Gap 3 — 詐騙者的主動偽裝行為（Camouflage）**

真實世界的詐騙者會主動建立與正常節點的連結（偽裝正常交易行為），讓自身節點特徵與行為模式刻意接近正常用戶。這種**偽裝行為**使 GNN 的鄰域聚合機制失效，正常鄰居的訊號淹沒了詐騙節點的異常訊號。

> **關鍵洞察：Gap 2 與 Gap 3 相互強化。** 詐騙節點越稀少，模型學習對抗偽裝行為的訓練樣本越少，魯棒性越弱；偽裝行為又讓稀少的詐騙樣本更難被正確識別。這兩個問題必須被**一起處理**才能從根本上解決。

### 2.2 現有方法的不足：系統性比較

| 方法類型 | 代表工作 | Gap 1（根因追溯）| Gap 2+3（不平衡+偽裝）| 備註 |
|---------|---------|:---:|:---:|------|
| 關聯性解釋器 | GNNExplainer、PGExplainer | ✗ | ✗ | 基於相關性，非因果 |
| 反事實解釋器 | CF-GNNExplainer、CLEAR | ✗ | ✗ | 無正向根因追溯 |
| 因果 GNN 解釋器 | CXGNN（ECCV 2024） | 部分 | ✗ | 同質圖、無根因追溯、無對抗 |
| 異質圖解釋器 | HGExplainer、CaGE | ✗ | ✗ | 無干預式因果、無根因 |
| 對抗訓練 GNN | CARE-GNN、DAGCN | ✗ | 部分 | 無因果解釋、無根因追溯 |
| 因果詐騙偵測 | CaT-GNN、SAGE-FIN | ✗ | ✗ | 缺乏系統性解釋與根因輸出 |
| **CI-RCT（本研究）** | — | **✓** | **✓** | 四個維度統一解決 |

### 2.3 研究動機的關鍵文獻支撐

SAGE-FIN（Nguyen et al., xAI 2025）是目前在 Elliptic++ 上唯一同時提供 GNN 詐騙偵測與因果解釋的工作。然而，其作者明確指出所使用的 Granger 因果在理論上屬於**相關性**方法，存在無法排除混淆因子的根本限制，並明確建議後續工作應升級到 Pearl 的干預式因果（do-calculus）。本研究直接回應這一研究號召，在 Elliptic++ 異質圖上建立首個基於 Pearl SCM 的干預式根因追溯框架。

---

## 3. 核心研究問題與目標

### 3.1 研究問題

> **RQ1**：如何從頭設計一個類型感知的結構因果模型（Typed SCM），使 do-calculus 的干預計算能在異質圖上感知節點類型與邊類型的語義差異，計算出有意義的跨類型因果效應？

> **RQ2**：基於因果干預估計的因果效應分數，如何設計一個有效的反向追溯演算法，從被偵測為詐騙的目標節點反向追溯到資金的根因源頭，並輸出完整的可解釋因果路徑？

> **RQ3**：如何設計因果約束 GAN（Causal-Constrained Adversarial GAN），在解決類別不平衡與偽裝問題的對抗訓練過程中，同時維持因果解釋的穩定性（即 Causal Shapley φ 值不因圖擾動而崩潰）？

> **RQ4**：CI-RCT 框架是否具備跨領域的泛化能力（dataset-agnostic），能在金融詐騙以外的異質圖任務上同樣有效？

### 3.2 研究目標

- **目標 1**：設計 `TypedCausalGraph`，一個支援異質節點類型與有向邊類型的結構因果模型建構模組，作為因果干預計算的圖結構基礎
- **目標 2**：設計 `HeteroNCM`，一個類型感知的 Neural Causal Model，為每種邊類型建立獨立的因果效應估計器，以 **Asymmetric Causal Shapley Value**（Frye et al. & Heskes et al., NeurIPS 2020）計算節點的時序感知因果貢獻分數，使根因節點在因果歸因中自然獲得更高權重
- **目標 3**：設計 `RootCauseTracer`，一個基於因果效應分數的反向 BFS 追溯演算法，從目標節點沿最強因果方向反向追溯，輸出根因節點與完整因果鏈
- **目標 4**：設計 `CausalAdversarialGAN`，以 DAG 拓撲排序為生成約束，透過 Generator-Discriminator 對抗博弈強化模型對偽裝詐騙的識別能力，並透過 Causal Consistency Loss 確保解釋穩定性
- **目標 5**：在 Elliptic++（主要場景）與 DBLP、ACM、IMDB（泛化驗證）等多個異質圖資料集上進行完整實驗驗證

---

## 4. 相關文獻定位

### 4.1 CI-RCT 所在的研究空白

本研究定位於以下四個維度的交叉空白點：

```
干預式因果（Interventional Causality / Pearl SCM + do-calculus）
                    ×
        根因追溯（Root Cause Tracing）
                    ×
   異質圖可解釋性（Heterogeneous GNN Explainability）
                    ×
    對抗魯棒性（Adversarial Robustness against Camouflage）
```

**核心發現：目前尚無完整工作同時涵蓋這四個維度。**

### 4.2 相關工作的定位與差距

#### 4.2.1 與最相關的七篇工作的差距分析

| 論文 | Venue | 與本研究的關係 | 與 CI-RCT 的差距 |
|------|-------|-------------|----------------|
| **CXGNN** (Behnam & Wang) | ECCV 2024 | 共享 NCM 理論工具，為 Baseline 比較對象 | 同質圖、圖分類任務、無根因追溯、無異質性感知、無對抗訓練 |
| **CaGE** | TOIS 2025 | 異質圖因果解釋的最近相關工作 | Granger 因果（非干預式）、推薦系統場景、無根因追溯 |
| **REASON** (Wang et al.) | KDD 2023 | 圖結構根因追溯代表性工作 | AIOps 場景、非干預式因果、同質圖 |
| **CaT-GNN** (Duan et al.) | arXiv 2024 | 因果干預用於金融 GNN 最新工作 | causal mixup 方法（非 SCM）、同質圖、無根因追溯 |
| **SAGE-FIN** (Nguyen et al.) | xAI 2025 | Elliptic++ 上唯一的因果解釋先行工作 | Granger 因果（非干預式）、無根因追溯、無對抗訓練 |
| **CARE-GNN** (Dou et al.) | CIKM 2020 | 抗偽裝 GNN 偵測代表作 | 無因果解釋、無根因追溯、無類型感知 |
| **DAGCN** (Wan et al.) | NPL 2026 | 對抗訓練 GNN 詐騙偵測最新工作 | 無因果解釋、無根因追溯 |

#### 4.2.2 CI-RCT 與 CXGNN 的關係說明（重要）

CI-RCT 與 CXGNN 共享相同的理論基礎（Pearl SCM、NCM、Causal Shapley），就如同許多論文共享 Transformer 架構但提出各自的新方法一樣。然而，CI-RCT 是針對**不同問題**從頭設計的獨立框架：

| 面向 | CXGNN | CI-RCT |
|------|-------|--------|
| 圖的類型 | 同質圖 | 異質圖（多節點/邊類型） |
| 任務 | 圖分類的解釋（Graph Classification） | 節點級根因追溯（Node-level Root Cause Tracing） |
| SCM 建構 | 無類型區分的 CausalGraph | 類型感知的 TypedCausalGraph |
| NCM 設計 | 單一 NNModel | Per-edge-type 獨立 HeteroNCM |
| 追溯能力 | 無 | 反向 BFS 根因追溯（RootCauseTracer） |
| 對抗訓練 | 無 | 因果約束 GAN（CausalAdversarialGAN） |
| 損失函數 | BCE + NCM loss | BCE + GAN loss + Causal Consistency Loss |

### 4.3 CI-RCT 的五項原創貢獻聲明

1. **TypedCausalGraph**：首個在有向異質圖上建構類型感知 SCM 的方法，使 do-calculus 介入計算感知節點類型與邊類型的語義差異
2. **HeteroNCM + Asymmetric Causal Shapley φ**：首個在異質圖節點分類任務上，以 DAG 拓撲排序為約束計算時序感知 Asymmetric Causal Shapley Value 的因果效應估計器，使根因節點的 φ 值天然高於中間傳遞節點
3. **RootCauseTracer**：首個基於干預式因果效應分數執行反向節點追溯、輸出完整因果路徑的根因追溯演算法
4. **CausalAdversarialGAN**：首個在對抗訓練的 Generator 設計中加入 DAG 拓撲排序約束的因果感知偽裝生成方法
5. **Causal Consistency Loss**：首個以 Causal Shapley 穩定性作為對抗訓練正則化目標的因果解釋保護機制

---

## 5. 模型架構設計

### 5.1 設計哲學

CI-RCT 的設計遵循三個核心原則：

- **理論嚴謹性**：以 Pearl (2009) 的 SCM 和 do-calculus 為數學基礎，而非 attention 權重或 gradient 等代理指標
- **異質性感知**：所有模組（因果圖、NCM、追溯演算法、GAN）均原生支援多節點類型與多邊類型，不做同質化簡化
- **端到端整合**：四個模組共享一個統一的聯合損失函數，訓練過程中相互促進而非獨立最佳化

### 5.2 整體框架概覽

```
輸入：有向異質圖 G = (V, E, τ_v, τ_e, T)
        │  τ_v：節點類型映射   τ_e：邊類型映射   T：時間戳
        ▼
┌──────────────────────────────────────────┐
│  Module 1：HeteroGNN Backbone（HGT）      │
│  ─────────────────────────────────────── │
│  輸入：HeteroData（多類型節點/邊）          │
│  輸出：h_v（per-type embedding）           │
│        ŷ_v（節點詐騙預測分數）              │
└──────────────────┬───────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────────┐  ┌────────────────────────────┐
│  Module 2：        │  │  Module 4（訓練期啟用）：    │
│  Causal           │  │  CausalAdversarialGAN       │
│  Intervention     │  │  ───────────────────────── │
│  Engine           │  │  Generator：               │
│  ──────────────── │  │    以真實詐騙節點為條件       │
│  TypedCausalGraph │  │    生成偽裝詐騙節點           │
│  ＋ HeteroNCM     │  │    約束：符合 DAG 拓撲排序    │
│  ＋ Causal        │  │  Discriminator = HeteroGNN  │
│    Shapley φ      │  │    識破 Generator 的偽裝      │
│                   │  └────────────────────────────┘
│  → CE(u→v)        │
└─────────┬─────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│  Module 3：RootCauseTracer               │
│  ─────────────────────────────────────── │
│  輸入：目標詐騙節點 v*、CE score 字典      │
│  演算法：沿最強因果效應方向反向 BFS         │
│  輸出：根因節點 r                          │
│        因果鏈 [v* → v_{k-1} → ... → r]   │
│        每條邊的 causal_score              │
└──────────────────────────────────────────┘
          │
          ▼
最終輸出：
  ① 詐騙節點標記（0/1）+ Causal Shapley φ 分數（解釋為何）
  ② 根因節點 r（資金最上游起源）
  ③ 完整因果鏈路徑（帶每跳 CE score）
  ④ 每條邊的 causal_score = φ_src × φ_dst × temporal_weight
```

### 5.3 Module 1：HeteroGNN Backbone

**目標**：對有向異質圖執行節點分類，輸出每個節點的 embedding 和預測分數，作為後續模組的輸入基礎。

**設計選擇**：採用 HGT（Heterogeneous Graph Transformer，Hu et al., KDD 2020）作為 backbone。HGT 原生支援多類型節點與多類型邊的注意力機制，每種（source type, edge type, target type）三元組有獨立的注意力頭，符合 CI-RCT 對異質性感知的核心要求。

```python
from torch_geometric.nn import HGTConv, Linear
import torch.nn as nn

class HeteroGNNBackbone(nn.Module):
    """
    HGT-based 異質圖節點分類 Backbone。

    dataset-agnostic 設計：
      只需提供 metadata（節點類型列表、邊類型三元組列表）
      即可自動建立對應的注意力機制，不依賴任何特定資料集結構。
    """
    def __init__(self, metadata, hidden_channels=64, num_heads=4, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            for _ in range(num_layers)
        ])
        # 每種節點類型的輸出分類器
        self.classifiers = nn.ModuleDict({
            node_type: Linear(hidden_channels, 1)
            for node_type in metadata[0]  # metadata[0] = node_type 列表
        })

    def forward(self, x_dict, edge_index_dict):
        """
        Returns
        -------
        h_dict   : {node_type: embedding_tensor}
        pred_dict: {node_type: fraud_probability}
        """
        h_dict = x_dict
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: v.relu() for k, v in h_dict.items()}

        pred_dict = {
            ntype: self.classifiers[ntype](h).sigmoid()
            for ntype, h in h_dict.items()
            if ntype in self.classifiers
        }
        return h_dict, pred_dict
```

**Dataset-agnostic 介面**：Module 1 不假設任何特定節點類型或邊類型名稱，使用者只需提供 PyG HeteroData 格式即可：

```python
data = HeteroData()
data['transaction'].x = tx_features   # 任意節點類型
data['actor'].x = actor_features
data[('actor', 'sends', 'transaction')].edge_index = ei_1
data[('transaction', 'to', 'actor')].edge_index = ei_2
# ... 任意其他類型節點與邊
```

### 5.4 Module 2：Causal Intervention Engine

這是 CI-RCT 的**核心理論貢獻**，包含三個緊密整合的子元件。

#### 5.4.1 TypedCausalGraph：類型感知的有向 SCM

**設計動機**：Pearl SCM 要求每個節點的因果機制 $v_i = f_i(\text{Pa}(v_i), U_i)$ 中，$\text{Pa}(v_i)$ 是有向父節點集合。在異質圖中，不同類型的父節點對目標節點的因果影響在語義上截然不同（例如 wallet 節點和 user 節點作為 transaction 的父節點，其因果貢獻性質不同）。TypedCausalGraph 在有向 DAG 結構的基礎上，記錄每條有向邊的類型資訊，使後續的 do-calculus 計算能感知這種語義差異。

```python
class TypedCausalGraph:
    """
    類型感知的有向因果圖（Typed Directed Causal Graph）。

    對應 Pearl SCM 的有向無環圖（DAG）結構，擴展為：
      - 節點有類型標記 τ_v
      - 有向邊有類型標記 τ_e
      - 維護嚴格的父節點集合 Pa(v)（do-calculus 截斷操作的基礎）
      - 支援時間戳（時序先行性驗證的基礎）

    與 CXGNN 的 CausalGraph 的關鍵區別：
      CXGNN 的 CausalGraph 使用無向鄰居字典 fn[v]，
      TypedCausalGraph 使用有向父節點字典 pa[v] 和子節點字典 ch[v]，
      且每個鄰接關係均附帶節點類型和邊類型資訊。
    """
    def __init__(self, V, node_types: dict, timestamps: dict = None):
        self.v = list(V)
        self.set_v = set(V)
        self.node_type = node_types          # {node_id: type_str}
        self.timestamps = timestamps or {}   # {(src, dst): timestamp}

        # 有向鄰接結構（do-calculus 所需）
        self.pa: dict[int, set] = {v: set() for v in V}  # 父節點
        self.ch: dict[int, set] = {v: set() for v in V}  # 子節點

        # 類型感知的鄰接資訊
        self.pa_typed: dict = {v: {} for v in V}
        # pa_typed[v][u] = (edge_type, u_node_type)
        self.edge_type_map: dict = {}
        # edge_type_map[(src, dst)] = edge_type

    def add_edge(self, src, dst, edge_type: str) -> None:
        """
        加入有向邊 src → dst，記錄類型資訊。
        語義：src 是 dst 的直接因果原因（parent）。
        時序驗證：若提供時間戳，確保 ts(src) < ts(dst)。
        """
        if src not in self.set_v or dst not in self.set_v:
            return
        # 時序先行性驗證（Granger 因果的圖結構對應）
        if self.timestamps:
            ts_src = self.timestamps.get(src)
            ts_dst = self.timestamps.get(dst)
            if ts_src and ts_dst and ts_src >= ts_dst:
                return  # 違反時序先行性，不加入

        self.pa[dst].add(src)
        self.ch[src].add(dst)
        self.pa_typed[dst][src] = (edge_type, self.node_type.get(src))
        self.edge_type_map[(src, dst)] = edge_type

    def parents(self, node) -> set:
        """回傳 node 的直接因果父節點集合 Pa(v)。"""
        return self.pa.get(node, set())

    def topological_order(self) -> list:
        """Kahn's algorithm 拓撲排序（上游 → 下游）。"""
        in_deg = {v: len(self.pa[v]) for v in self.v}
        from collections import deque
        q = deque([v for v in self.v if in_deg[v] == 0])
        order = []
        while q:
            node = q.popleft()
            order.append(node)
            for child in self.ch[node]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    q.append(child)
        return order
```

#### 5.4.2 HeteroNCM：類型感知的 Neural Causal Model

**理論對應**：Pearl SCM 中每個節點的結構方程式 $v_i = f_i(\text{Pa}(v_i), U_i)$ 在 HeteroNCM 中以神經網路近似。關鍵創新是為每種邊類型建立**獨立的** NNModel，使不同語義的邊（如 wallet→transaction 的入金邊 vs. user→transaction 的操作邊）各自學習獨立的因果機制。

```python
class HeteroNCM(nn.Module):
    """
    類型感知的 Neural Causal Model。

    核心設計：
      Per-edge-type NNModel：每種邊類型有獨立的神經網路估計因果效應
      Type Embedding：節點類型以 embedding 向量表示，加入輸入特徵
      do-calculus 介入：截斷指定父節點的入邊（強制設為基準值 0.5）

    理論對應：
      CXGNN 論文 Definition 3 / Theorem 2（GNN-NCM 的 G-Constrained 設計）
      CI-RCT 的擴展：G-Constrained + Type-Aware + Heterogeneous Edge Models
    """
    def __init__(self, graph: TypedCausalGraph, target_node: int,
                 node_emb_dim: int = 64, type_emb_dim: int = 16,
                 h_size: int = 64, h_layers: int = 2,
                 learning_rate: float = 0.005):
        super().__init__()
        self.graph = graph
        self.target_node = target_node

        # 節點類型 embedding
        all_node_types = list(set(graph.node_type.values()))
        self.type_to_idx = {t: i for i, t in enumerate(all_node_types)}
        self.type_emb = nn.Embedding(len(all_node_types), type_emb_dim)

        # 父節點集合（do-calculus 的截斷對象）
        self.parents = list(graph.parents(target_node))

        # Per-edge-type 獨立 NNModel
        all_edge_types = list(set(graph.edge_type_map.values()))
        input_size = 1 + len(self.parents) * 2 + type_emb_dim
        self.edge_models = nn.ModuleDict({
            et: self._build_nn(input_size, h_size, h_layers)
            for et in all_edge_types
        })

    def _build_nn(self, input_size, h_size, h_layers):
        layers = [nn.Linear(input_size, h_size), nn.ReLU()]
        for _ in range(h_layers - 1):
            layers += [nn.Linear(h_size, h_size), nn.ReLU()]
        layers.append(nn.Linear(h_size, 1))
        return nn.Sequential(*layers)

    def forward(self, node_labels: dict, intervene_nodes: set = None) -> torch.Tensor:
        """
        計算 target_node 的預測分數，支援 do-calculus 介入。

        Parameters
        ----------
        node_labels  : {node_id: label_value}，各節點的觀測標籤
        intervene_nodes : do(X = 0.5) 的節點集合（截斷入邊，設為基準值）

        Returns
        -------
        prediction : shape (1,)，target_node 的因果預測分數
        """
        if intervene_nodes is None:
            intervene_nodes = set()

        # 建立輸入向量
        y_target = torch.tensor(
            [float(node_labels.get(self.target_node, 0.0))]
        )
        target_type = self.graph.node_type.get(self.target_node, 'unknown')
        type_idx = torch.tensor(self.type_to_idx.get(target_type, 0))
        type_emb_vec = self.type_emb(type_idx)

        parent_features = []
        dominant_edge_type = None
        for pa in self.parents:
            # do-calculus 介入：被截斷的節點強制設為基準值
            if pa in intervene_nodes:
                u_node = torch.tensor([0.5])
                u_edge = torch.tensor([0.5])
            else:
                u_node = torch.tensor([float(node_labels.get(pa, 0.0))])
                u_edge = torch.tensor([0.5])  # 邊效應初始值，訓練時更新
            parent_features.extend([u_node, u_edge])
            dominant_edge_type = self.graph.edge_type_map.get(
                (pa, self.target_node), list(self.edge_models.keys())[0]
            )

        u = torch.cat([y_target] + parent_features + [type_emb_vec], dim=0)
        model = self.edge_models.get(
            dominant_edge_type, list(self.edge_models.values())[0]
        )
        return torch.sigmoid(model(u))

    def causal_effect(self, source_node: int) -> float:
        """
        計算 source_node → target_node 的因果效應（CE）。

        CE(u→v) = P(ŷ_v | do(h_u = observed)) - P(ŷ_v | do(h_u = 0.5))

        這是 do-calculus 的直接介入定義：
        有介入（觀測值）vs 無介入（基準值）的預測差異。
        """
        all_labels = {
            v: float(self.graph.node_type.get(v, 0.0 ))
            for v in self.graph.v
        }
        # 有介入：source_node 保持觀測值
        with torch.no_grad():
            p_with = float(self.forward(all_labels, intervene_nodes=set()).item())
        # 無介入：source_node 截斷為基準值 0.5
        with torch.no_grad():
            p_without = float(
                self.forward(all_labels, intervene_nodes={source_node}).item()
            )
        return max(0.0, p_with - p_without)
```

#### 5.4.3 Asymmetric Causal Shapley Value 計算

**為什麼選 Asymmetric 而非 Symmetric？**

標準（對稱）Causal Shapley 對所有父節點的排列順序一視同仁，每種組合的貢獻被平等加權。這在一般解釋任務中是公平的，但在**根因追溯**場景中存在語義錯誤：時間順序應當影響因果歸因的大小——越早進入資金鏈的節點（越接近根因）越應得到更高的歸因權重。Asymmetric Causal Shapley 正是解決這個問題的升級版本。

**Asymmetric Causal Shapley 的核心思路**（Frye et al., NeurIPS 2020）：

只考慮**符合因果排序（causal ordering）的聯盟子集**，跳過違反時序先行性的聯盟。具體而言，若父節點集合的拓撲順序為 $\text{Pa}(v) = \{u_1, u_2, \ldots, u_n\}$（按時間戳由早到晚排序），則只允許「包含 $u_i$ 時也必須包含所有 $u_j,\; j < i$」的聯盟進入計算：

$$\phi_i^{asym} = \sum_{S \in \mathcal{S}_i} w(S) \left[ v(S \cup \{i\}) - v(S) \right]$$

其中 $\mathcal{S}_i$ 是符合拓撲排序約束的聯盟集合（排除所有讓 $u_i$ 出現但其時序前驅缺席的聯盟），$w(S)$ 是對應的 Shapley 權重。

**聯盟函數沿用介入期望（do-calculus）**：

$$v(S) = \mathbb{E}\left[f(u) \;\middle|\; do(X_S = x_S,\; X_{V \setminus S} = \mathbf{0.5})\right]$$

**與對稱 Causal Shapley 的關鍵差異及意義**：

| 面向 | 對稱 Causal Shapley | Asymmetric Causal Shapley（CI-RCT 採用）|
|------|-------------------|----------------------------------------|
| 聯盟考慮範圍 | 所有 $2^n$ 個子集 | 只考慮符合時序排序的子集 |
| 歸因偏向 | 各層節點平均分配 | 自然偏向時序更早的根因節點 |
| 語義對應 | 平均邊際貢獻（與順序無關）| 時序感知的因果責任歸屬 |
| 根因追溯適合度 | 中（需額外 backward_score）| 高（歸因本身已偏向根因）|
| 計算複雜度 | $O(2^n)$（或 MC 近似）| $\leq O(2^n)$（排除部分聯盟，更快）|

**為什麼這對根因追溯語義更正確？**

Frye et al.（2020）指出：當因果關係具有明確時序時，人類傾向於把因果責任歸屬給最遠端的根因（distal cause），而非中間的傳遞節點（proximate cause）。Asymmetric Causal Shapley 的設計正好捕捉了這個直覺：它讓 $\phi^{asym}$ 值在時序最早的根因節點上最高，自然地為 `RootCauseTracer` 提供了語義一致的分數基礎，不需要額外引入 backward_score 的啟發式補償。

**實作**：

```python
def compute_asymmetric_causal_shapley(
    ncm: HeteroNCM,
    topo_order: list[int],
) -> dict[int, float]:
    """
    計算 Asymmetric Causal Shapley Value。

    Parameters
    ----------
    ncm        : 已訓練的 HeteroNCM
    topo_order : TypedCausalGraph.topological_order() 的結果
                 （時序由早到晚排列，index 越小 = 越接近根因）

    Returns
    -------
    phi_dict : {parent_node_id: phi_asymmetric_score}

    核心邏輯：
      只枚舉符合拓撲順序的「前綴聯盟」（prefix coalitions），
      即 S = {u_1, ..., u_k}（前 k 個節點）的集合，
      跳過所有違反時序前驅關係的聯盟。
    """
    parents = ncm.parents
    n = len(parents)
    if n == 0:
        return {}

    # 按拓撲排序重新排列父節點（最早的排最前）
    topo_idx = {v: i for i, v in enumerate(topo_order)}
    parents_sorted = sorted(
        parents,
        key=lambda p: topo_idx.get(p, float('inf'))
    )

    phi_dict = {p: 0.0 for p in parents_sorted}
    u_obs = ncm._u.detach().clone()

    # 只枚舉前綴子集（符合時序因果排序的聯盟）
    # S = {parents_sorted[0..k-1]}，對第 k 個節點計算邊際貢獻
    for k in range(n):
        target_pa = parents_sorted[k]
        # S = 前 k 個節點（已知排在 target_pa 之前）
        S = set(parents_sorted[:k])

        # v(S ∪ {target_pa})：S 和 target_pa 都在聯盟中
        mask_with = [p in S or p == target_pa for p in parents_sorted]
        # v(S)：只有 S 在聯盟中，target_pa 不在
        mask_without = [p in S for p in parents_sorted]

        from alg1 import _causal_value_fn
        v_with    = _causal_value_fn(ncm.model, u_obs, mask_with)
        v_without = _causal_value_fn(ncm.model, u_obs, mask_without)

        # 前綴聯盟的 Shapley 權重 = 1/n（均等分配 n 個前綴位置）
        phi_dict[target_pa] = (v_with - v_without) / n

    return phi_dict
```

**當 $n > 8$ 時的近似策略**：

前綴聯盟的總數為 $n$（線性），遠少於對稱 Shapley 的 $2^n$。因此 Asymmetric Causal Shapley 在計算效率上比對稱版本更優，不需要 Monte Carlo 近似即可精確計算。這是額外的實作優勢。

### 5.5 Module 3：RootCauseTracer

**設計哲學**：RootCauseTracer 是 CI-RCT 全新設計的模組，在現有任何因果 GNN 解釋器中均無對應設計。其核心思路是：詐騙行為在圖上存在一條「因果信號傳遞路徑」，從根因節點（資金源頭）沿有向邊傳遞到最終詐騙節點。RootCauseTracer 透過沿最強因果效應方向反向追溯，復原這條路徑。

```python
class RootCauseTracer:
    """
    基於因果效應分數的反向根因追溯器。

    演算法：從目標詐騙節點出發，每一步選擇因果效應最強的父節點，
    直到到達圖的源頭（無父節點）或因果效應低於閾值。

    輸出：根因節點、完整因果鏈、每跳的因果效應分數
    """

    def __init__(self, graph: TypedCausalGraph):
        self.graph = graph

    def trace(
        self,
        target_node: int,
        causal_effects: dict,
        max_hops: int = 5,
        threshold: float = 0.1,
    ) -> tuple[int, list, list]:
        """
        Parameters
        ----------
        target_node    : 被偵測為詐騙的目標節點
        causal_effects : {(src, dst): CE_score}，HeteroNCM 計算的因果效應
        max_hops       : 最大追溯深度 K（停止條件之一）
        threshold      : 因果效應最低閾值 ε（停止條件之二）

        Returns
        -------
        root_cause : 根因節點 id
        chain      : 完整路徑 [target_node, v_{k-1}, ..., root_cause]
        scores     : 每一跳的 CE score 列表
        """
        chain = [target_node]
        scores = []
        current = target_node
        visited = {target_node}

        for _ in range(max_hops):
            parents = list(self.graph.parents(current))

            # 停止條件 1：無父節點（已達圖的源頭）
            if not parents:
                break

            # 選擇因果效應最強的父節點
            best_pa = max(
                parents,
                key=lambda u: causal_effects.get((u, current), 0.0)
            )
            best_ce = causal_effects.get((best_pa, current), 0.0)

            # 停止條件 2：因果效應低於閾值
            if best_ce < threshold:
                break

            # 停止條件 3：防止環路
            if best_pa in visited:
                break

            chain.append(best_pa)
            scores.append(best_ce)
            visited.add(best_pa)
            current = best_pa

        root_cause = chain[-1]
        return root_cause, chain, scores

    def causal_chain_score(self, scores: list) -> float:
        """
        計算整條因果鏈的綜合可信度分數。
        使用幾何平均確保路徑上任何弱環節都會降低整體信心。
        """
        if not scores:
            return 0.0
        import math
        log_sum = sum(math.log(max(s, 1e-9)) for s in scores)
        return math.exp(log_sum / len(scores))
```

**停止條件的形式化定義**：

追溯在以下任一條件成立時停止：

| 條件 | 形式化 | 對應情境 |
|------|--------|---------|
| 圖的源頭 | $\|\text{Pa}(v)\| = 0$ | 當前節點無上游原因，已到資金起點 |
| 弱因果訊號 | $\max_{u \in \text{Pa}(v)} CE(u \rightarrow v) < \varepsilon$ | 上游因果連結不顯著，追溯不可靠 |
| 深度上限 | $\text{hop} \geq K$ | 防止在大圖中無限追溯 |
| 環路防護 | $v' \in \text{visited}$ | 防止在含有向環的圖中陷入迴圈 |

### 5.6 Module 4：CausalAdversarialGAN

這是 CI-RCT 解決 Gap 2（類別不平衡）和 Gap 3（偽裝行為）的核心機制，也是與現有所有因果 GNN 解釋器最大的架構差異點。

#### 5.6.1 設計動機與創新點

**現有對抗訓練方法的問題**：CARE-GNN、DAGCN 等對抗訓練方法中，偽裝節點的生成沒有因果結構約束，生成的偽裝邊可能違反圖的 DAG 拓撲排序（即製造時序上不合理的「逆時序資金流」），這不符合真實詐騙者的行為模式。

**CI-RCT 的創新**：Generator 在生成偽裝節點的連結時，必須滿足 TypedCausalGraph 的拓撲排序約束，確保生成的偽裝資金流向在時序上是「合法的」（因在時間上先於果）。這一設計同時有 Granger 時序先行性的理論支撐。

#### 5.6.2 對抗訓練的語義映射

```
真實世界：
  詐騙者（攻擊者）← → 偵測器（防禦者）

CI-RCT 中的對應：
  Generator（模擬詐騙者偽裝行為）← → Discriminator = HeteroGNN（偵測器）
```

這種語義映射使 GAN 的選擇不僅是技術決策，更是問題建模的自然反映。

#### 5.6.3 Generator 設計

```python
class CausalAdversarialGenerator(nn.Module):
    """
    因果約束的偽裝詐騙節點生成器。

    生成目標：
      特徵層面：接近正常節點（讓 GNN 難以區分）
      結構層面：連結符合 DAG 拓撲排序（不違反時序先行性）

    約束實現：
      只允許偽裝節點連接到拓撲排序中位置更前（時間更早）的節點，
      確保生成的資金流向在因果上是合法的。
    """
    def __init__(self, node_feature_dim: int, type_emb_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        # 特徵生成器：以真實詐騙節點特徵為條件，生成偽裝特徵
        self.feature_gen = nn.Sequential(
            nn.Linear(node_feature_dim + type_emb_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, node_feature_dim),
            nn.Tanh(),  # 歸一化輸出到 [-1, 1]
        )
        # 邊生成器：預測偽裝節點應連接到哪些節點
        self.edge_gen = nn.Sequential(
            nn.Linear(node_feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, fraud_features, type_emb, topo_order: list,
                noise_std: float = 0.05):
        """
        Parameters
        ----------
        fraud_features : 已知詐騙節點特徵（生成條件）
        type_emb       : 節點類型 embedding
        topo_order     : TypedCausalGraph 的拓撲排序列表
        noise_std      : 注入高斯噪聲增加生成多樣性

        Returns
        -------
        fake_features : 偽裝節點特徵（接近正常，特徵偽裝）
        edge_probs    : 與各合法上游節點的連結機率（結構約束）
        """
        # 加入噪聲增加多樣性（小量，避免破壞偽裝語義）
        z = torch.randn_like(fraud_features) * noise_std
        condition = torch.cat([fraud_features + z, type_emb], dim=-1)
        fake_features = self.feature_gen(condition)

        # 邊生成：只考慮拓撲排序中位置更前的合法上游節點
        # 這對應「偽裝者只能從時間更早的節點收款」的因果約束
        valid_upstream_count = max(1, len(topo_order) // 2)
        valid_upstream = topo_order[:valid_upstream_count]

        edge_probs = {}
        for upstream_node_idx in valid_upstream:
            upstream_feat = fraud_features  # 簡化：用詐騙節點特徵代表上游
            pair_feat = torch.cat([fake_features, upstream_feat], dim=-1)
            edge_probs[upstream_node_idx] = self.edge_gen(pair_feat)

        return fake_features, edge_probs
```

### 5.7 聯合損失函數與訓練策略

#### 5.7.1 三項損失函數

$$\mathcal{L}_{total} = \mathcal{L}_{detection} + \lambda_1 \cdot \mathcal{L}_{adversarial} + \lambda_2 \cdot \mathcal{L}_{causal}$$

| 損失項 | 公式 | 理論依據 | 作用 |
|--------|------|---------|------|
| $\mathcal{L}_{detection}$ | $\text{BCE}(\hat{y}_v, y_v)$ | 標準節點分類目標 | 確保 HeteroGNN 正確偵測詐騙節點 |
| $\mathcal{L}_{adversarial}$ | WGAN-GP 目標 | Arjovsky et al. (ICML 2017) | Generator 與 Discriminator 的對抗博弈，提升偽裝辨識能力 |
| $\mathcal{L}_{causal}$ | $\|\phi_{t} - \phi_{t-1}\|_2^2$ | Li et al. (ICLR 2025) | 確保對抗訓練前後 Causal Shapley φ 值穩定，防止解釋崩潰 |

**採用 WGAN-GP 而非標準 GAN 的原因**：Wasserstein 距離提供更穩定的梯度訊號，避免訓練初期的梯度消失問題，特別適合異質圖這種高維複雜生成場景。

#### 5.7.2 訓練流程

```
每個 Training Step：
  Step 1：前向傳播
    x_dict, edge_index_dict → HeteroGNN → h_dict, pred_dict

  Step 2：因果計算
    h_dict → TypedCausalGraph → HeteroNCM → CE(u→v) → φ_v

  Step 3：GAN 對抗（每 n_critic 步更新一次 Generator）
    Generator → 生成偽裝節點（特徵偽裝 + DAG 拓撲約束邊）
    增強圖 = 原圖 + 偽裝節點
    Discriminator（= HeteroGNN）→ 識別增強圖中的詐騙節點

  Step 4：計算聯合損失
    L_detection：原圖上的節點分類損失
    L_adversarial：WGAN-GP 對抗損失
    L_causal：φ 穩定性損失（當前 φ vs. 上一步 φ 的差距）

  Step 5：反向傳播與參數更新
    optimizer_G.step()   （每 n_critic 步）
    optimizer_D.step()   （每步）
```

---

## 6. 理論基礎說明

### 6.1 Pearl do-calculus 與干預式因果

CI-RCT 的核心因果理論基礎是 Pearl（2009）的 Structural Causal Model（SCM）。**干預**操作 $do(X = x)$ 在因果圖中截斷節點 $X$ 的所有入邊，強制設定 $X = x$，從而消除混淆因子的影響。

這與基於相關性的方法（GNNExplainer 的 conditional expectation $P(Y | X = x)$）有根本差異：

$$P(Y \mid do(X = x)) \neq P(Y \mid X = x)$$

前者是**干預分佈**（因果），後者是**條件分佈**（統計相關），後者受混淆因子影響，無法識別真正的因果結構。CI-RCT 的 HeteroNCM 通過截斷父節點入邊實現 do-calculus，直接估計干預分佈。

### 6.2 Asymmetric Causal Shapley Value（Frye et al. & Heskes et al., NeurIPS 2020）

CI-RCT 採用 **Asymmetric Causal Shapley Value** 作為節點因果貢獻的核心量化指標，而非標準的對稱版本。這是一個有明確理論依據的設計選擇。

**對稱 vs 非對稱的本質差異**：

對稱 Causal Shapley 對所有 $2^n$ 個父節點子集一視同仁，回答「在平均所有可能排列下，節點 $i$ 的邊際因果貢獻是多少」。這在沒有時序資訊的場景中是公平的。然而，在時序資金鏈追溯場景中，**時間先後性是因果歸因的天然先驗**——根因節點（最早進入鏈條的節點）在語義上應承擔最大的因果責任。

Asymmetric Causal Shapley 的解方是：只考慮**符合 DAG 拓撲排序的聯盟**，跳過所有「子節點出現但時序前驅缺席」的不合理聯盟。這讓 $\phi^{asym}$ 值天然偏向時序最早的根因節點，與人類對因果責任歸屬的直覺一致（Frye et al. 稱此為 "bias towards distal/root causes"）。

**與 RootCauseTracer 的理論銜接**：

採用 Asymmetric Causal Shapley 後，`RootCauseTracer` 的追溯過程可以直接以 $\phi^{asym}$ 作為評分依據，不需要額外的 backward_score 啟發式補償，整個框架的理論一致性更強。

**計算複雜度優勢**：

前綴聯盟的總數為 $O(n)$（線性），遠低於對稱版本的 $O(2^n)$，因此 Asymmetric Causal Shapley 可以精確計算而不需要 Monte Carlo 近似，這是額外的實作優勢。

**聯盟函數沿用 do-calculus 介入期望**（保持理論嚴謹性）：

$$v(S) = \mathbb{E}\left[f(u) \;\middle|\; do(X_S = x_S,\; X_{V \setminus S} = \mathbf{0.5})\right]$$

這確保 $\phi^{asym}$ 衡量的是真實的因果貢獻，而非條件相關性，完全繼承 do-calculus 的理論保證。

### 6.3 Neural Causal Model 的理論支撐

NCM 是 SCM 的神經網路近似版本，由 Xia et al.（2021）提出。其核心性質是：在給定因果圖結構 $G$ 的約束下，NCM 能表示所有與 $G$ 相容的 SCM，且可透過反向傳播訓練學習因果效應估計。CI-RCT 的 HeteroNCM 在此基礎上加入了類型感知（type-aware）的約束，使不同類型邊的因果機制相互獨立，符合異質圖的語義結構。

### 6.4 Granger 時序先行性與 DAG 拓撲約束

CausalAdversarialGAN 中 Generator 的 DAG 拓撲排序約束，在理論上對應 Granger（1969）的時序先行性原則：**因（cause）在時間上先於果（effect）**。TypedCausalGraph 在加入有向邊時驗證時間戳的先後關係，確保整個圖結構符合時序因果的基本假設。Generator 在這個約束下生成偽裝節點的連結，模擬了真實詐騙者的行為模式——詐騙者可以偽裝特徵，但無法改變資金流動的時間方向。

---

## 7. 實驗規劃

### 7.1 資料集

| 資料集 | 節點類型 | 邊類型數 | 任務 | CI-RCT 的使用目的 |
|--------|---------|---------|------|----------------|
| **Elliptic++** | Transaction, Actor | 4 | Bitcoin 詐騙節點分類 | **主要驗證場景**：金融詐騙根因追溯 |
| **DBLP** | Author, Paper, Venue | 3 | 作者學術領域分類 | Dataset-agnostic 泛化驗證 |
| **ACM** | Paper, Subject, Author | 2 | 論文分類 | Dataset-agnostic 泛化驗證 |
| **IMDB** | Movie, Director, Actor | 2 | 電影類型分類 | Dataset-agnostic 泛化驗證 |

### 7.2 Baseline 方法（三個類別）

**類別 A：關聯性/反事實解釋器（比較基準）**
- GNNExplainer（NeurIPS 2019）
- PGExplainer（NeurIPS 2020）
- CF-GNNExplainer（AISTATS 2022）

**類別 B：因果性解釋器（強比較基準，共享理論工具）**
- CXGNN（ECCV 2024）：共享 NCM 理論工具，異質圖任務上的性能上界參考
- CaGE（TOIS 2025）：最接近的異質圖因果解釋工作

**類別 C：根因追溯 + 對抗訓練方法**
- REASON（KDD 2023）：圖上根因追溯代表性工作
- SAGE-FIN（xAI 2025）：Elliptic++ 上的唯一先行工作
- CARE-GNN（CIKM 2020）：抗偽裝 GNN 偵測代表作

### 7.3 評估指標（四個維度）

**維度 A：解釋品質**
- Explanation Accuracy（EA）：估計因果子圖與 ground-truth 的節點重疊比例
- Groundtruth Match Accuracy（GMA）：精確找到 ground-truth 的嚴格比例
- Explanation Recall（ER）：覆蓋 ground-truth 的完整性

**維度 B：根因追溯品質**
- Root Cause Precision（RCP）：追溯到的根因節點是否確為詐騙相關節點
- Causal Chain Validity（CCV）：因果路徑上有多少比例節點屬於詐騙相關
- Mean Tracing Depth（MTD）：平均追溯深度（衡量框架找到真正源頭的能力）

**維度 C：偵測效能**
- F1-Score（主要指標，對類別不平衡更穩健）
- AUC-ROC
- Recall（詐騙偵測中比 Precision 更重要）

**維度 D：魯棒性與解釋穩定性**
- F1 under camouflage（偽裝比例 10%、30%、50% 下的 F1）
- φ-Stability：$\text{Std}(\phi_i^{(t)} - \phi_i^{(t-1)})$（對抗訓練前後 Causal Shapley 穩定性）

### 7.4 消融實驗（Ablation Study）

| 變體 | 移除/替換內容 | 驗證目的 |
|------|-------------|---------|
| **CI-RCT（完整）** | — | 完整框架效能基準 |
| w/o TypedCausalGraph | 退化為無類型區分的同質 CausalGraph | 驗證類型感知 SCM 的必要性 |
| w/o Per-type NCM | 所有邊類型共用一個 NNModel | 驗證 per-edge-type 獨立模型的貢獻 |
| w/ Symmetric Shapley | 以對稱 Causal Shapley 替換 Asymmetric 版本 | **驗證時序偏向歸因對根因追溯的提升效果** |
| w/o RootCauseTracer | 只輸出因果子圖，不執行根因追溯 | 驗證追溯模組的獨立價值 |
| w/o GAN | 移除對抗訓練，只用 L_detection | 驗證 GAN 對魯棒性的實際提升 |
| w/o L_causal | 移除 Causal Consistency Loss | 驗證 φ 穩定性保證的必要性 |
| w/ Granger（替換） | 用 Granger 因果替換 do-calculus + NCM | 干預式 vs 相關式因果效果對比 |
| w/ CVAE（替換 GAN） | 用 CVAE 替換 GAN | 驗證 GAN 對抗博弈結構的優勢 |

---

## 8. 論文章節結構

```
第一章 緒論
  1.1 研究背景（GNN 詐騙偵測的現實需求）
  1.2 研究動機（三個 Gap 的提出與連結）
  1.3 研究問題與目標（RQ1–RQ4）
  1.4 研究貢獻（五點原創貢獻聲明）
  1.5 本論文架構

第二章 文獻探討
  2.1 因果推斷理論基礎
      2.1.1 Pearl SCM 與 do-calculus
      2.1.2 Neural Causal Model（NCM）
      2.1.3 Symmetric vs Asymmetric Causal Shapley Value：理論比較與選擇依據
  2.2 GNN 可解釋性方法回顧
      2.2.1 關聯性解釋器（GNNExplainer、PGExplainer）
      2.2.2 反事實解釋器（CF-GNNExplainer、CLEAR）
      2.2.3 因果性解釋器（CXGNN、OrphicX、Gem、RC-Explainer）
  2.3 異質圖神經網路與可解釋性
      2.3.1 異質 GNN 架構（HGT、HAN、R-GCN）
      2.3.2 異質圖解釋器（xPath、CF-HGExplainer、CaGE、HGExplainer）
  2.4 圖結構上的根因分析
      2.4.1 AIOps 場景（REASON、CORAL、RCD）
      2.4.2 工業診斷場景（CIGNN、CHASE）
  2.5 因果方法用於 GNN 詐騙偵測
      2.5.1 CaT-GNN、Causal-DHG
      2.5.2 SAGE-FIN（Elliptic++ 先行工作，指出 Granger 因果的局限）
  2.6 GNN 偽裝行為與對抗魯棒性
      2.6.1 偽裝問題分析（CARE-GNN、PC-GNN）
      2.6.2 對抗訓練方法（DAGCN、GLSGNN）
  2.7 研究空白綜合分析與 CI-RCT 的定位

第三章 研究方法（CI-RCT 框架）
  3.1 問題形式化定義
      3.1.1 有向異質圖的符號定義 G = (V, E, τ_v, τ_e, T)
      3.1.2 因果根因追溯任務的形式化定義
      3.1.3 Causal Shapley Value 在本任務的形式化
  3.2 框架設計哲學與整體概覽
  3.3 Module 1：HeteroGNN Backbone（HGT）
      3.3.1 HGT 的 per-relation 注意力機制
      3.3.2 Dataset-agnostic 介面設計
  3.4 Module 2：Causal Intervention Engine
      3.4.1 TypedCausalGraph：類型感知 SCM 建構
      3.4.2 HeteroNCM：Per-edge-type 因果效應估計
      3.4.3 do-calculus 介入操作的實作
      3.4.4 Asymmetric Causal Shapley Value 計算（含時序排序約束與複雜度分析）
  3.5 Module 3：RootCauseTracer
      3.5.1 反向因果追溯演算法設計
      3.5.2 停止條件的形式化定義與選擇依據
      3.5.3 因果鏈綜合可信度分數
  3.6 Module 4：CausalAdversarialGAN
      3.6.1 設計動機：為何選擇 GAN 而非 CVAE 或 Diffusion
      3.6.2 Generator 的 DAG 拓撲約束設計
      3.6.3 WGAN-GP 訓練目標的選擇依據
      3.6.4 完整對抗訓練流程
  3.7 聯合損失函數 L_total
      3.7.1 三項損失的定義與理論依據
      3.7.2 Causal Consistency Loss 的必要性：基於 Li et al. (ICLR 2025)
      3.7.3 超參數 λ₁、λ₂ 的選擇策略

第四章 實驗設計與結果
  4.1 資料集說明（Elliptic++、DBLP、ACM、IMDB）
  4.2 實驗設置
      4.2.1 Baseline 方法清單與設定
      4.2.2 評估指標定義（四個維度）
      4.2.3 超參數設定
  4.3 主要實驗結果（Main Results）
      4.3.1 解釋品質比較（vs GNNExplainer、CXGNN、CaGE 等）
      4.3.2 根因追溯品質比較（vs REASON、SAGE-FIN）
      4.3.3 詐騙偵測效能比較（vs CARE-GNN、DAGCN）
      4.3.4 魯棒性與 φ 穩定性比較
  4.4 消融實驗（Ablation Study，9 種變體）
  4.5 根因追溯案例分析（Case Study）
      4.5.1 Bitcoin 詐騙資金鏈視覺化（3–5 個真實案例）
      4.5.2 偽裝詐騙節點識別案例分析
      4.5.3 因果路徑 vs 關聯性解釋子圖的視覺化對比
  4.6 Dataset-agnostic 泛化驗證（DBLP、ACM、IMDB）
  4.7 超參數敏感性分析（λ₁、λ₂、max_hops、threshold）
  4.8 計算效率分析（訓練時間、推論延遲）

第五章 討論
  5.1 研究發現與理論意義
  5.2 研究限制
      5.2.1 有向 DAG 假設在含環圖結構中的適用性
      5.2.2 GAN 訓練穩定性的實作挑戰
      5.2.3 Asymmetric Causal Shapley 的前綴聯盟假設在非線性時序圖中的適用性邊界
  5.3 未來工作方向
      5.3.1 擴展到動態異質圖（Temporal Heterogeneous Graph）
      5.3.2 整合 LLM 生成自然語言因果路徑解釋
      5.3.3 聯邦學習場景下的因果根因追溯

第六章 結論

參考文獻

附錄
  附錄 A：Elliptic++ 資料集詳細說明
  附錄 B：超參數設定完整表格
  附錄 C：完整實驗數據表格（含標準差）
  附錄 D：Asymmetric Causal Shapley 的前綴聯盟推導與對稱版本的理論比較
  附錄 E：TypedCausalGraph 的形式化定義與性質證明
```

---

## 9. 開發路線圖

> **開發策略**：CI-RCT 是從頭設計的獨立框架，不 fork 任何現有 repo。開發從理論驗證（小型合成圖）開始，逐步擴展到真實異質圖資料集。

### Phase 1：理論基礎與環境建置（Week 1–2）

- [ ] 閱讀並理解 Pearl SCM、NCM（Xia et al. 2021）、Symmetric Causal Shapley（Heskes et al. 2020）、Asymmetric Causal Shapley（Frye et al. 2020）的完整理論推導，釐清對稱與非對稱版本的差異
- [ ] 參考 CXGNN 原始碼理解 NCM 的 PyTorch 實作模式（作為理論參考，非 fork 對象）
- [ ] 建立 Python 開發環境：PyTorch、PyG（torch_geometric）、NetworkX
- [ ] 設計 CI-RCT 的模組介面規格（TypedCausalGraph、HeteroNCM、RootCauseTracer、CausalAdversarialGAN 的 API 文件）

### Phase 2：TypedCausalGraph 實作與驗證（Week 3–4）

- [ ] 實作 `TypedCausalGraph` 類別（有向邊、類型記錄、時間戳驗證）
- [ ] 實作 `add_edge()`、`parents()`、`topological_order()` 方法
- [ ] 撰寫 unit test：在手動建立的小型異質圖上驗證 DAG 結構、類型資訊正確性
- [ ] 驗證拓撲排序（Kahn's algorithm）的正確性與時序先行性的過濾邏輯

### Phase 3：HeteroNCM 實作與因果效應驗證（Week 5–7）

- [ ] 實作 `HeteroNCM` 類別（type embedding、per-edge-type NNModel）
- [ ] 實作 `forward()` 方法（支援 do-calculus 介入的前向傳播）
- [ ] 實作 `causal_effect()` 方法（有介入 vs 無介入的預測差異）
- [ ] 實作 `compute_asymmetric_causal_shapley()`（前綴聯盟枚舉，$O(n)$ 複雜度，無需 Monte Carlo 近似）
- [ ] 實作 `compute_symmetric_causal_shapley()`（對照用，供消融實驗 w/ Symmetric Shapley 變體）
- [ ] 在 ba_house 合成異質圖上驗證：Asymmetric 版本的 φ 值是否確實對時序最早的節點（根因）最高

### Phase 4：RootCauseTracer 實作與追溯驗證（Week 8–9）

- [ ] 實作 `RootCauseTracer` 類別與 `trace()` 方法
- [ ] 設計並實作四個停止條件
- [ ] 實作 `causal_chain_score()`（幾何平均）
- [ ] 在合成詐騙鏈資料上驗證追溯正確率（能否找到預設的根因節點）
- [ ] 撰寫視覺化工具：輸出帶因果效應分數標注的路徑圖

### Phase 5：HeteroGNN Backbone 實作（Week 10）

- [ ] 實作 `HeteroGNNBackbone` 類別（HGTConv + per-type classifier）
- [ ] 確認 HeteroData 輸入格式的 dataset-agnostic 相容性
- [ ] 在 DBLP 小型資料集上驗證節點分類基礎效能（F1 > 0.7 作為合格線）

### Phase 6：CausalAdversarialGAN 實作（Week 11–12）

- [ ] 實作 `CausalAdversarialGenerator`（特徵生成器 + 邊生成器）
- [ ] 實作 DAG 拓撲排序約束的邊生成邏輯
- [ ] 實作 WGAN-GP 訓練目標（梯度懲罰項）
- [ ] 整合 $\mathcal{L}_{causal}$（Causal Consistency Loss）
- [ ] 建立 `L_total = L_detection + λ₁ · L_adversarial + λ₂ · L_causal` 聯合訓練流程
- [ ] 驗證生成節點的特徵分佈（t-SNE 可視化：偽裝節點應接近正常節點分佈）

### Phase 7：Elliptic++ 整合與主要實驗（Week 13–15）

- [ ] 下載 Elliptic++ 資料集，撰寫資料預處理腳本（轉換為 HeteroData 格式）
- [ ] 確認 Transaction、Actor 兩種節點類型和四種邊類型的正確對應
- [ ] 跑通完整 CI-RCT pipeline（四個模組端到端整合）
- [ ] 計算四個維度的評估指標（EA、GMA、RCP、CCV、F1、φ-Stability）
- [ ] 跑通所有 Baseline 方法取得對比數據

### Phase 8：消融實驗、泛化驗證、Case Study（Week 16–17）

- [ ] 執行完整消融實驗（8 種變體）
- [ ] 在 DBLP、ACM、IMDB 驗證 dataset-agnostic 泛化能力
- [ ] 選取 3–5 個 Bitcoin 詐騙真實案例，製作因果路徑視覺化圖（供論文 Figure 用）
- [ ] 超參數敏感性分析（λ₁、λ₂ 掃描範圍、max_hops 1–8、threshold 0.05–0.3）
- [ ] 計算效率測試（訓練時間 vs CXGNN、REASON、SAGE-FIN）

### Phase 9：論文撰寫（Week 18–22）

- [ ] 第三章（方法）初稿，包含所有數學推導
- [ ] 第四章（實驗）初稿，包含所有表格與圖
- [ ] 第二章（文獻探討）
- [ ] 第一章（緒論）+ 第五/六章（討論/結論）
- [ ] 指導教授 Review 第一輪 + 修改
- [ ] 英文摘要撰寫（Abstract）
- [ ] 最終定稿

---

## 10. 口試問答預備

### Q1：CI-RCT 和 CXGNN 到底是什麼關係？你是不是只是在 extend CXGNN？

> CI-RCT 和 CXGNN 共享部分理論工具基礎（Pearl SCM、NCM），但在三個關鍵點上有本質差異。第一，CXGNN 針對同質圖的圖分類解釋任務；CI-RCT 針對異質圖的節點級根因追溯任務。第二，CXGNN 使用對稱 Causal Shapley；CI-RCT 採用 **Asymmetric Causal Shapley**（Frye et al., NeurIPS 2020），利用 TypedCausalGraph 的 DAG 拓撲排序讓 φ 值天然偏向根因節點，這在理論上更符合根因追溯的語義，且計算複雜度從 $O(2^n)$ 降至 $O(n)$。第三，CI-RCT 新增了 RootCauseTracer 和 CausalAdversarialGAN 兩個完全原創的模組。CXGNN 在我的論文中是 Related Work 和 Baseline 比較對象，不出現在 Method 章節。

### Q2：為什麼用 Pearl SCM + NCM，而不是 attention 或 gradient 作為因果代理？

> Attention 和 gradient 衡量的是統計相關性，不是因果效應。如果詐騙行為和某個時間特徵恰好相關（例如集中在特定時段），gradient 方法會誤將時間特徵識別為根因，這是 Spurious Correlation 問題。Pearl SCM 的 do-calculus 透過截斷入邊的介入操作，能排除混淆因子的影響，計算出真正的因果貢獻。這不是技術細節上的改進，而是因果推斷能力的根本性提升。

### Q3：你的方法和 SAGE-FIN 都在 Elliptic++ 上做因果解釋，差在哪裡？

> SAGE-FIN 使用 Granger 因果，其作者在論文中明確承認這是相關性方法，存在無法排除混淆因子的根本限制，並建議升級到 Pearl SCM。CI-RCT 正是對這個建議的直接回應，使用 do-calculus 的干預式因果計算。此外，SAGE-FIN 只做節點分類解釋，無根因追溯；CI-RCT 提供完整的資金鏈路徑輸出，這對執法單位更有實用價值。

### Q4：為什麼用 GAN 而不是 CVAE 或 Diffusion 來解決類別不平衡和偽裝問題？

> 這是問題建模層面的選擇，不只是技術比較。金融詐騙場景本質上是一個「攻防博弈」：詐騙者持續適應偵測策略，偵測器持續升級辨識能力。GAN 的 Generator-Discriminator 極小極大博弈天然對應這個現實情境。CVAE 最大化 ELBO，Diffusion 做去噪重建，兩者都是概率生成，無法捕捉這種持續對抗的動態。此外，我使用 WGAN-GP 而非標準 GAN，可以有效避免訓練不穩定的問題。

### Q5：L_causal 這一項損失函數的設計根據是什麼？

> 動機來自 Li et al.（ICLR 2025）的發現：GNN 解釋器對圖擾動攻擊非常脆弱，攻擊者只需微小擾動就能讓解釋器輸出完全不同的解釋邊，而預測結果不變。在 CI-RCT 中，對抗訓練加入偽裝節點本質上是一種隱性的圖結構擾動。$\mathcal{L}_{causal} = \|\phi_t - \phi_{t-1}\|_2^2$ 確保在這種擾動下，Causal Shapley φ 值的輸出保持穩定，防止因果解釋崩潰。這讓 CI-RCT 能同時提供「魯棒的偵測」和「穩定的解釋」，而不是在兩者間取捨。

### Q6：DAG 假設在真實金融圖中是否成立？

> 真實的金融交易圖在時序上確實是有向無環的：一筆資金從 A 流向 B 之後，無法在同一時序路徑上倒流回 A。TypedCausalGraph 在建圖時加入時間戳驗證（新增邊時確認 ts(src) < ts(dst)），能自動過濾違反時序先行性的邊，確保最終圖結構符合 DAG 假設。如果資料中存在真正的環路（例如迴圈洗錢），我在論文的研究限制章節會明確說明這種情況下的適用性邊界。

### Q7：為什麼選 Asymmetric Causal Shapley 而不是標準的（對稱）Causal Shapley？

> 這是一個有明確理論動機的選擇，不是隨意的技術替換。標準的對稱 Causal Shapley 對所有父節點的排列順序一視同仁，每種聯盟組合被平等加權。這在沒有時序資訊的場景中是公平的，但在根因追溯場景中存在語義問題：資金鏈的時間順序應當影響因果歸因的大小，越早進入鏈條的根因節點越應承擔更大的因果責任。Frye et al.（NeurIPS 2020）稱此為「偏向遠端根因（bias towards distal/root causes）」，正好符合我的追溯目標。此外，Asymmetric Causal Shapley 只枚舉符合拓撲排序的前綴聯盟，計算複雜度從 $O(2^n)$ 降至 $O(n)$，可以精確計算而不需要 Monte Carlo 近似，這是額外的工程優勢。消融實驗中的「w/ Symmetric Shapley」變體會直接驗證這個選擇對根因追溯精度的實際影響。

---

## 11. 參考文獻核心清單

### 因果推斷理論基礎（CI-RCT 的數學根基）

- **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- **Heskes, T., et al.** (2020). Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models. *NeurIPS 2020*. ← 對稱 Causal Shapley 的理論基礎
- **Frye, C., Rowat, C., & Feige, I.** (2020). Asymmetric Shapley Values: Incorporating Causal Knowledge into Model-Agnostic Explainability. *NeurIPS 2020*. ← **CI-RCT 採用的 Asymmetric Causal Shapley 原始論文**
- **Xia, R., et al.** (2021). Neural Causal Models for Counterfactual Identification and Estimation. *ICLR 2022* (NeurIPS 2021 Workshop version).
- **Granger, C.W.J.** (1969). Investigating Causal Relations by Econometric Models and Cross-Spectral Methods. *Econometrica*.

### 因果 GNN 解釋器（Related Work，含 CXGNN 作為 Baseline）

- **CXGNN**: Behnam, A., & Wang, B. (2024). Graph Neural Network Causal Explanation via Neural Causal Models. *ECCV 2024*. ← Baseline 比較對象
- **Gem**: Lin, W., et al. (2021). Generative Causal Explanations for Graph Neural Networks. *ICML 2021*.
- **OrphicX**: Lin, W., et al. (2022). OrphicX: A Causality-Inspired Latent Variable Model for Interpreting GNNs. *CVPR 2022*.
- **RC-Explainer**: Wang, X., et al. (2022). Reinforced Causal Explainer for GNNs. *IEEE TPAMI 2023*.
- **XGNNCert**: (2025). Certifiably Robust GNN Explainer. *ICLR 2025*. ← L_causal 設計依據

### 異質圖 GNN（HGT Backbone 依據）

- **HGT**: Hu, W., et al. (2020). Heterogeneous Graph Transformer. *KDD 2020*. ← Module 1 設計基礎
- **xPath**: Li, X., et al. (2023). Towards Fine-Grained Explainability for Heterogeneous GNN. *AAAI 2023*.
- **CaGE**: (2025). CaGE: A Causality-inspired GNN Explainer for Recommender Systems. *ACM TOIS 2025*.

### 圖上根因分析

- **REASON**: Wang, D., et al. (2023). Hierarchical GNNs for Causal Discovery and Root Cause Localization. *KDD 2023*.
- **CORAL**: Wang, et al. (2023). *KDD 2023*.
- **RCD**: Ikram, A., et al. (2022). Root Cause Analysis of Failures in Microservices. *NeurIPS 2022*.
- **CHASE**: (2024). CHASE: A Causal Heterogeneous Graph Framework for Root Cause Analysis. *arXiv 2024*.

### 因果 GNN 詐騙偵測

- **CaT-GNN**: Duan, Y., et al. (2024). CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal GNNs. *arXiv 2024*.
- **SAGE-FIN**: Nguyen, L., et al. (2025). Detecting Fraud in Financial Networks: A Semi-Supervised GNN Approach with Granger-Causal Explanations. *xAI 2025*. ← 直接先行工作，指出 Granger 局限
- **xFraud**: Rao, Y., et al. (2022). xFraud: Explainable Fraud Transaction Detection. *VLDB 2022*.

### GNN 對抗魯棒性

- **CARE-GNN**: Dou, Y., et al. (2020). Enhancing GNN-based Fraud Detectors against Camouflaged Fraudsters. *CIKM 2020*.
- **DAGCN**: Wan, Z., et al. (2026). Dynamic Adversarial GNN for Real-Time Fraud Detection. *NPL 2026*.
- **GLSGNN**: (2024). Safeguarding Fraud Detection from Attacks. *IJCAI 2024*.
- **WGAN-GP**: Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs. *NeurIPS 2017*. ← GAN 穩定化依據

### 資料集

- **Elliptic++**: Elmougy, A., & Liu, L. (2023). Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network. *KDD 2023*.
- **DBLP / ACM / IMDB**: 標準異質圖 benchmark，廣泛用於 HAN、HGT 等論文的實驗評估。

---

*本文件為論文提案（v0.4），理論工具升級：以 Asymmetric Causal Shapley（Frye et al. & Heskes et al., NeurIPS 2020）取代對稱版本，充分利用 TypedCausalGraph 的 DAG 時序排序，使根因追溯的因果歸因語義更嚴謹，計算複雜度同時從 $O(2^n)$ 降至 $O(n)$。*

*Document maintained by: Tommy（淡江大學統計學研究所）*

*Version history:*
*v0.1（2026-03-23）初始規劃*
*v0.2（2026-04-01）加入 GAN 模組、口試問答、理論基礎章節*
*v0.3（2026-04-01）定位策略調整：移除「擴展 CXGNN」框架，改採中間路線——以成熟理論工具為基礎、從頭設計獨立框架*
*v0.4（2026-04-01）理論工具升級：採用 Asymmetric Causal Shapley 取代對稱版本；新增 Q7 口試問答；消融實驗加入 w/ Symmetric Shapley 變體（共 9 種）*