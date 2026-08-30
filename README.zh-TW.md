# CI-RCT

**基於因果干預的異質圖根因追溯框架**(Causal Intervention-based Root Cause Tracing)

English version: [README.md](README.md)

給定一筆被判定為可疑的交易,CI-RCT 沿著**時序尊重的金流有向無環圖反向回溯**,跨越交易與錢包兩種節點型別,追至實際掌控資金的來源實體,並輸出一條可供稽核的因果金流鏈——而非僅僅一個分數。

---

## 研究動機

圖神經網路在詐欺偵測上已有良好表現,但對於稽核人員接下來真正會問的問題——**這件事從哪裡開始?沿什麼路徑擴散?最終是誰造成的?**——現有方法的回答能力明顯不足。本研究由三個缺口出發:

| 缺口 | 問題所在 |
|------|----------|
| **異質性未被妥善建模** | 真實網絡由多種節點與多種連結構成。將它們視為等價,會抹除讓金流軌跡得以被讀懂的型別語義。 |
| **停留在相關性而非因果性** | GNNExplainer、PGExplainer 等方法找出的是與預測**統計上最相關**的子圖,而這與**造成**該預測的子圖並非同一件事。 |
| **缺乏由結果回溯源頭的能力** | 多數流程止於偵測,鮮少能從已觀察到的異常反向追蹤至其源頭,更少能跨越系統的多個層次。 |

CI-RCT 針對三者提出對應設計:型別感知的有向圖、以干預(Pearl do-calculus)而非關聯所估計的因果效應,以及一個能將「果」還原回「因」的追溯器。

## 與既有方法的差異

- **以干預估計,而非以關聯估計**
  逐邊的因果效應來自「切斷父邊後重新前向傳播」,而非梯度顯著性或學習得到的邊遮罩。

- **跨型別的根因歸屬**
  追溯所得的鏈由錢包與交易交替構成,因此一筆被標記的交易可以被歸因至實際掌控它的錢包。**這是僅輸出節點分數的偵測器在原理上無法提供的資訊。**

- **時序正確性由結構保證**
  邊依時間戳定向,因果圖會直接拒絕任何會讓「因晚於果」的邊。

- **歸因效度延伸至深鏈**
  逐段局部因果責任採用滾動讀出計算,使距離目標超過骨幹感受野的節點仍能獲得可量測的歸因值。

## 系統架構

四個模組串接。其中 GAN 僅在訓練階段參與,推論時不介入運算。

![CI-RCT 系統架構:時序異質圖輸入模組一(HeteroGNN 骨幹,HGTConv),其表徵驅動模組二(因果干預引擎——TypedCausalGraph、產生 CE 的 HeteroNCM,以及以聯盟 do-intervention 計算的非對稱因果 Shapley);模組三(根因追溯器)依帶號 CE 反向回溯輸出犯罪金流鏈;模組四(因果對抗生成網路)僅於訓練階段以 WGAN-GP 生成偽裝樣本。](CI-RCT/viz/ch3_architecture_v3.svg)

模組二輸出**兩個職責不同的平行訊號**:

- **CE**(逐邊因果效應)負責對上游候選排序,驅動追溯過程
- **φ**(非對稱因果 Shapley)負責量化各上游節點對其緊鄰下游交易所負的局部因果責任

簡言之:**CE 負責追溯,φ 負責解釋。**

## 評估設計

四個維度,均實作於 `evaluate.py`:

| 維度 | 所回答的問題 | Ground truth |
|------|-------------|--------------|
| **A・分類效能** | 偵測能力是否與同型方法相當? | 資料集標籤 |
| **B・根因追溯** | 追溯終點是否落在真實詐欺實體上? | 詐欺標註實體集合 |
| **C・解釋品質** | 金流鏈是否涵蓋真實的發起錢包? | LFPN 準則,含嚴格與 k 跳延伸兩種 |
| **D・歸因穩健性** | 歸因是否耐得住輸入擾動? | 對節點表徵施加高斯擾動的 σ 掃描 |

**維度 A 是前提檢核,而非本研究的貢獻所在**:唯有偵測能力未被犧牲,追溯與解釋的貢獻才不會被解讀為以偵測力換來的。

三個訓練變體共用相同的圖結構與因果解釋機制,差異僅在監督訊號的配置——`transaction`(以交易為監督目標)、`wallet`(以錢包為監督目標)、`joint`(交易為主、錢包為輔)。三者並列比較同時也是「監督目標如何影響各評估維度」的檢驗。

> 量化結果於論文中報告,此處尚未轉載。重現這些結果所需的一切均已包含在本 repository 中。

## 開始使用

### 環境需求

- Python 3.10 以上
- PyTorch 2.0+ 與 PyTorch Geometric 2.4+
- Node.js 18+(僅可解釋性檢視器需要)

### 安裝

```bash
pip install -r CI-RCT/requirements.txt
```

> **PyG 的擴充套件是唯一真正的安裝陷阱。**
> `torch-scatter` 與 `torch-sparse` 是針對特定 torch 版本編譯的。若版本不匹配,`import torch_geometric` 會直接 **segfault 而不是拋出例外**,極難除錯。請依照
> [PyG 官方說明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> 針對你的 torch 與 CUDA 版本安裝。`run_pipeline.py` 會在啟動任何長時間工作前先幫你檢查這一點。

### 資料集放置

本 repository 不轉散布 Elliptic++。請自行下載並依下列結構放置:

```
CI-RCT/data/Elliptic++/
  txs_features.csv       txs_classes.csv       txs_edgelist.csv
  wallets_features.csv   wallets_classes.csv
  AddrTx_edgelist.csv    TxAddr_edgelist.csv   AddrAddr_edgelist.csv
```

## 執行方式

整條流程一個指令即可跑完。它會依序串接訓練、評估、金流鏈匯出與檢視器建置,並自動將每個階段的產出接到下一階段。

```bash
cd CI-RCT

python run_pipeline.py --dry-run        # 顯示執行計畫與確切指令,不動任何東西
python run_pipeline.py --device cuda    # 完整執行:三個變體,最後建置檢視器
python run_pipeline.py --from evaluate  # 沿用既有的 checkpoint
python run_pipeline.py --force evaluate # 重跑某階段(以及其所有下游)
python run_pipeline.py --only frontend  # 只重建檢視器
```

**產物已存在的階段會自動跳過**,因此重複執行的成本很低。強制重跑某階段時,其下游也會一併重跑——否則你會得到「用舊模型算出的解釋」配上「剛重訓的模型」,而這種錯誤在畫面上完全看不出來。

產出位置即檢視器預期讀取的位置:

```
checkpoints/ci_rct_elliptic++[_變體]_best.pt
viz/crime_chains[_變體].json     追溯鏈 + 逐節點 φ + 特徵歸因
results/crime_chains.csv         一列一條鏈的扁平表
results/chain_neighbors.json     真實一階鄰居覆蓋層
frontend_temp/dist/              自帶資料的靜態檢視器
```

### 單獨執行各階段

```bash
cd CI-RCT

python train.py --dataset elliptic++ --variant joint --epochs 400 --use_gan true
python evaluate.py --dataset elliptic++ --checkpoint <路徑> --lfpn_mode both
python infer.py --dataset elliptic++ --checkpoint <路徑> --target_node <索引>
```

> **有兩個旗標刻意與 CLI 預設值不同,而且訓練與評估之間也彼此不同。**
> `run_pipeline.py` 會自動帶上;若你直接呼叫這些腳本,必須自行指定。
>
> | 旗標 | 訓練 | 評估 | 原因 |
> |------|------|------|------|
> | `--ncm_baseline` | `type_mean` | `marginal` | `marginal` **在訓練時會卡死**(需對約 80 萬個錢包做 per-step MLP),它只能用於評估階段。 |
> | `--tracer_score` | — | `ce_signed` | 與論文報告的設定一致;argparse 預設值為 `ce`。 |
> | `--shapley_topk` | — | 需設上限(如 `8`) | 每個聯盟都是一次完整的骨幹前向傳播。不設上限,φ 的匯出在實務上跑不完。 |

### 記憶體受限時的選項

| 旗標 | 效果 |
|------|------|
| `--subsample_tx 20000` | 保留全部詐欺交易 + 隨機抽樣的合法交易 |
| `--labeled_only true` | 僅保留帶標籤交易及其一階鄰居 |
| `--fraud_subgraph true` | 保留全部交易,但錢包限縮於帶標籤交易的 BFS 鄰域 |
| `--include_addr_addr true` | 納入錢包→錢包邊(論文設定所需,較耗記憶體) |

## 可解釋性檢視器

`frontend_temp/` 是以 React + Vite 實作的金流鏈檢視器,採三層呈現:

- **L1**——以圖呈現金流鏈,逐節點標示因果責任
- **L2**——沿鏈的逐節點貢獻長條圖
- **L3**——責任樞紐節點上的逐特徵因果歸因

它直接讀取 `viz/` 與 `results/` 底下的匯出檔(見 `frontend_temp/vite.config.ts`),因此重新產生資料後不需搬動檔案即可生效。

```bash
cd CI-RCT/frontend_temp
npm install
npm run dev       # 開發模式,資料重產後即時反映
npm run build     # 產出 dist/,資料一併打包
```

## 專案結構

```
CI-RCT/
  run_pipeline.py      一鍵流程入口
  pipeline/            階段圖、凍結設定、前置檢查
  train.py             訓練(Phase 1 無 GAN / Phase 2 WGAN-GP)
  evaluate.py          四維度評估與金流鏈匯出
  infer.py             單圖推論
  model/               四個模組;tracer_strategies/ 收錄各追溯演算法變體
  utils/               Elliptic++ 載入器、因果圖建構、評估指標、LFPN
  configs/config.py    凍結的 CI_RCT_Config dataclass——模型檔中不寫死任何超參數
  scripts/             消融實驗驅動、繪圖與匯出工具
  tests/               pytest 測試套件
  frontend_temp/       可解釋性檢視器
CXGNN/                 上游參考實作,作為相關工作基線
```

### 關於全域節點 ID

`TypedCausalGraph` 與 `RootCauseTracer` 一律以**全域 ID** 定址節點——將所有節點型別依排序後串接編號。使用 `compute_type_offsets(data)` 可將區域索引轉為全域索引。**在需要全域 ID 之處誤傳區域索引,是最常見的錯誤來源**,且症狀通常不是報錯而是結果詭異。

## 測試

```bash
cd CI-RCT
pytest tests/
pytest tests/ -v --cov=model --cov=utils --cov=pipeline
```

## 相關工作與出處聲明

`CXGNN/` 內含 **"Graph Neural Network Causal Explanation via Neural Causal Models"**(ECCV 2024,[arXiv:2407.09378](https://arxiv.org/pdf/2407.09378))的原作者參考實作,本研究將其作為相關工作基線使用。該目錄採 MIT 授權並附有自己的 `LICENSE` 檔案,**非本專案作者所撰寫**。

## 引用

<!-- TODO: 公開前請填入作者、論文題目、學校與年份。 -->

```bibtex
@mastersthesis{circt,
  title  = {{TODO: 論文題目}},
  author = {{TODO: 作者}},
  school = {{TODO: 學校}},
  year   = {{TODO}}
}
```

## 授權

<!-- TODO: 選定授權條款並在 repository 根目錄新增 LICENSE 檔案。
     若沒有 LICENSE,法律預設為「保留一切權利」,任何人都不得重用本程式碼。
     注意 CXGNN/ 目錄另採 MIT 授權。 -->

TODO。
