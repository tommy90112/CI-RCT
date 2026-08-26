# 語言設定
請永遠使用繁體中文回覆，不要使用其他任何語言。
All responses must be in Traditional Chinese (繁體中文).

## Communication Language
Always respond in Traditional Chinese (繁體中文).
Never switch to English, Simplified Chinese, or any other language,
even if the user writes in English.

# Session Handoff 規則

## 何時觸發
出現以下任一情況，主動用一句話提醒 user：
- 當前 session 訊息數已超過 50 則
- 完成一個明確的子任務
- User 說「告一段落」「先這樣」「明天繼續」

提醒要短：「這段告一段落了,建議寫 handoff 後 /clear」。
不要每則都提醒。

## 檔案位置
寫到 `.claude/handoff/<topic-slug>.md`。
topic-slug 用 kebab-case。
若檔案已存在,append 新段落在最下方,不覆蓋舊內容。

## 格式
每段包含:
- `## YYYY-MM-DD HH:MM` 標題
- **目前狀態**: 卡在哪 / 完成到哪
- **已確認 OK**: 驗證過的部分
- **已試過但失敗**: 避免重蹈覆轍
- **下一步**: 具體動作
- **關鍵檔案**: 路徑列表

## 接續 session
User 說「繼續 X」「讀 handoff」時:
- 只 read 對應檔案的**最後一段**
- 不要 read 整個檔案
- 讀完直接開始工作

## 寫完後
1. append 到檔案
2. 只回:「Handoff 已寫入 <path>,可以 /clear」
3. 不要繼續其他任務

---