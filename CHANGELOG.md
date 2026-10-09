# 變更紀錄

本檔案記錄 `room-geometry-inference` 的所有重要變更。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/)。

## [Unreleased]

### 新增

- 英文版 README（`README.md`，GitHub 預設顯示），繁體中文版在 `README.zh-TW.md`。
- 流程圖也有英文版（`docs/workflow.html`、`docs/images/workflow.png`），中文版是 `docs/workflow.zh-TW.html` 與 `docs/images/workflow.zh-TW.png`。

---

## [0.1.0] - 2026-10-09

### 新增

- 首次釋出。
- 矩形房間的鏡像源法（image-source method）脈衝響應模擬：以二維鏡像晶格直接模擬的平面房間（一階共 9 個到達）與三維房間、每軸或總反射次數兩種截斷方式、每面牆各自的反射係數、整數或窗化 sinc 分數延遲，聲速可由氣溫換算。
- 由激勵與麥克風兩聲道錄音辨識脈衝響應：逐樣本 NLMS、LMS（含穩定上限檢查）與 Levinson 遞迴的最小平方解，只保留最終權重，另提供學習曲線與正規化失準度（misalignment）。
- 一階回音反解房間大小與聲源距離：1680 種回音指派的閉式解、以角落反射檢查一致性的殘差、Levenberg–Marquardt 精修、32 種等價佈局去重，以及重合成比對排序。一階到達無法決定聲源方位角，只能假設；`bearing_family` 列出整個方位角解族。
- 回音擷取：最小間距的峰值偵測、依延遲補償 1/d 衰減、拋物線次樣本內插；四種敏感度掃描（到達時間偏移、聲源方位角、聲源位移、訊雜比），以表格與誤差棒圖輸出。
- 房間大小分類：能量衰減曲線、回音到達時間、IMF 能量與脈衝響應振幅四種特徵，SVM、加權 KNN、決策樹與可回答 `unknown` 的回音樣板比對，以房間分組的分層交叉驗證評估。
- 脈衝響應去噪比較：線性校正、PCA 投影、scikit-learn MLP 與 PyTorch 深度去噪自編碼器（`dl` extra，可用 GPU），以房間分組的交叉驗證回報每筆響應的 MSE（dB）。
- 經驗模態分解（EMD）：鏡射端點的三次樣條包絡、三種停止準則與迭代上限。
- typer CLI：`synthesize`、`index`、`prepare`、`evaluate`、`train`、`predict`、`simulate`、`invert`、`sensitivity`、`denoise`；每次執行都寫出 `config.yaml` 與 `manifest.json`（`index` 只寫 `dataset.csv`，已存在時需加 `--force` 才會覆寫）。
- 已知幾何的合成兩聲道量測資料集，以及五支範例腳本。
