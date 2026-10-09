# 參與開發

歡迎回報問題或送 pull request。

---

## 開發環境

需要 Python 3.10 以上。建議使用虛擬環境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 共用套件 speechdsp

單聲道 WAV 讀取、UAR 與 sensitivity／specificity 來自 `speechdsp`，安裝本專案時 pip 會自動從 PyPI 安裝它。

- PyPI：<https://pypi.org/project/speechdsp/>
- 原始碼：<https://github.com/recklight/SpeechDsp>

### 安裝本專案

```bash
pip install -e ".[dev]"
room-geometry-inference --help
```

需要 PyTorch 深度去噪自編碼器時再加裝 `pip install -e ".[dl]"`。

---

## 提交前的檢查

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

CI 會執行相同的檢查，並在 Linux 與 Windows 上跑 Python 3.10–3.13、最舊的相依版本，以及安裝 CPU 版 PyTorch 的測試。

### Python 3.10 相容性

套件宣告 `requires-python = ">=3.10"`，請避開 3.11 以後才有的語法與標準函式庫：

| 不要用 | 請用 |
| --- | --- |
| `tomllib` | 設定檔是 YAML，用不到 TOML |
| `typing.Self` | 類別名稱字串 |
| `enum.StrEnum` | `class Foo(str, Enum)` |
| `datetime.UTC` | `datetime.timezone.utc` |
| `except*`、`ExceptionGroup` | 一般的 `except` 與單一例外 |
| `asyncio.TaskGroup` | `asyncio.gather` |
| PEP 695（`class Foo[T]`、`type X = ...`） | `TypeVar` 與一般的型別註記 |
| `itertools.batched` | 明確的切片迴圈 |
| `np.float_`、`np.trapz`、`np.in1d`、`np.trapezoid`、`np.concat` 等 NumPy 2.0 才有或已移除的名稱 | `np.float64`、`scipy.integrate.trapezoid`、`np.isin`、`np.concatenate` |

任何 Python 版本都能執行的 3.10 語法閘門：

```bash
python -c "import ast,pathlib; [ast.parse(p.read_text(encoding='utf-8'), str(p), feature_version=(3, 10)) for p in pathlib.Path('.').rglob('*.py') if not any(x in p.parts for x in ('.venv', 'build', 'dist', '__pycache__'))]"
```

`tests/test_python310_compat.py` 會自動掃描上述語法、標準函式庫、NumPy 名稱與相依版本下限，並確認匯入套件時不會載入選用套件。

---

## 程式碼風格

- 每個模組開頭加 `from __future__ import annotations`。
- 公開函式都要有 type hints 與英文 numpy 風格 docstring；實作文獻方法的公開函式要有 `References` 段。
- 識別字與 docstring 一律用美式拼法（normalize、analyze、color）。
- 用 `logging` 輸出進度與警告，不要用 `print`。
- 陣列運算盡量向量化，不寫逐元素的迴圈。
- 所有參數都放在 `config.py` 的 dataclass，不要在程式裡寫死路徑或數值。

---

## 資料

repository 裡沒有任何量測資料，測試只用 `synthetic.py` 產生的訊號。請不要提交錄音、`.wav`、`.mat`、`.npy`、模型檔，或任何含個人資訊的檔案。

---

## 提交訊息

提交訊息用英文：第一行寫清楚改了什麼，需要時空一行，再用幾句話說明原因。有功能變動時，請同步更新 `CHANGELOG.md` 的 `[Unreleased]` 段。
