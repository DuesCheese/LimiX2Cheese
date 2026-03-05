import json
import logging
import os
import queue
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "torch": "torch",
    "sklearn": "scikit-learn",
    "huggingface_hub": "huggingface-hub",
}

OPTIONAL_PACKAGES = {
    "openpyxl": "openpyxl",
}


@dataclass
class RunConfig:
    task: str
    data_path: str
    predict_data_path: str
    model_size: str
    model_path: str
    target_columns: List[str]
    test_size: float
    random_state: int
    use_cpu: bool
    use_retrieval: bool
    local_output_dir: str


class TkQueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        self.q.put(self.format(record))


class LimiXGuiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LimiX2Cheese - 简易推理前端")
        self.root.geometry("1120x760")

        self.log_queue = queue.Queue()
        self.logger = logging.getLogger("limix_gui")
        self.logger.setLevel(logging.INFO)

        self.log_file_path = None
        self.worker_thread = None
        self.stop_requested = False
        self.project_root = Path(__file__).resolve().parent

        self.dataframe = None
        self.current_columns: List[str] = []

        self._build_ui()
        self._setup_logger()
        self._poll_log_queue()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="数据文件:").grid(row=0, column=0, sticky=tk.W)
        self.data_path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.data_path_var, width=85).grid(row=0, column=1, padx=5)
        ttk.Button(top, text="选择 CSV/XLSX", command=self.select_data_file).grid(row=0, column=2)

        ttk.Label(top, text="预测文件(可选):").grid(row=1, column=0, sticky=tk.W)
        self.predict_data_path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.predict_data_path_var, width=85).grid(row=1, column=1, padx=5)
        ttk.Button(top, text="选择预测文件", command=self.select_predict_file).grid(row=1, column=2)

        ttk.Label(top, text="任务:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.task_var = tk.StringVar(value="Classification")
        task_box = ttk.Combobox(top, textvariable=self.task_var, state="readonly", values=["Classification", "Regression", "Missing Value Imputation"], width=35)
        task_box.grid(row=2, column=1, sticky=tk.W)
        task_box.bind("<<ComboboxSelected>>", lambda _: self.update_target_visibility())

        ttk.Label(top, text="模型大小:").grid(row=2, column=1, sticky=tk.E)
        self.model_size_var = tk.StringVar(value="16M")
        ttk.Combobox(top, textvariable=self.model_size_var, state="readonly", values=["16M", "2M"], width=10).grid(row=2, column=2, sticky=tk.W)

        model_frame = ttk.LabelFrame(self.root, text="模型与运行配置", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(model_frame, text="模型文件(.ckpt，可选):").grid(row=0, column=0, sticky=tk.W)
        self.model_path_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=78).grid(row=0, column=1, padx=5)
        ttk.Button(model_frame, text="选择模型", command=self.select_model_file).grid(row=0, column=2)

        self.use_cpu_var = tk.BooleanVar(value=False)
        self.use_retrieval_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(model_frame, text="强制 CPU", variable=self.use_cpu_var).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(model_frame, text="开启 Retrieval (需较强显卡)", variable=self.use_retrieval_var).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(model_frame, text="测试集比例:").grid(row=2, column=0, sticky=tk.W)
        self.test_size_var = tk.StringVar(value="0.2")
        ttk.Entry(model_frame, textvariable=self.test_size_var, width=10).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(model_frame, text="随机种子:").grid(row=2, column=1, sticky=tk.E)
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(model_frame, textvariable=self.seed_var, width=10).grid(row=2, column=2, sticky=tk.W)

        ttk.Label(model_frame, text="输出目录:").grid(row=3, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value="./outputs")
        ttk.Entry(model_frame, textvariable=self.output_dir_var, width=78).grid(row=3, column=1, padx=5)
        ttk.Button(model_frame, text="选择目录", command=self.select_output_dir).grid(row=3, column=2)

        target_frame = ttk.LabelFrame(self.root, text="目标列选择（分类/回归）", padding=10)
        target_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

        self.target_hint = ttk.Label(target_frame, text="请选择需要预测的列。分类建议选 1 列，回归支持多列。")
        self.target_hint.pack(anchor=tk.W)

        list_frame = ttk.Frame(target_frame)
        list_frame.pack(fill=tk.BOTH, expand=False)
        self.target_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=8)
        self.target_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.target_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.target_listbox.config(yscrollcommand=scrollbar.set)

        actions = ttk.Frame(self.root, padding=10)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="1) 检测依赖", command=self.check_dependencies).pack(side=tk.LEFT)
        ttk.Button(actions, text="2) 自动安装缺失依赖", command=self.install_missing_dependencies).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="3) 开始推理", command=self.start_run).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(actions, mode="determinate", length=260)
        self.progress.pack(side=tk.LEFT, padx=10)

        self.status_var = tk.StringVar(value="状态: 就绪")
        ttk.Label(actions, textvariable=self.status_var).pack(side=tk.LEFT)

        log_frame = ttk.LabelFrame(self.root, text="运行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = tk.Text(log_frame, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _setup_logger(self):
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler = TkQueueHandler(self.log_queue)
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _set_progress(self, value: int, message: str = None):
        self.progress["value"] = value
        if message:
            self.status_var.set(f"状态: {message}")

    def _poll_log_queue(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.root.after(120, self._poll_log_queue)

    def log(self, msg: str):
        self.logger.info(msg)

    def select_data_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Tabular files", "*.csv *.xlsx *.xls"), ("All files", "*.*")])
        if file_path:
            self.data_path_var.set(file_path)
            self.load_columns(file_path)

    def select_model_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Checkpoint", "*.ckpt"), ("All files", "*.*")])
        if file_path:
            self.model_path_var.set(file_path)

    def select_predict_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Tabular files", "*.csv *.xlsx *.xls"), ("All files", "*.*")])
        if file_path:
            self.predict_data_path_var.set(file_path)

    def select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)

    def load_columns(self, file_path: str):
        try:
            import pandas as pd

            if file_path.lower().endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            self.dataframe = df
            self.current_columns = list(df.columns)
            self.target_listbox.delete(0, tk.END)
            for c in self.current_columns:
                self.target_listbox.insert(tk.END, c)
            self.log(f"已加载数据文件: {file_path}, 行数={len(df)}, 列数={len(df.columns)}")
        except Exception as exc:
            messagebox.showerror("读取失败", f"读取文件失败: {exc}")
            self.log(f"读取文件失败: {exc}")

    def update_target_visibility(self):
        task = self.task_var.get()
        if task == "Missing Value Imputation":
            self.target_hint.configure(text="缺失值插补任务不需要目标列。")
        elif task == "Classification":
            self.target_hint.configure(text="分类任务建议只选择 1 个目标列。")
        else:
            self.target_hint.configure(text="当前版本回归任务仅支持 1 个目标列。")

    def _missing_packages(self) -> Tuple[List[str], List[str]]:
        missing_required, missing_optional = [], []
        for mod, pip_name in REQUIRED_PACKAGES.items():
            try:
                __import__(mod)
            except Exception:
                missing_required.append(pip_name)
        for mod, pip_name in OPTIONAL_PACKAGES.items():
            try:
                __import__(mod)
            except Exception:
                missing_optional.append(pip_name)
        return missing_required, missing_optional

    def check_dependencies(self):
        missing_required, missing_optional = self._missing_packages()
        if not missing_required and not missing_optional:
            messagebox.showinfo("依赖检查", "依赖完整，可以运行。")
            self.log("依赖检查完成: 全部已安装")
            return

        msg = []
        if missing_required:
            msg.append("缺失必需依赖: " + ", ".join(missing_required))
        if missing_optional:
            msg.append("缺失可选依赖: " + ", ".join(missing_optional))
        text = "\n".join(msg)
        messagebox.showwarning("依赖检查结果", text)
        self.log(text)

    def install_missing_dependencies(self):
        missing_required, missing_optional = self._missing_packages()
        packages = list(dict.fromkeys(missing_required + missing_optional))
        if not packages:
            self.log("无需安装依赖")
            messagebox.showinfo("安装依赖", "没有缺失依赖。")
            return

        self.log(f"开始安装依赖: {packages}")
        cmd = [sys.executable, "-m", "pip", "install", *packages]
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.log(result.stdout)
            if result.returncode != 0:
                self.log(result.stderr)
                messagebox.showerror("安装失败", f"安装依赖失败，退出码={result.returncode}\n{result.stderr[:500]}")
            else:
                messagebox.showinfo("安装依赖", "依赖安装成功。")
        except Exception as exc:
            self.log(f"安装依赖异常: {exc}")
            messagebox.showerror("安装失败", str(exc))

    def get_selected_targets(self) -> List[str]:
        selected = [self.target_listbox.get(i) for i in self.target_listbox.curselection()]
        return selected

    def _prepare_run_config(self) -> RunConfig:
        data_path = self.data_path_var.get().strip()
        predict_data_path = self.predict_data_path_var.get().strip()
        if not data_path:
            raise ValueError("请先选择数据文件")
        task = self.task_var.get().strip()
        model_size = self.model_size_var.get().strip()
        model_path = self.model_path_var.get().strip()
        targets = self.get_selected_targets()

        if task in ("Classification", "Regression") and not targets:
            raise ValueError("分类/回归任务请至少选择一个目标列")
        if task == "Classification" and len(targets) > 1:
            self.log("提示: 分类任务选了多列，将使用第一个目标列。")
            targets = [targets[0]]
        if task == "Regression" and len(targets) > 1:
            self.log("提示: 当前回归任务仅支持单目标，将使用第一个目标列。")
            targets = [targets[0]]

        test_size = float(self.test_size_var.get().strip())
        if not 0.05 <= test_size <= 0.8:
            raise ValueError("测试集比例建议在 0.05 ~ 0.8")
        random_state = int(self.seed_var.get().strip())

        out_dir = self.output_dir_var.get().strip() or "./outputs"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        return RunConfig(
            task=task,
            data_path=data_path,
            predict_data_path=predict_data_path,
            model_size=model_size,
            model_path=model_path,
            target_columns=targets,
            test_size=test_size,
            random_state=random_state,
            use_cpu=bool(self.use_cpu_var.get()),
            use_retrieval=bool(self.use_retrieval_var.get()),
            local_output_dir=out_dir,
        )

    def start_run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("运行中", "当前任务仍在运行，请稍后。")
            return
        try:
            run_cfg = self._prepare_run_config()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(run_cfg.local_output_dir, f"run_{now}.log")
        file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(file_handler)

        self.log("=" * 80)
        self.log(f"任务开始，参数: {run_cfg}")

        self.progress["value"] = 0
        self.worker_thread = threading.Thread(target=self._run_pipeline_thread, args=(run_cfg, file_handler), daemon=True)
        self.worker_thread.start()

    def _run_pipeline_thread(self, run_cfg: RunConfig, file_handler: logging.FileHandler):
        try:
            self.root.after(0, lambda: self._set_progress(5, "加载依赖"))
            import numpy as np
            import pandas as pd
            import torch
            from huggingface_hub import hf_hub_download
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder

            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))

            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")

            from inference.predictor import LimiXPredictor

            self.root.after(0, lambda: self._set_progress(15, "加载数据"))
            if run_cfg.data_path.lower().endswith(".csv"):
                df = pd.read_csv(run_cfg.data_path)
            else:
                df = pd.read_excel(run_cfg.data_path)

            predict_df = None
            if run_cfg.predict_data_path:
                if run_cfg.predict_data_path.lower().endswith(".csv"):
                    predict_df = pd.read_csv(run_cfg.predict_data_path)
                else:
                    predict_df = pd.read_excel(run_cfg.predict_data_path)

            self.log(f"数据加载完成: shape={df.shape}")

            self.root.after(0, lambda: self._set_progress(28, "数据预处理"))

            def _normalize_missing_values(frame):
                cleaned = frame.copy()
                text_cols = cleaned.select_dtypes(include=["object", "string"]).columns
                if len(text_cols) > 0:
                    cleaned[text_cols] = cleaned[text_cols].replace(r"^\s*$", np.nan, regex=True)
                return cleaned

            def _normalize_column_names(frame):
                cleaned = frame.copy()
                normalized_cols = cleaned.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
                if normalized_cols.duplicated().any():
                    dup_cols = normalized_cols[normalized_cols.duplicated()].tolist()
                    raise ValueError(f"检测到重复列名（规范化后）: {dup_cols}")
                cleaned.columns = normalized_cols
                return cleaned

            def _align_columns(frame, expected_cols):
                normalized = _normalize_column_names(frame)
                col_lookup = {str(c).strip().casefold(): c for c in normalized.columns}
                aligned = {}
                missing = []
                for col in expected_cols:
                    key = str(col).strip().casefold()
                    if key in col_lookup:
                        aligned[col] = normalized[col_lookup[key]]
                    else:
                        missing.append(col)

                if missing:
                    raise ValueError(
                        "预测文件缺少训练特征列: "
                        + ", ".join(missing)
                        + f"。可用列为: {list(normalized.columns)}"
                    )
                return pd.DataFrame(aligned)

            def _encode_features(feature_df, fit_info=None):
                work_df = feature_df.copy()
                if fit_info is None:
                    fit_info = {"datetime": {}, "category": {}}
                    fit_mode = True
                else:
                    fit_mode = False

                for col in work_df.columns:
                    if fit_mode:
                        if pd.api.types.is_datetime64_any_dtype(work_df[col]):
                            fit_info["datetime"][col] = True
                        elif work_df[col].dtype in ["object", "string"]:
                            parsed = pd.to_datetime(work_df[col], errors="coerce")
                            if parsed.notna().mean() > 0.8:
                                fit_info["datetime"][col] = True
                    if col in fit_info["datetime"]:
                        parsed = pd.to_datetime(work_df[col], errors="coerce")
                        work_df[f"{col}__year"] = parsed.dt.year
                        work_df[f"{col}__month"] = parsed.dt.month
                        work_df[f"{col}__day"] = parsed.dt.day
                        work_df[f"{col}__dayofweek"] = parsed.dt.dayofweek
                        timestamp_vals = parsed.view("int64").astype("float64")
                        timestamp_vals[timestamp_vals == np.iinfo(np.int64).min] = np.nan
                        work_df[f"{col}__timestamp"] = timestamp_vals / 1e9
                        work_df = work_df.drop(columns=[col])

                for col in work_df.columns:
                    if work_df[col].dtype in ["object", "string", "category", "bool"]:
                        if fit_mode:
                            categories = pd.Series(work_df[col].astype("string").dropna().unique()).tolist()
                            fit_info["category"][col] = {v: i for i, v in enumerate(categories)}
                        mapping = fit_info["category"].get(col, {})
                        work_df[col] = work_df[col].astype("string").map(mapping)

                return np.asarray(work_df, dtype=np.float32), fit_info, work_df.columns.tolist()

            if run_cfg.task == "Missing Value Imputation":
                source_df = _normalize_column_names(_normalize_missing_values(df))
                feature_train_df = source_df[~source_df.isna().any(axis=1)].copy()
                feature_test_df = _align_columns(_normalize_missing_values(predict_df), source_df.columns) if predict_df is not None else source_df[source_df.isna().any(axis=1)].copy()
                if feature_test_df.empty:
                    raise ValueError("未找到待插补样本。请在数据中保留空值，或提供预测文件。")
                y_train = np.zeros((len(feature_train_df),), dtype=np.int64)
            else:
                df = _normalize_column_names(_normalize_missing_values(df))
                target_col = run_cfg.target_columns[0]
                feature_cols = [c for c in df.columns if c not in run_cfg.target_columns]
                train_df = df[df[target_col].notna()].copy()
                y_raw = train_df[target_col]
                feature_train_df = train_df[feature_cols]

                if predict_df is not None:
                    feature_test_df = _align_columns(_normalize_missing_values(predict_df), feature_cols).copy()
                else:
                    missing_target_df = df[df[target_col].isna()].copy()
                    if not missing_target_df.empty:
                        feature_test_df = missing_target_df[feature_cols]
                    else:
                        train_part, test_part = train_test_split(train_df, test_size=run_cfg.test_size, random_state=run_cfg.random_state)
                        feature_train_df = train_part[feature_cols]
                        y_raw = train_part[target_col]
                        feature_test_df = test_part[feature_cols]

                if run_cfg.task == "Classification":
                    y_train = LabelEncoder().fit_transform(y_raw.fillna("__NA__"))
                    y_train = np.asarray(y_train, dtype=np.int64)
                else:
                    y_train = np.asarray(y_raw, dtype=np.float32)

            x_train, encoder_info, encoded_cols = _encode_features(feature_train_df)
            x_test, _, _ = _encode_features(feature_test_df, encoder_info)

            self.root.after(0, lambda: self._set_progress(45, "准备模型"))

            use_cuda = torch.cuda.is_available() and not run_cfg.use_cpu
            device = torch.device("cuda" if use_cuda else "cpu")

            if run_cfg.model_path:
                model_path = run_cfg.model_path
            else:
                repo_id = "stableai-org/LimiX-16M" if run_cfg.model_size == "16M" else "stableai-org/LimiX-2M"
                filename = "LimiX-16M.ckpt" if run_cfg.model_size == "16M" else "LimiX-2M.ckpt"
                self.log(f"未指定本地模型，开始下载: {repo_id}/{filename}")
                model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(self.project_root / "cache"))

            if run_cfg.task == "Classification":
                config_name = "config/cls_default_16M_retrieval.json" if run_cfg.use_retrieval else "config/cls_default_noretrieval.json"
            elif run_cfg.task == "Regression":
                config_name = "config/reg_default_16M_retrieval.json" if run_cfg.use_retrieval else "config/reg_default_noretrieval.json"
            else:
                config_name = "config/reg_default_noretrieval_MVI.json"

            if run_cfg.model_size == "2M" and run_cfg.task != "Missing Value Imputation":
                config_name = (
                    "config/cls_default_2M_retrieval.json" if run_cfg.task == "Classification" and run_cfg.use_retrieval
                    else "config/reg_default_2M_retrieval.json" if run_cfg.task == "Regression" and run_cfg.use_retrieval
                    else "config/cls_default_noretrieval.json" if run_cfg.task == "Classification"
                    else "config/reg_default_noretrieval.json"
                )

            if device.type == "cpu" and "retrieval" in config_name and "noretrieval" not in config_name:
                self.log("CPU 不支持 retrieval，自动切换 noretrieval 配置")
                config_name = "config/cls_default_noretrieval.json" if run_cfg.task == "Classification" else "config/reg_default_noretrieval.json"

            config_name = str(self.project_root / config_name)

            self.log(f"使用设备={device}, config={config_name}, model={model_path}")

            self.root.after(0, lambda: self._set_progress(65, "模型推理"))
            predictor = LimiXPredictor(
                device=device,
                model_path=model_path,
                inference_config=config_name,
                mask_prediction=(run_cfg.task == "Missing Value Imputation"),
            )

            if run_cfg.task == "Missing Value Imputation":
                pred, reconstructed = predictor.predict(x_train, y_train, x_test, task_type="Regression")
                pred_values = reconstructed[-x_test.shape[0]:]
                if len(pred_values) != x_test.shape[0]:
                    pred_values = pred_values[: x_test.shape[0]]
            else:
                pred = predictor.predict(x_train, y_train, x_test, task_type=run_cfg.task)
                if hasattr(pred, "detach"):
                    pred = pred.detach().cpu().numpy()
                elif hasattr(pred, "to"):
                    pred = pred.to("cpu").numpy()
                pred_values = pred
                if getattr(pred_values, "shape", [0])[0] != x_test.shape[0]:
                    self.log(f"模型输出行数({getattr(pred_values, 'shape', [0])[0]})与预测样本数({x_test.shape[0]})不一致，截断到预测样本数。")
                    pred_values = pred_values[: x_test.shape[0]]

            self.root.after(0, lambda: self._set_progress(86, "保存结果"))
            out_base = Path(run_cfg.local_output_dir)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            pred_path = out_base / f"prediction_{run_cfg.task.replace(' ', '_')}_{now}.csv"

            if run_cfg.task == "Classification":
                pred_df = pd.DataFrame(pred_values)
                pred_df.insert(0, "pred_label", pred_df.values.argmax(axis=1))
            elif run_cfg.task == "Regression":
                pred_df = feature_test_df.reset_index(drop=True).copy()
                pred_col_name = f"pred_{run_cfg.target_columns[0]}"
                pred_df[pred_col_name] = pred_values if pred_values.ndim == 1 else pred_values[:, 0]

                # 小样本、单特征的数值回归场景，优先给出线性回归兜底结果，避免基础模型在外推时出现量纲偏移。
                if feature_train_df.shape[1] >= 1 and len(feature_train_df) >= 5:
                    try:
                        from sklearn.linear_model import LinearRegression

                        x_lr_train = feature_train_df.apply(pd.to_numeric, errors="coerce")
                        x_lr_test = feature_test_df.apply(pd.to_numeric, errors="coerce")
                        y_lr_train = pd.Series(y_train).astype(float)
                        valid_mask = (~x_lr_train.isna().any(axis=1)) & y_lr_train.notna()
                        if valid_mask.sum() >= 5 and not x_lr_test.isna().any().any():
                            lr_model = LinearRegression()
                            lr_model.fit(x_lr_train.loc[valid_mask], y_lr_train.loc[valid_mask])
                            lr_pred = lr_model.predict(x_lr_test)
                            limix_span = float(np.nanmax(pred_df[pred_col_name]) - np.nanmin(pred_df[pred_col_name])) if len(pred_df) else 0.0
                            lr_span = float(np.nanmax(lr_pred) - np.nanmin(lr_pred)) if len(lr_pred) else 0.0
                            if limix_span < 0.5 * lr_span:
                                self.log("检测到深度模型回归外推幅度明显偏小，已采用线性回归兜底预测结果。")
                                pred_df[pred_col_name] = lr_pred
                    except Exception as lr_exc:
                        self.log(f"线性回归兜底未生效: {lr_exc}")

                importances = []
                x_imp = np.nan_to_num(x_train.copy(), nan=np.nanmean(x_train, axis=0))
                y_imp = np.nan_to_num(y_train.copy(), nan=np.nanmean(y_train))
                x_std = np.std(x_imp, axis=0)
                y_std = np.std(y_imp)
                x_std[x_std == 0] = 1.0
                y_std = y_std if y_std > 0 else 1.0
                x_norm = (x_imp - np.mean(x_imp, axis=0)) / x_std
                y_norm = (y_imp - np.mean(y_imp)) / y_std
                coeff = (x_norm.T @ y_norm) / max(len(y_norm) - 1, 1)
                for idx, name in enumerate(encoded_cols):
                    importances.append({"feature": name, "impact": float(coeff[idx])})
                pd.DataFrame(importances).sort_values("impact", key=lambda s: s.abs(), ascending=False).to_csv(
                    out_base / f"feature_impact_{now}.csv", index=False
                )

                relation_rows = []
                pred_array = pred_values if pred_values.ndim == 1 else pred_values[:, 0]
                pred_std = float(np.std(pred_array))
                pred_std = pred_std if pred_std > 0 else 1.0
                pred_norm = (pred_array - np.mean(pred_array)) / pred_std
                x_test_imp = np.nan_to_num(x_test.copy(), nan=np.nanmean(x_train, axis=0))
                x_test_std = np.std(x_test_imp, axis=0)
                x_test_std[x_test_std == 0] = 1.0
                x_test_norm = (x_test_imp - np.mean(x_test_imp, axis=0)) / x_test_std
                corr_with_pred = (x_test_norm.T @ pred_norm) / max(len(pred_norm) - 1, 1)

                for idx, name in enumerate(encoded_cols):
                    relation_rows.append(
                        {
                            "feature": name,
                            "corr_with_target": float(coeff[idx]),
                            "corr_with_prediction": float(corr_with_pred[idx]),
                            "causal_hint": float(coeff[idx] * corr_with_pred[idx]),
                        }
                    )

                pd.DataFrame(relation_rows).sort_values("causal_hint", key=lambda s: s.abs(), ascending=False).to_csv(
                    out_base / f"feature_relation_{now}.csv", index=False
                )
                self.log(f"已输出特征相关性/因果线索: {out_base / f'feature_relation_{now}.csv'}")
            else:
                imputed = pd.DataFrame(pred_values, columns=feature_train_df.columns)
                base = feature_test_df.reset_index(drop=True).copy()
                for col in base.columns:
                    if base[col].isna().any() and col in imputed.columns:
                        fill_values = imputed[col]
                        if col in encoder_info.get("category", {}):
                            inverse_mapping = {v: k for k, v in encoder_info["category"][col].items()}
                            fill_values = pd.Series(fill_values).round().astype("Int64").map(inverse_mapping)
                        base[col] = base[col].fillna(fill_values)

                # 若模型插补后仍存在空值，则使用经典线性回归兜底插补，保证输出完整。
                if base.isna().any().any():
                    from sklearn.linear_model import LinearRegression

                    complete_rows = source_df[~source_df.isna().any(axis=1)].copy()
                    for col in base.columns:
                        missing_mask = base[col].isna()
                        if not missing_mask.any():
                            continue
                        feature_cols_for_col = [c for c in base.columns if c != col]
                        train_part = complete_rows[feature_cols_for_col + [col]].dropna()
                        if train_part.empty:
                            continue
                        x_fit = train_part[feature_cols_for_col]
                        y_fit = train_part[col]
                        x_pred = base.loc[missing_mask, feature_cols_for_col]
                        if x_pred.isna().any().any():
                            x_pred = x_pred.fillna(x_fit.mean(numeric_only=True))
                        model = LinearRegression()
                        model.fit(x_fit, y_fit)
                        base.loc[missing_mask, col] = model.predict(x_pred)
                pred_df = base

            pred_df.to_csv(pred_path, index=False)

            meta_path = out_base / f"run_meta_{now}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(run_cfg.__dict__, f, ensure_ascii=False, indent=2)

            self.log(f"推理完成，结果已保存: {pred_path}")
            self.log(f"日志文件: {self.log_file_path}")
            self.root.after(0, lambda: self._set_progress(100, "完成"))
            self.root.after(0, lambda: messagebox.showinfo("完成", f"推理完成!\n输出: {pred_path}\n日志: {self.log_file_path}"))

        except Exception as exc:
            self.log(f"运行失败: {exc}")
            self.log(traceback.format_exc())
            self.root.after(0, lambda: self._set_progress(0, "失败"))
            self.root.after(0, lambda: messagebox.showerror("运行失败", str(exc)))
        finally:
            self.logger.removeHandler(file_handler)
            file_handler.close()


def main():
    root = tk.Tk()
    app = LimiXGuiApp(root)
    app.log("欢迎使用 LimiX2Cheese GUI。建议顺序：检测依赖 -> 选择文件与任务 -> 开始推理")
    root.mainloop()


if __name__ == "__main__":
    main()
