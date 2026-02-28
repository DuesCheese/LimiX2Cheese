# LimiX2Cheese GUI 使用说明

`limix_gui.py` 提供一键式推理桌面前端。

## 功能
- 依赖检查 + 一键安装
- 文件选择（`csv/xlsx/xls`）
- 任务选择：分类 / 回归 / 缺失值插补
- 选择本地 `.ckpt` 或自动下载模型
- 目标列选择
- 进度条与实时日志
- 导出预测 CSV、元数据 JSON、日志文件

## 启动
```bash
python limix_gui.py
# Linux/macOS
./run_gui.sh
# Windows
run_gui.bat
```

## 打包 EXE（可选）
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name LimiX2Cheese-GUI limix_gui.py
```

## 常见问题
若提示 Excel 依赖缺失但你已安装，通常是解释器不一致。
先看 GUI 日志：`Python 解释器: ...`，然后在同一解释器安装：
```bash
python -m pip install openpyxl
```
