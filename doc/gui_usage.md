# LimiX2Cheese GUI Usage

Use `limix_gui.py` for one-click inference workflows.

## Features
- Dependency check + one-click install
- Input file picker (`csv/xlsx/xls`)
- Task selection: classification / regression / missing-value imputation
- Local `.ckpt` selection or auto-download from Hugging Face
- Target-column selection
- Progress bar + real-time log
- Exports: prediction CSV, metadata JSON, log file

## Launch
```bash
python limix_gui.py
# Linux/macOS
./run_gui.sh
# Windows
run_gui.bat
```

## Package as EXE (optional)
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name LimiX2Cheese-GUI limix_gui.py
```

## FAQ
If GUI says Excel dependencies are missing although installed, it is usually an interpreter mismatch.
Check the GUI log line: `Python 解释器: ...` and install into that interpreter:
```bash
python -m pip install openpyxl
```
