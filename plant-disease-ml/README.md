## Quick start (Windows 11 + VS Code + Python 3.12)

```powershell
# 1) Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt


# 6) Run desktop app
>> python ui\app.py `
>>   --weights runs\best_model.pth `
>>   --label-map data\label_map.json `
>>   --model-name efficientnet_b0 `
>>   --multiclass true `
>>   --detector-onnx runs\leafdet_yolov8n_480.onnx `
>>   --det-names runs\leafdet_classes.json `
>>   --det-classes leaf `
>>   --det-conf 0.32 `
>>   --det-iou 0.45 `
>>   --det-input 480
