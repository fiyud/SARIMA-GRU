# SARIMA-GRU Time Series Forecasting
A hybrid deep learning model that combines SARIMA (Seasonal AutoRegressive Integrated Moving Average) components with GRU (Gated Recurrent Unit) neural networks for time series forecasting, specifically designed for lake water level prediction.

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended for faster training)
- **CPU**: Multi-core processor (fallback option)
- **RAM**: Minimum 8GB, 16GB+ recommended for larger datasets

## Quick Start

### 1. Prepare Your Data
Place your CSV file in the `Dataset/` directory. The data should have:
- A time column (will be automatically parsed)
- Target column (e.g., 'Lake water level (m)')
- Feature columns (environmental variables, meteorological data, etc.)

Example data format:
```csv
Time,Lake water level (m),Temperature,Precipitation,Humidity,...
2019-01-01 00:00:00,15.2,25.3,0.0,65.2,...
2019-01-01 01:00:00,15.1,25.1,0.2,66.1,...
```

### 2. Basic Training
```bash
python train.py

python train.py --data_path Dataset/your_data.csv
```

### 3. Custom Training
```bash
python train.py --data_path Dataset/KaNak.csv --target_column "Lake water level (m)" --num_epochs 200 --batch_size 32 --learning_rate 0.0005 --hidden_size 64 --num_layers 3
```

### Loading for Inference
```python
import torch
from sarimaGRU import SARIMAGRU

checkpoint = torch.load('sarima_gru_model.pth')
model = SARIMAGRU(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
## Getting Started Checklist

- [ ] Install required dependencies
- [ ] Prepare your dataset in CSV format
- [ ] Place data in `Dataset/` directory
- [ ] Run basic training: `python train.py`
- [ ] Check output visualizations
- [ ] Experiment with different parameters
- [ ] Save and document your best model
