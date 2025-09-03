import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Using CPU")

class SARIMAGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, seasonal_period=24):
        super(SARIMAGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seasonal_period = seasonal_period
        
        self.reset_gate = nn.Linear(input_size + hidden_size + hidden_size + 1, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size + hidden_size + 1, hidden_size)
        
        self.seasonal_gate = nn.Linear(input_size + hidden_size + hidden_size, hidden_size)
        self.trend_gate = nn.Linear(2, 1)
        
        self.hidden_transform = nn.Linear(input_size + hidden_size, hidden_size)
        self.seasonal_transform = nn.Linear(input_size + hidden_size, hidden_size)
        self.trend_transform = nn.Linear(2, 1)
        
        self.ar_order = 3
        self.ma_order = 3
        self.ar_weights = nn.Parameter(torch.randn(self.ar_order) * 0.1)
        self.ma_weights = nn.Parameter(torch.randn(self.ma_order) * 0.1)
        
        self.seasonal_weight = nn.Linear(hidden_size, hidden_size)
        self.trend_weight = nn.Linear(1, hidden_size)
        self.ar_weight = nn.Linear(1, hidden_size)
        self.ma_weight = nn.Linear(1, hidden_size)
        
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_t, hidden_state, seasonal_memory, trend_state, ar_buffer, ma_buffer, prev_error):
        batch_size = x_t.size(0)
        
        if seasonal_memory.size(1) >= self.seasonal_period:
            seasonal_component = seasonal_memory[:, -self.seasonal_period, :]
        else:
            seasonal_component = torch.zeros(batch_size, self.hidden_size, device=x_t.device)
        
        gate_input = torch.cat([x_t, hidden_state, seasonal_component, trend_state], dim=1)
        reset_gate = self.sigmoid(self.reset_gate(gate_input))
        update_gate = self.sigmoid(self.update_gate(gate_input))
        
        seasonal_gate_input = torch.cat([x_t, hidden_state, seasonal_component], dim=1)
        seasonal_gate = self.sigmoid(self.seasonal_gate(seasonal_gate_input))
        
        trend_gate_input = torch.cat([trend_input, trend_state], dim=1)
        trend_gate = self.sigmoid(self.trend_gate(trend_gate_input))
        
        trend_input = x_t[:, 0:1]
        new_trend_input = torch.cat([trend_input, trend_state], dim=1)
        new_trend = self.trend_transform(new_trend_input)
        trend_state = (1 - trend_gate) * trend_state + trend_gate * new_trend
        
        seasonal_input = torch.cat([x_t, reset_gate * seasonal_component], dim=1)
        new_seasonal = self.activation(self.seasonal_transform(seasonal_input))
        updated_seasonal = (1 - seasonal_gate) * seasonal_component + seasonal_gate * new_seasonal
        
        seasonal_memory = torch.cat([seasonal_memory, updated_seasonal.unsqueeze(1)], dim=1)
        if seasonal_memory.size(1) > self.seasonal_period * 2:
            seasonal_memory = seasonal_memory[:, -self.seasonal_period * 2:, :]
        
        ar_component = torch.zeros(batch_size, 1, device=x_t.device)
        if ar_buffer.size(1) >= self.ar_order:
            for i in range(self.ar_order):
                if ar_buffer.size(1) > i:
                    past_hidden = ar_buffer[:, -(i+1), :].mean(dim=1, keepdim=True)
                    ar_component += self.ar_weights[i] * past_hidden
        
        ma_component = torch.zeros(batch_size, 1, device=x_t.device)
        if ma_buffer.size(1) >= self.ma_order:
            for i in range(self.ma_order):
                if ma_buffer.size(1) > i:
                    past_error = ma_buffer[:, -(i+1), :]
                    ma_component += self.ma_weights[i] * past_error
        
        hidden_input = torch.cat([x_t, reset_gate * hidden_state], dim=1)
        new_hidden_base = self.activation(self.hidden_transform(hidden_input))
        
        seasonal_contribution = self.seasonal_weight(updated_seasonal)
        trend_contribution = self.trend_weight(trend_state)
        ar_contribution = self.ar_weight(ar_component)
        ma_contribution = self.ma_weight(ma_component)
        
        new_hidden = new_hidden_base + seasonal_contribution + trend_contribution + ar_contribution + ma_contribution
        hidden_state = (1 - update_gate) * hidden_state + update_gate * new_hidden
        
        ar_buffer = torch.cat([ar_buffer, hidden_state.unsqueeze(1)], dim=1)
        if ar_buffer.size(1) > self.ar_order * 2:
            ar_buffer = ar_buffer[:, -self.ar_order * 2:, :]
        
        current_error = prev_error.unsqueeze(1) if prev_error.dim() == 1 else prev_error
        ma_buffer = torch.cat([ma_buffer, current_error.unsqueeze(1)], dim=1)
        if ma_buffer.size(1) > self.ma_order * 2:
            ma_buffer = ma_buffer[:, -self.ma_order * 2:, :]
        
        return hidden_state, seasonal_memory, trend_state, ar_buffer, ma_buffer

class SARIMAGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, seasonal_period=24):
        super(SARIMAGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seasonal_period = seasonal_period
        
        self.sarima_gru_layers = nn.ModuleList([
            SARIMAGRUCell(input_size if i == 0 else hidden_size, hidden_size, seasonal_period)
            for i in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, hidden_states=None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        if hidden_states is None:
            hidden_states = []
            for i in range(self.num_layers):
                h = torch.zeros(batch_size, self.hidden_size, device=device)
                s = torch.zeros(batch_size, 0, self.hidden_size, device=device)
                t = torch.zeros(batch_size, 1, device=device)
                ar = torch.zeros(batch_size, 0, self.hidden_size, device=device)
                ma = torch.zeros(batch_size, 0, 1, device=device)
                hidden_states.append((h, s, t, ar, ma))
        
        outputs = []
        prev_error = torch.zeros(batch_size, 1, device=device)
        
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for i, sarima_gru_layer in enumerate(self.sarima_gru_layers):
                h, s, trend, ar, ma = hidden_states[i]
                h, s, trend, ar, ma = sarima_gru_layer(
                    layer_input, h, s, trend, ar, ma, prev_error
                )
                hidden_states[i] = (h, s, trend, ar, ma)
                layer_input = h
            
            output = self.output_layer(self.dropout(layer_input))
            outputs.append(output)
            
            if len(outputs) > 1:
                prev_error = torch.abs(outputs[-1] - outputs[-2]).mean(dim=1, keepdim=True)
        
        return torch.stack(outputs, dim=1), hidden_states

class NormalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(NormalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)        
        output = self.output_layer(self.dropout(gru_out))        
        return output




if __name__ == "__main__":
    SEQUENCE_LENGTH = 24  # hours lookback
    HIDDEN_SIZE = 32
    NUM_LAYERS = 2
    SEASONAL_PERIOD = 24  # Daily seasonality
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    print("="*60)
    print(" SARIMA-GRU Time Series Forecasting")
    print("="*60)
    
    print("\n Preparing data...")
    train_dataset, test_dataset, scaler_features, scaler_target = prepare_data(
        'DataSet/KaNak.csv', 
        sequence_length=SEQUENCE_LENGTH,
        target_column='Lake water level (m)'
    )
    
    print(f"\n Creating data loaders (batch size: {BATCH_SIZE})...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True if torch.cuda.is_available() else False)
    
    input_size = len(train_dataset.feature_columns)
    model = SARIMAGRU(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=1,
        num_layers=NUM_LAYERS,
        seasonal_period=SEASONAL_PERIOD
    )

    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n Model initialized:")
    print(f"   Device: {device}")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {HIDDEN_SIZE}")
    print(f"   Layers: {NUM_LAYERS}")
    print(f"   Seasonal period: {SEASONAL_PERIOD}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    if torch.cuda.is_available():
        print(f" GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
        print(f" GPU Memory cached: {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")
    
    print(f"\n Training model for {NUM_EPOCHS} epochs...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    print(f"\n Evaluating model...")
    predictions, actuals = evaluate_model(model, test_loader, scaler_target, device=device)
    
    print(f"\n Generating visualizations...")
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plot_samples = min(200, len(actuals))
    plt.plot(actuals[:plot_samples], label='Actual', alpha=0.7, linewidth=2)
    plt.plot(predictions[:plot_samples], label='Predicted', alpha=0.7, linewidth=2)
    plt.title(f'Predictions vs Actual (First {plot_samples} points)')
    plt.xlabel('Time')
    plt.ylabel('Lake Water Level (normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    residuals = actuals - predictions
    plt.plot(residuals[:plot_samples], alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title(f'Residuals (First {plot_samples} points)')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.scatter(actuals, predictions, alpha=0.5, s=10)
    min_val, max_val = min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    plt.title('Actual vs Predicted Scatter')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    model_path = 'sarima_gru_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': HIDDEN_SIZE,
            'output_size': 1,
            'num_layers': NUM_LAYERS,
            'seasonal_period': SEASONAL_PERIOD
        },
        'scaler_features': scaler_features,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'device': str(device)
    }, model_path)
    
    print(f"\n Saved model to '{model_path}'")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()