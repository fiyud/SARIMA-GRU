import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
import os

from loadData import prepare_data
from sarimaGRU import SARIMAGRU
from general import train_model, evaluate_model

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='SARIMA-GRU Time Series Forecasting')
    
    parser.add_argument('--data_path', type=str, default='DataSet/KaNak.csv',
                       help='Path to the CSV data file')
    parser.add_argument('--target_column', type=str, default='Lake water level (m)',
                       help='Name of the target column to predict')
    parser.add_argument('--sequence_length', type=int, default=24,
                       help='Length of input sequences')
    
    parser.add_argument('--hidden_size', type=int, default=32,
                       help='Hidden size of GRU layers')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GRU layers')
    parser.add_argument('--seasonal_period', type=int, default=24,
                       help='Seasonal period for SARIMA component')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    
    parser.add_argument('--model_path', type=str, default='sarima_gru_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--plot_samples', type=int, default=200,
                       help='Number of samples to plot in visualizations')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip plotting visualizations')
    
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). If None, auto-detect')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Using CPU")

    print("="*60)
    print(" SARIMA-GRU Time Series Forecasting")
    print("="*60)
    print(f"Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Target column: {args.target_column}")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Seasonal period: {args.seasonal_period}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Model save path: {args.model_path}")

    print("\n Preparing data...")
    train_dataset, test_dataset, scaler_features, scaler_target = prepare_data(
        args.data_path, 
        sequence_length=args.sequence_length,
        target_column=args.target_column
    )

    print(f"\n Creating data loaders (batch size: {args.batch_size})...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True if torch.cuda.is_available() else False)

    input_size = len(train_dataset.feature_columns)
    model = SARIMAGRU(
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=1,
        num_layers=args.num_layers,
        seasonal_period=args.seasonal_period
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n Model initialized:")
    print(f"   Device: {device}")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {args.hidden_size}")
    print(f"   Layers: {args.num_layers}")
    print(f"   Seasonal period: {args.seasonal_period}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    if torch.cuda.is_available():
        print(f" GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
        print(f" GPU Memory cached: {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")

    print(f"\n Training model for {args.num_epochs} epochs...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, 
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate,
        device=device
    )

    print(f"\n Evaluating model...")
    predictions, actuals = evaluate_model(model, test_loader, scaler_target, device=device)

    if not args.no_plot:
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
        plot_samples = min(args.plot_samples, len(actuals))
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

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': args.hidden_size,
            'output_size': 1,
            'num_layers': args.num_layers,
            'seasonal_period': args.seasonal_period
        },
        'scaler_features': scaler_features,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'device': str(device)
    }, args.model_path)

    print(f"\n Saved model to '{args.model_path}'")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()