import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='cuda'):
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", 
                         leave=False, unit="batch")
        
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            outputs, _ = model(batch_x)
            loss = criterion(outputs[:, -1, :], batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            
            train_pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
        
        model.eval()
        val_loss = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val", 
                       leave=False, unit="batch")
        
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs, _ = model(batch_x)
                loss = criterion(outputs[:, -1, :], batch_y)
                batch_loss = loss.item()
                val_loss += batch_loss
                
                val_pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        if (epoch + 1) % 10 == 0:
            tqdm.write(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if torch.cuda.is_available() and (epoch + 1) % 20 == 0:
            torch.cuda.empty_cache()
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, scaler_target, device='cuda'):
    model.eval()
    predictions = []
    actuals = []
    
    eval_pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for batch_x, batch_y in eval_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs, _ = model(batch_x)
            pred = outputs[:, -1, :].cpu().numpy()
            actual = batch_y.cpu().numpy()
            
            predictions.extend(pred)
            actuals.extend(actual)
            
            eval_pbar.set_postfix({'batch_size': len(pred)})
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    
    print(f'\n=== Model Evaluation Results ===')
    print(f'Test MSE: {mse:.6f}')
    print(f'Test MAE: {mae:.6f}')
    print(f'Test RMSE: {rmse:.6f}')
    
    return predictions, actuals