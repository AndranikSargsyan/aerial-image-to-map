import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .unet_model import *
from .setup_dataloaders import*


unet_model = UNet(3,3)
unet_model.load_state_dict(torch.load('best_qartezator_unet_0.06958.pth'))


# Define loss function and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.00001)


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device = 'cuda'
):
    model.train()
    train_loss = 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} ")

    
def val_step(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device = 'cuda'
):
    test_loss = 0
    model.to(device)
    model.eval()
    with torch.no_grad(): 
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
        test_loss /= len(data_loader)
        print(f"Validation loss: {test_loss:.5f}\n")
    return test_loss

def train_model(model, model_name, loss_fn, optimizer, train_loader, val_loader, epochs=500, seed=0, scheduler=None, device='cuda'):
    best_val_loss = np.inf
    best_path = ''
    
    torch.manual_seed(seed)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n---------")

        train_step(data_loader=train_loader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        
        if scheduler is not None:
            scheduler.step()
 
        val_loss = val_step(data_loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            device=device
        )

        if epoch % 5 == 0:
            for source, target in val_dataloader:
                source = source.to(device)
                model.to(device)
                test_pred = model(source)
                test_pred = test_pred.detach().permute(0,2,3,1).clip(0, 1).cpu().numpy()
                grid = []
                for i in range(len(test_pred)):
                    s = source[i].permute(1,2,0).detach().cpu().numpy()
                    s = (s * 255).astype(np.uint8)
                    t = target[i].permute(1,2,0).detach().cpu().numpy()
                    t = (t * 255).astype(np.uint8)
                    tp = test_pred[i]
                    tp = (tp * 255).astype(np.uint8)
                    grid.append(np.hstack([s, t, tp]))
                grid = np.vstack(grid)
                Image.fromarray(grid).save(f'grid_{epoch}.png')
                break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f'best_{model_name}_{val_loss:.5f}.pth'
            print(f"Saving best {model_name} with val_loss={val_loss}")
            torch.save(obj=model.state_dict(), f=best_path)

                
        torch.save(obj=model.state_dict(), f=f'last_{model_name}.pth')
    
    return model.load_state_dict(torch.load(f=best_path))

    
train_model(unet_model, 'qartezator_unet', loss_fn, optimizer, train_dataloader, val_dataloader, epochs=500)