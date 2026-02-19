import io
import torch
import os

def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema, save_every_epoch=200):
    """
    save the model to path
    """
    save_ckpt = {
        'epoch': epoch + 1, 
        'model': model.state_dict(), 
        'ema_state_dict': ema.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'schedule': scheduler.state_dict(), 
        'loss': train_loss,
        'wandb_id': wandb_id
    }
    
    torch.save(save_ckpt, f"{save_path}/latest.pth")

    if epoch+1 >= save_every_epoch:
        with open(f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth', "wb") as f:
            torch.save(save_ckpt, f)


def save_model_lora(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema, save_every_epoch=200):
    """
    Save model for LoRA training: save full state_dict (base + LoRA) + LoRA-only state_dict.
    """
    # Extract LoRA-only weights for lightweight saving
    full_sd = model.state_dict()
    lora_sd = {k: v for k, v in full_sd.items() if 'lora' in k.lower()}
    
    save_ckpt = {
        'epoch': epoch + 1, 
        'model': full_sd, 
        'lora_state_dict': lora_sd,  # LoRA-only for quick inspection
        'ema_state_dict': ema.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'schedule': scheduler.state_dict(), 
        'loss': train_loss,
        'wandb_id': wandb_id
    }
    
    torch.save(save_ckpt, f"{save_path}/latest.pth")

    if epoch+1 >= save_every_epoch:
        with open(f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth', "wb") as f:
            torch.save(save_ckpt, f)


def load_model(path: str):
    """
    load ckpt from path
    """
    ckpt = torch.load(path, weights_only=True)

    return ckpt


def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    load ckpt from path
    """
    path = os.path.join(path, 'latest.pth')
    ckpt = torch.load(path, weights_only=True)

    # load model               
    try:
        model.load_state_dict(ckpt['model'])
    except:
        # Try stripping "module." prefix from DDP
        try:
            model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt['model'].items()})
        except:
            # For LoRA resume: strict=False allows missing/extra keys
            missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
            print(f"Resume with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    print("Model load done")
    
    # load optimizer
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer load done")
    except:
        print("no pretrained optimizer found")
            
    # load schedule
    try:
        scheduler.load_state_dict(ckpt['schedule'])
        print("Schedule load done")
    except:
        print("no schedule found,")
    
    # load step
    try:
        init_epoch = ckpt['epoch']
        print("Step load done")
    except:
        init_epoch = 0
    

    # Load wandb id
    try:
        wandb_id = ckpt['wandb_id']
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema.ema.load_state_dict({n: v for n, v in ckpt['ema_state_dict'].items()})
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print('no ema shadow found')

    return model, optimizer, scheduler, init_epoch, wandb_id, ema