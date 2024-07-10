import torch
import os
import logging

def save_state(path,accelerator,iteration):
        if not os.path.exists(path):
            os.makedirs(path)
        accelerator.save_state(os.path.join(
                path, f"checkpoint_{iteration}_save"
            ))
        
def save_pt(path,accelerator,model,optimizer,iteration):
    if accelerator.is_main_process:
        if not os.path.exists(path):
            os.makedirs(path)
        model = accelerator.unwrap_model(model)
        torch.save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        iteration=iteration,
                    ),
                    f=os.path.join(
                        path, f"checkpoint_{iteration}.pt"
                    ),
                )
        
def load_pt(path,accelerator,model):
        checkpoint_path = path
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path) or not os.listdir(checkpoint_path):
                logging.warning(f"Checkpoint path {checkpoint_path} does not exist or is empty")
                return -1

            checkpoint_files = sorted(
                [f for f in os.listdir(checkpoint_path) if f.endswith(".pt")],
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            if not checkpoint_files:
                logging.warning(f"No checkpoint files found in {checkpoint_path}")
                return -1

            checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[-1])
            with accelerator.main_process_first():
                checkpoint = torch.load(checkpoint_file, map_location="cpu")
                model.load_state_dict(checkpoint["model"])
            return checkpoint.get("iteration", -1)
        return -1


def load_state(path, accelerator):
    if path is not None:
        checkpoint_path = path

        if not os.path.exists(checkpoint_path) or len(os.listdir(checkpoint_path)) == 0:
            logging.warning(f"Checkpoint path {checkpoint_path} does not exist or is empty")
            return None
        else:
            checkpoint_files = [
                f for f in os.listdir(checkpoint_path) if f.endswith("_save")
            ]
            if not checkpoint_files:
                logging.warning(f"No checkpoint files found in {checkpoint_path}")
                return None

            checkpoint_files = sorted(
                checkpoint_files,
                key=lambda x: int(os.path.basename(x).split("_")[1])
            )
            accelerator.load_state(os.path.join(checkpoint_path, checkpoint_files[-1]))
            return int(os.path.basename(checkpoint_files[-1]).split("_")[1]) + 1
    else:
        logging.warning("No checkpoint path specified in the configuration")
        return -1
    
def find_max_param_and_grad(model):
    max_param_val = -float('inf')
    max_grad_val = -float('inf')
    min_grad_val = float('inf')
    for name, param in model.named_parameters():
        max_param_val = max(max_param_val, param.data.abs().max().item())
        if param.grad is not None:
            max_grad_val = max(max_grad_val, param.grad.abs().max().item())
            min_grad_val = min(min_grad_val, param.grad.abs().min().item())
    if max_grad_val == -float('inf'):
        max_grad_val = None
    if min_grad_val == float('inf'):
        min_grad_val = None
    return max_param_val, max_grad_val, min_grad_val
