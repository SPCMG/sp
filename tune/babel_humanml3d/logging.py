import wandb
from omegaconf import OmegaConf


def setup_wandb_and_logging(cfg):
    """
    Initializes a Weights & Biases run, logs hyperparameters,
    and returns the run name (which we'll also use for checkpoint directory).
    
    Args:
        cfg: The OmegaConf config or dict with your hyperparams.
    
    Returns:
        run_name (str): A unique, descriptive name for this run.
    """
    # 1) Generate a descriptive run name
    #    E.g., include date/time, or major hyperparams
    run_name = (
        f"BEST_dropout_coslr_FinetuneLaCLIP_"
        f"DS_HumanML3D_"
        f"LR_{cfg.train.learning_rate}_"
        f"WD_{cfg.train.weight_decay}_"
        f"EP_{cfg.train.num_epochs}_"
        f"{wandb.util.generate_id()}"  
    )

    # 2) Initialize wandb 
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="sp_motion_generation",
        config=config_dict,
        name=run_name
    )

    return run_name