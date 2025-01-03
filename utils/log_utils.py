import wandb
from omegaconf import OmegaConf


def setup_wandb_and_logging(cfg):
    """
    Initializes a Weights & Biases run, logs hyperparameters,
    and returns the run name (which we'll also use for checkpoint directory).
    """
    # 1) Generate a descriptive run name
    #    E.g., include date/time, or major hyperparams
    run_name = (
        f"2ndtry_Aligner_"
        f"ME_{cfg.model.motion_encoder}_"
        f"TE_{cfg.model.text_encoder}_"
        f"LR_{cfg.train.learning_rate}_"
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
