from omegaconf import OmegaConf

def load_config(path="configs/config.yaml"):
    # Load the YAML file into an OmegaConf DictConfig
    config = OmegaConf.load(path)
    return config
