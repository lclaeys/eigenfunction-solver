from omegaconf import OmegaConf
from copy import deepcopy

def flatten_cfg(cfg, prefix=""):
    """
    Recursively flatten a plain dictionary.
    """
    flat = {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_cfg(v, full_key))
        else:
            flat[full_key] = v
    return flat

def set_nested(cfg, key_path, value):
    """
    Set a value in a nested dictionary given a dot-separated key path.
    """
    keys = key_path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value

def expand_cfg(cfg):
    """
    Expand a config by detecting list-valued keys, verifying that all such
    lists have the same length, and creating a separate OmegaConf config for
    each index.
    """
    # Convert cfg to a plain dict with all interpolations resolved.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Flatten the plain dictionary.
    flat = flatten_cfg(cfg_dict)
    
    # Identify keys with list values.
    list_keys = [k for k, v in flat.items() if isinstance(v, list)]
    
    # If no list-valued keys, return the original cfg.
    if not list_keys:
        return [cfg]
    
    # Ensure that all list-valued keys have the same length.
    list_lengths = [len(flat[k]) for k in list_keys]
    if len(set(list_lengths)) != 1:
        raise ValueError("All list-valued arguments must have the same length.")
    
    num_configs = list_lengths[0]
    expanded_cfgs = []
    
    # For each index, create a new config where list values are replaced by their corresponding element.
    for i in range(num_configs):
        new_cfg = {}
        for k, v in flat.items():
            if k in list_keys:
                set_nested(new_cfg, k, v[i])
            else:
                set_nested(new_cfg, k, v)
        expanded_cfgs.append(OmegaConf.create(new_cfg))
        
    return expanded_cfgs

# Example usage:
if __name__ == "__main__":
    # Simulate a CLI input with nested keys.
    cli_cfg = OmegaConf.from_cli()
    
    cfgs = expand_cfg(cli_cfg)
    for cfg in cfgs:
        print(OmegaConf.to_yaml(cfg))
