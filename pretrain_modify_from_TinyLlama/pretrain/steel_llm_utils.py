


def compatible_tiny_llama_config(config, block_size):
    """fit tinyllama's some function"""
    assert(block_size >0 and block_size < 1000000)
    config.block_size = block_size
    config.n_layer = config.num_hidden_layers
    config.n_embd = config.hidden_size
    return config
    