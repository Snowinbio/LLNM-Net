import ml_collections

def get_LLNM_Net_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 16
    config.transformer.attention_dropout_rate = 0.2 # 0.0 - 0.2
    config.transformer.dropout_rate = 0.3 # 0.1 - 0.3
    config.classifier = 'token'
    config.representation_size = None
    config.rr_len = 300
    config.img_feature_len = 2   
    return config
