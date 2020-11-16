import dill
def pretrain_loader(filename):
    with open(filename, 'rb') as f:
        model_pretrained = dill.load(f)
    return model_pretrained
