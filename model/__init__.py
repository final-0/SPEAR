from model.waft_a2 import WAFTv2

def fetch_model(args):
    if args.algorithm == 'waft-a2':
        model = WAFTv2(args)
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))
    return model