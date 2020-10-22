import argparse
import tensorflow as tf

def crush_data(arr):
    to_return = []
    for bbox in arr:
        box = []
        for c in bbox:
            if type(c) == tuple:
                box += list(c)
            else:
                box += [c]
        to_return.append(box)
    return to_return

def print_box(txt, l):
    nb = int((l - len(txt)) / 2)
    if nb * 2 + len(txt) != l:
        nb1 = nb + 1
    else:
        nb1 = nb
    t = "+" + l * "-" + "+\n"
    t += "|" + nb * " " + txt + nb1 * " " + "|\n"
    t += "+" + l * "-" + "+\n"
    print(t)

def get_args_training():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--nbrepoch", type=int, default=1,
        help="Number of epoch.",
    )
    parser.add_argument(
        "-b", "--batchsize", type=int, default=1,
        help="Size of the batch.",
    )
    parser.add_argument(
        "-d", "--datadir", nargs='+', type=str,
        help="Directory containing data to train.", required=True,
    )
    parser.add_argument(
        "-m", "--model", type=str, default="DSOD300",
        help="Model to train.",
    )
    parser.add_argument(
        "-w", "--weights", type=str, default=None,
        help="Model to train.",
    )
    args = parser.parse_args()
    return args.nbrepoch, args.batchsize, args.datadir, args.model, args.weights

def get_model_by_name(model_name, cpu_fix=False):

    from ssd_test.models.ssd_model import SSD300, SSD512, DSOD300, DSOD512, SSD512_resnet

    models = ["SSD300", "SSD512", "DSOD300", "DSOD512", "SSD512_resnet"]
    if model_name not in models:
        print("Model `{}` do not exist...".format(model_name))
        exit(0)
    if cpu_fix:
        with tf.device('/cpu:0'):
            model = eval("{}(num_classes=2)".format(model_name))
    else:
        model = eval("{}(num_classes=2)".format(model_name))
    return model
