import json
from glob import glob
import argparse
import os

from ssd.utils.generator import Generator
from ssd.utils import utils

if __name__ == "__main__":
    # Path manager
    GATE_TEMPLATE = "ressources/gate_template-modif.png"
    SAVE_TEMPLATE = "/IMG_CUSTOM"
    TARGET_DEFAULT = "/data/Data_Synthbackgrounds2"
    PROCESS_NB = 8

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nbsamples",
        type=int,
        help="number of images generate",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="target directory to save data",
        required=False,
    )

    args = parser.parse_args()

    def update_path(path):
        if path[0] != '/':
            path = os.path.dirname(os.path.realpath(__file__)) + '/' + path.strip('./')
        return path

    if args.target:
        SAVE_TEMPLATE = update_path(args.target) + SAVE_TEMPLATE
    else:
        SAVE_TEMPLATE = update_path(TARGET_DEFAULT) + SAVE_TEMPLATE

    generator = Generator()

    # Load backgrounds

    BACKGROUND_DIR = "ressources/drl_backgrounds"
    for file in glob(BACKGROUND_DIR + "/*.png"):
        generator.setBackground(file)
    BACKGROUND_DIR = "ressources/park_img"
    for file in glob(BACKGROUND_DIR + "/*.png"):
        generator.setBackground(file)
    BACKGROUND_DIR = "ressources/synt_backgrounds"
    for file in glob(BACKGROUND_DIR + "/*.jpg"):
        generator.setBackground(file)
    for file in glob(BACKGROUND_DIR + "/*.png"):
        generator.setBackground(file)

    # Load gates
    generator.setGate(GATE_TEMPLATE)

    # Setup settings
    generator.setMultigate(0.00, mult=2)
    generator.setLightglobal(0.20)
    generator.setLightellipse(0.75)
    generator.setBlur(0.20)
    generator.setBurnGate(0.75)

    # TODO: Implement those
    generator.setObstacle(0.0)  # Not yet implemented
    generator.setRotation(0.0)  # Not yet implemented

    results = generator.multiprocessImage(PROCESS_NB, SAVE_TEMPLATE, nb=args.nbsamples, debug=True)

    to_dump = {}
    for key, value in results:
        to_dump[key] = utils.crush_data(value)
    # save data to a `.json`

    with open("{}.json".format(SAVE_TEMPLATE), "w") as outfile:
        json.dump(to_dump, outfile)
