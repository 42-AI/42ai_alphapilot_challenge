# Basic imports
from random import randint, random
import numpy as np

# Local imports
from ground_truth import GroundTruth

class InputGenerator(object):
    """docstring for InputGenerator."""
    def __init__(self, gt_utils, chunk_size=5, batch_size=16, nb_steplag=0):
        self.gt_utils = gt_utils

        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.nb_steplag = nb_steplag

        self.nb_sample_train = self.gt_utils.idxs_train.shape[0]
        self.steps_per_epoch_train = np.floor(self.nb_sample_train / batch_size)

        self.nb_sample_valid = self.gt_utils.idxs_valid.shape[0]
        self.steps_per_epoch_valid = np.floor(self.nb_sample_valid / batch_size)

    """Settings fucntions"""
    #If needed
    def settings(self):
        pass

    """Transformation functions"""
    #If needed
    def mirrorPlanXZ(self, input, output):
        input_rev = np.array([  -1,  1, -1, -1,  1,
            -1,  1,  1,  1,  1, -1,  1, -1,  1, -1,
             1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
             1, -1,  1, -1,  1, -1,  1])
        output_rev = np.array([  -1,  1, -1,  1])
        return input * input_rev, output * output_rev


    """Processing fucntions"""
    def __resetEpoch(self, set='train'):
        if set == 'train':
            data = list(self.gt_utils.idxs_train.copy())
        elif set == 'valid':
            data = list(self.gt_utils.idxs_valid.copy())
        else:
            print('No mode set for InputGenerator: [`train` | `valid`]')
            exit(0)
        return data

    def __getBatch(self, data, batch_size):
        batch = []
        for _ in range(batch_size):
            idx = randint(0, len(data) - 1)
            batch.append(data[idx].copy())
            del data[idx]
        return data, batch

    def __getInputOutput(self, batch):
        idr, idf = batch

        # Has to use a custom stacking function
        input = [np.array(self.gt_utils.data[idr][idf][0]).copy()] # current frame only
        # step lag
        idf_o = idf + self.nb_steplag if len(self.gt_utils.data[idr]) < idf + self.nb_steplag else idf
        output = np.array(self.gt_utils.data[idr][idf_o][1]).copy() # current frame + recup les 5 d'avant
        if self.chunk_size == None:
            return input[0], output
        for _ in range(self.chunk_size - 1):
            idf = idf - 1 if idf > 0 else 0 # get previous frames
            input = np.concatenate(([np.array(self.gt_utils.data[idr][idf][0]).copy()], input), axis=0)
        # Encode
        input = np.float32(input.reshape(input.shape))
        output = np.float32(output.reshape(output.shape))
        input, output = self.gt_utils.data_formater.encode(input, output=output)
        if random() < 0.0:
            input, output = self.mirrorPlanXZ(input, output)
        return input, output

    def __preprocessInputOutput(self, frames, cmds):
        return frames, cmds

    def generate(self, set='train', debug=False):
        data = self.__resetEpoch(set)
        batch_size = self.batch_size

        while True:
            # Reset data if needed
            # print(len(data)) # DEBUG
            if len(data) < batch_size:
                if debug:
                    print("Reset") # DEBUG
                data = self.__resetEpoch(set)

            # Get batch data
            data, batch_data = self.__getBatch(data, batch_size)
            # print(batch_data)
            batch_input = []
            batch_output = []

            # Read in each input, perform preprocessing and get labels
            for val in batch_data:
                input, output = self.__getInputOutput(val)

                input, output = self.__preprocessInputOutput(frames=input, cmds=output)
                batch_input += [input]
                batch_output += [output]

            # Return a tuple of (input,output) to feed the network
            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield batch_x, batch_y

if __name__ == "__main__":
    import cPickle
    import time

    with open('../resources/data_killian.pkl', 'r') as f:
        gt = cPickle.load(f)

    print('Loading done')
    generator = InputGenerator(gt)

    for x, y in generator.generate():
        print(x[0], y[0])
        # time.sleep(10)
        pass
