"""Sensor class code"""
import ujson  # ultra fast json encoder
import time
import os


class Sensors(object):
    """docstring for Sensors."""

    def __init__(self, directory="sensors_logs"):
        self.__sensorlogged__ = []
        self.directory = directory
        self.filename = None
        self._i = 0
        self.file = None
        self.mem = {}

    def add(self, topic, val=None):
        if not hasattr(self, topic):
            self.__sensorlogged__.append(topic)
        setattr(self, topic, val)

    def get_all(self):
        data = {}
        for sensor in self.__sensorlogged__:
            val = getattr(self, sensor)
            if hasattr(val, "value"):
                val = val.value
            data[sensor] = val
        return data

    def dumpMem(self):
        self.mem.update({str(self._i): self.get_all()})
        self._i += 1

    def saveMem(self):
        self.filename = "{}/{}.json".format(
            self.directory, time.strftime("%Y%m%d%H%M%S")
        )
        with open(self.filename, "w+") as f:
            ujson.dump(self.mem, f, indent=4)

    def endMem(self, status=True):
        if status:
            self.saveMem()
        del self.mem
        self.mem = {}

    def dumpJSON(self):
        if not self.file:
            self.filename = "{}/{}.json".format(
                self.directory, time.strftime("%Y%m%d%H%M%S")
            )
            self.file = open(self.filename, "w+")
            self.file.write("{\n")
        if self.file:
            if self._i != 0:
                self.file.write(",\n")
            self.file.write('\t"{}":'.format(self._i))
            ujson.dump(self.get_all(), self.file, indent=4)
            self._i += 1

    def endRecording(self, status=True):
        self.closeJSON()
        if not status:
            os.system("rm -f {}".format(self.filename))

    def closeJSON(self):
        if self.file:
            self.file.write("}")
            self.file.close()
            self.file = None
            self._i = 0


if __name__ == "__main__":
    import multiprocessing as mp

    s = Sensors()

    s.add("maxime_z", 1)
    s.add("tristan", 2)
    s.add("baptiste", mp.Value("i", 5))

    data = s.get_all()

    print(data)
    for i in range(10):
        s.maxime_z = i
        s.dumpJSON()
    s.closeJSON()

    with open(s.filename, "r") as f:
        data = ujson.load(f)
        print(data["0"])
