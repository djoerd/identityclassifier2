
import numpy

class TweetsBatcher:

    __slots__ = ["data", "target", "index"]

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.index = 0


    def reshuffle(self):
        index = numpy.arange(numpy.shape(self.data)[0])
        numpy.random.shuffle(index)
        self.data = self.data[index, :]
        self.target = self.target[index, :]
        self.index = 0
        #print "SHUFFLED!"


    def next_batch(self, size):
        total_size = numpy.shape(self.data)[0]
        if (self.index + size > total_size):
            size = total_size - self.index
        data_batch   = self.data[self.index: self.index + size]
        target_batch = self.target[self.index: self.index + size]
        self.index += size
        if (self.index >= total_size):
            self.reshuffle()
        return data_batch.todense(), target_batch.todense()
        
    
