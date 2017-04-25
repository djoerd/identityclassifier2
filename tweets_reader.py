import csv
import re
import numpy

from scipy.sparse import csr_matrix

# See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

class TweetsReader:

    __slots__ = ["vocabulary", "train_data", "train_target", "test_data", "test_target"]

    def __init__(self, filename_train, filename_test):
        self.vocabulary   = {};

        data1, target1, d_indices1, t_indices1, d_indptr1, t_indptr1 = self._csv_to_matrix(filename_train)
        data2, target2, d_indices2, t_indices2, d_indptr2, t_indptr2 = self._csv_to_matrix(filename_test)

        x_train  = len(d_indptr1) - 1
        x_test   = len(d_indptr2) - 1 
        y_data   = len(self.vocabulary)
        y_target = 6

        self.train_data   = csr_matrix((data1,   d_indices1, d_indptr1), dtype=int, shape=(x_train, y_data))
        self.train_target = csr_matrix((target1, t_indices1, t_indptr1), dtype=int, shape=(x_train, y_target))
        self.test_data    = csr_matrix((data2,   d_indices2, d_indptr2), dtype=int, shape=(x_test,  y_data))
        self.test_target  = csr_matrix((target2, t_indices2, t_indptr2), dtype=int, shape=(x_test,  y_target))

        
    def get_train_data(self):
        return self.train_data, self.train_target


    def get_test_data(self):
        return self.test_data, self.test_target


    def _csv_to_matrix(self, filename):
        data, target = [], []
        d_indices, t_indices = [], []
        d_indptr, t_indptr = [0], [0]
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                text = row[2]   # the profile text
                cats = row[3:8] # the 5 identities
                if (text != 'from_user_description'):
                    for term in (re.split("[^a-z]+", text.lower())):
                        if (term != ''):
                            index = self.vocabulary.setdefault(term, len(self.vocabulary))
                            d_indices.append(index)
                            data.append(1)
                    d_indptr.append(len(d_indices))
                    found = 0
                    for index in range(len(cats)):
                        if (cats[index] == "1"):
                            t_indices.append(index)
                            target.append(1)
                            found = 1
                    if (found == 0): # add a 'no identity class' similar to column 6 in data
                        t_indices.append(len(cats))
                        target.append(1)
                    t_indptr.append(len(t_indices))
        return (data, target,
                d_indices, t_indices,
                d_indptr, t_indptr)

