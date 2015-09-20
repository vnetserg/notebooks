#!/usr/bin/python3.4

from inspect import ismethod

class BaseTransformer:
    def __init__(self, train, target, fulldata):
        for attr in dir(self):
            val = getattr(self, attr)
            if ismethod(val) and attr.startswith("init"):
                val(train, target, fulldata)
        self._codes = {}

    def apply(self, data, target=None):
        for attr in dir(self):
            if attr == "apply": continue
            val = getattr(self, attr)
            if ismethod(val) and attr.startswith("apply"):
                result = val(data, target)
                if result is not None:
                    data = result
        for column in self.do_encode:
            data = self.encode(data, column)
        for column in self.do_oneHot:
            data = self.oneHot(data, column)
        for column, function in self.do_rescale.items():
            data[column] = [function(x) for x in data[column]]
        return data.drop(self.do_drop, axis=1)

    def encode(self, data, column):
        values = data[column].drop_duplicates()
        if column in self._codes:
            mapping = self._codes[column]
            for val in values:
                if val not in mapping:
                    mapping[val] = len(mapping)
        else:
            mapping = {v: i for i, v in enumerate(sorted(values))}
            self._codes[column] = mapping
        data[column] = [mapping[v] for v in data[column]]
        return data

    def oneHot(self, data, column):
        values = data[column].drop_duplicates()
        for val in sorted(values):
            data.insert(len(data.columns), "{}_{}".format(column, val),
                        [1 if rowval == val else 0 for rowval in data[column]])
        return data.drop([column], axis=1)

    