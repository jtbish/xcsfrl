class Population:
    """Population is just a list of macroclassifiers with tracking of the number of
    microclassifiers, in order to avoid having to calculate this number on the
    fly repeatedly and waste time."""
    def __init__(self):
        self._clfrs = []
        self._num_micros = 0
        self._ops_history = {
            "covering": 0,
            "absorption": 0,
            "insertion": 0,
            "deletion": 0,
            "ga_subsumption": 0,
            "as_subsumption": 0
        }

    @property
    def num_macros(self):
        return len(self._clfrs)

    @property
    def num_micros(self):
        return self._num_micros

    @property
    def ops_history(self):
        return self._ops_history

    def add_new(self, clfr, op):
        self._clfrs.append(clfr)
        self._num_micros += clfr.numerosity
        assert op in ("covering", "insertion")
        self._ops_history[op] += clfr.numerosity

    def alter_numerosity(self, clfr, delta, op):
        clfr.numerosity += delta
        self._num_micros += delta
        assert op in ("absorption", "deletion", "ga_subsumption",
                      "as_subsumption")
        # delta can be neg. (obviously), but op. counts are pos.
        self._ops_history[op] += abs(delta)

    def remove(self, clfr, op=None):
        self._clfrs.remove(clfr)
        self._num_micros -= clfr.numerosity
        if op is not None:
            assert op == "deletion"
            self._ops_history[op] += clfr.numerosity

    def __iter__(self):
        return iter(self._clfrs)

    def __getitem__(self, idx):
        return self._clfrs[idx]
