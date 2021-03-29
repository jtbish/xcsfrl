class Condition:
    def __init__(self, alleles, encoding):
        self._alleles = list(alleles)
        self._encoding = encoding
        self._phenotype = self._encoding.decode(self._alleles)

    @property
    def alleles(self):
        return self._alleles

    def does_match(self, obs):
        for (interval, obs_val) in zip(self._phenotype, obs):
            if not interval.contains_val(obs_val):
                return False
        return True

    def calc_generality(self):
        return self._encoding.calc_condition_generality(self._phenotype)

    def __len__(self):
        return len(self._pheontype)

    def __str__(self):
        return " && ".join([str(interval) for interval in self._phenotype])
