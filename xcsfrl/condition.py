_SPAN_FRAC_MIN_INCL = 0
_SPAN_FRAC_MAX_INCL = 1


class Condition:
    def __init__(self, alleles, encoding):
        self._alleles = list(alleles)
        self._encoding = encoding
        self._phenotype = self._encoding.decode(self._alleles)
        self._generality = self._encoding.calc_condition_generality(
            self._phenotype)

        # try and be smart and use a heuristic to speed up does_match() method.
        # idea is this: the condition is a collection of intervals and each
        # dimension of an input obs must lie in the respective interval for the
        # obs to match the condition.
        # by default, would just iterate over intervals in a static order:
        # dim1, dim2, etc.
        # but... assuming that all points in the obs space are equally likely,
        # makes more sense to check the *smaller* intervals first (ones with
        # lower spans), might then be able to quickly rule out certain points,
        # especially if lowest span is quite small.
        # this will likely be more effective in saving time when the
        # dimensionality of the obs space is high
        self._matching_idx_order = \
            self._calc_matching_idx_order(self._phenotype,
                                          obs_space=self._encoding.obs_space)

    @property
    def alleles(self):
        return self._alleles

    @property
    def generality(self):
        return self._generality

    def _calc_matching_idx_order(self, phenotype, obs_space):
        # first calc "span fracs" of all intervals in phenotype relative to
        # each dim span
        assert len(phenotype) == len(obs_space)
        span_fracs_with_idxs = []
        for (idx, (interval, dim)) in enumerate(zip(phenotype, obs_space)):
            span_frac = (interval.span / dim.span)
            assert _SPAN_FRAC_MIN_INCL <= span_frac <= _SPAN_FRAC_MAX_INCL
            span_fracs_with_idxs.append((idx, span_frac))

        # then sort the span fracs in ascending order
        sorted_span_fracs_with_idxs = sorted(span_fracs_with_idxs,
                                             key=lambda tup: tup[1],
                                             reverse=False)
        matching_idx_order = [tup[0] for tup in sorted_span_fracs_with_idxs]
        assert len(matching_idx_order) == len(phenotype)
        return matching_idx_order

    def does_match(self, obs):
        for idx in self._matching_idx_order:
            interval = self._phenotype[idx]
            obs_val = obs[idx]
            if not interval.contains_val(obs_val):
                return False
        return True

    def does_subsume(self, other):
        """Does this condition subsume other condition?"""
        for (my_interval, other_interval) in zip(self._phenotype,
                                                 other._phenotype):
            if not my_interval.does_subsume(other_interval):
                return False
        return True

    def __eq__(self, other):
        # This was bugged and originally compared equality of alleles,
        # which is OK for 1 to 1 genotype to phenotype mappings but otherwise
        # not OK! Change it to instead compare phenotypic equality.
        return self._phenotype == other._phenotype

    def __len__(self):
        return len(self._phenotype)

    def __str__(self):
        return " && ".join([str(interval) for interval in self._phenotype])
