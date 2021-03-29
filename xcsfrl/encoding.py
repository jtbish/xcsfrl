import abc

import numpy as np
from rlenvs.obs_space import IntegerObsSpace, RealObsSpace

from .hyperparams import get_hyperparam as get_hp
from .interval import Interval
from .rng import get_rng
from .condition import Condition

_GENERALITY_LB_EXCL = 0.0
_GENERALITY_UB_INCL = 1.0


class EncodingABC(metaclass=abc.ABCMeta):
    def __init__(self, obs_space):
        self._obs_space = obs_space

    @abc.abstractmethod
    def gen_covering_condition(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, cond_alleles):
        raise NotImplementedError

    @abc.abstractmethod
    def calc_condition_generality(self, cond_intervals):
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_condition_alleles(self, cond_alleles):
        raise NotImplementedError


class UnorderedBoundEncodingABC(EncodingABC, metaclass=abc.ABCMeta):
    def gen_covering_condition(self, obs):
        num_alleles = len(self._obs_space) * 2
        cond_alleles = []
        assert len(obs) == len(self._obs_space)
        for (obs_compt, dim) in zip(obs, self._obs_space):
            (lower, upper) = self._gen_covering_alleles(obs_compt, dim)
            cover_alleles = [lower, upper]
            # to avoid bias, insert alleles into genotype in random order
            get_rng().shuffle(cover_alleles)
            for allele in cover_alleles:
                cond_alleles.append(allele)
        assert len(cond_alleles) == num_alleles
        return Condition(cond_alleles, self)

    @abc.abstractmethod
    def _gen_covering_alleles(self, obs_compt, dim):
        raise NotImplementedError

    def decode(self, cond_alleles):
        phenotype = []
        assert len(cond_alleles) % 2 == 0
        for i in range(0, len(cond_alleles), 2):
            first_allele = cond_alleles[i]
            second_allele = cond_alleles[i + 1]
            lower = min(first_allele, second_allele)
            upper = max(first_allele, second_allele)
            phenotype.append(Interval(lower, upper))
        assert len(phenotype) == len(cond_alleles) // 2
        return phenotype

    @abc.abstractmethod
    def calc_condition_generality(self, cond_intervals):
        raise NotImplementedError

    def mutate_condition_alleles(self, alleles):
        assert len(alleles) % 2 == 0
        allele_pairs = [(alleles[i], alleles[i + 1])
                        for i in range(0, len(alleles), 2)]
        mut_alleles = []
        for (allele_pair, dim) in zip(allele_pairs, self._obs_space):
            for allele in allele_pair:
                if get_rng().random() < get_hp("p_mut"):
                    noise = self._gen_mutation_noise(dim)
                    sign = get_rng().choice([-1, 1])
                    mut_allele = allele + sign * noise
                    mut_allele = np.clip(mut_allele, dim.lower, dim.upper)
                    mut_alleles.append(mut_allele)
                else:
                    mut_alleles.append(allele)
        assert len(mut_alleles) == len(alleles)
        return mut_alleles

    @abc.abstractmethod
    def _gen_mutation_noise(self, dim=None):
        raise NotImplementedError


class IntegerUnorderedBoundEncoding(UnorderedBoundEncodingABC):
    def __init__(self, obs_space):
        assert isinstance(obs_space, IntegerObsSpace)
        super().__init__(obs_space)

    def _gen_covering_alleles(self, obs_compt, dim):
        r_nought = get_hp("r_nought")
        # rand integer ~ (0, r_nought)
        lower = obs_compt - get_rng().randint(low=1, high=r_nought)
        upper = obs_compt + get_rng().randint(low=1, high=r_nought)
        lower = max(lower, dim.lower)
        upper = min(upper, dim.upper)
        return (lower, upper)

    def calc_condition_generality(self, cond_intervals):
        # condition generality calc as in
        # Wilson '00 Mining Oblique Data with XCS
        numer = sum([(interval.upper - interval.lower + 1)
                     for interval in cond_intervals])
        denom = sum([(dim.upper - dim.lower + 1) for dim in self._obs_space])
        generality = numer / denom
        assert _GENERALITY_LB_EXCL < generality <= _GENERALITY_UB_INCL
        return generality

    def _gen_mutation_noise(self, dim=None):
        # integer ~ [1, m_0]
        return get_rng().randint(low=1, high=(get_hp("m_nought") + 1))


class RealUnorderedBoundEncoding(UnorderedBoundEncodingABC):
    def __init__(self, obs_space):
        assert isinstance(obs_space, RealObsSpace)
        super().__init__(obs_space)

    def _gen_covering_alleles(self, obs_compt, dim):
        raise NotImplementedError

    def calc_condition_generality(self, cond_intervals):
        numer = sum([(interval.upper - interval.lower)
                     for interval in cond_intervals])
        denom = sum([(dim.upper - dim.lower) for dim in self._obs_space])
        generality = numer / denom
        assert _GENERALITY_LB_EXCL < generality <= _GENERALITY_UB_INCL
        return generality

    def _gen_mutation_noise(self, dim):
        # m_0 interpreted as fraction of dim span to draw uniform random
        # noise from
        dim_span = (dim.upper - dim.lower)
        m_nought = get_hp("m_nought")
        assert 0.0 < m_nought <= 1.0
        mut_high = m_nought*dim_span
        return get_rng().uniform(low=0, high=mut_high)
