from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np

from six.moves import range
from six import iteritems

def pearson(n_x, yr, min_support):
    # https://github.com/NicolasHug/Surprise/blob/711fb80748140c44e0ed870e573c735307e6c3cc/surprise/similarities.pyx
    """Compute the Pearson correlation coefficient between all pairs of users
    (or items).
    Only **common** users (or items) are taken into account. The Pearson
    correlation coefficient can be seen as a mean-centered cosine similarity,
    and is defined as:
    .. math ::
        \\text{pearson_sim}(u, v) = \\frac{ \\sum\\limits_{i \in I_{uv}}
        (r_{ui} -  \mu_u) \cdot (r_{vi} - \mu_{v})} {\\sqrt{\\sum\\limits_{i
        \in I_{uv}} (r_{ui} -  \mu_u)^2} \cdot \\sqrt{\\sum\\limits_{i \in
        I_{uv}} (r_{vi} -  \mu_{v})^2} }
    or
    .. math ::
        \\text{pearson_sim}(i, j) = \\frac{ \\sum\\limits_{u \in U_{ij}}
        (r_{ui} -  \mu_i) \cdot (r_{uj} - \mu_{j})} {\\sqrt{\\sum\\limits_{u
        \in U_{ij}} (r_{ui} -  \mu_i)^2} \cdot \\sqrt{\\sum\\limits_{u \in
        U_{ij}} (r_{uj} -  \mu_{j})^2} }
    depending on the ``user_based`` field of ``sim_options`` (see
    :ref:`similarity_measures_configuration`).
    Note: if there are no common users or items, similarity will be 0 (and not
    -1).
    For details on Pearson coefficient, see `Wikipedia
    <https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample>`__.
    """

    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] prods
    # sum (rxy ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqi
    # sum (rx'y ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqj
    # sum (rxy) for common ys
    cdef np.ndarray[np.double_t, ndim=2] si
    # sum (rx'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sj
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sprt = min_support

    freq = np.zeros((n_x, n_x), np.int)
    prods = np.zeros((n_x, n_x), np.double)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    si = np.zeros((n_x, n_x), np.double)
    sj = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                prods[xi, xj] += ri * rj
                freq[xi, xj] += 1
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2
                si[xi, xj] += ri
                sj[xi, xj] += rj

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):

            if freq[xi, xj] < min_sprt:
                sim[xi, xj] == 0
            else:
                n = freq[xi, xj]
                num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                denum = np.sqrt((n * sqi[xi, xj] - si[xi, xj]**2) *
                                (n * sqj[xi, xj] - sj[xi, xj]**2))
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = num / denum

            sim[xj, xi] = sim[xi, xj]

    return sim