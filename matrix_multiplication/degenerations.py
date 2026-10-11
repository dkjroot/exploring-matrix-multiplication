"""Executable polynomial degenerations with explicitly paid exact recovery cost.

The CW formula is from Blaser and Lysikov, MFCS 2016, equation (2), after
multiplication by epsilon^3 to remove negative powers. We implement its algebra
independently. Recovering a coefficient uses exact rational interpolation,
never a small floating epsilon. Tensor powers keep one shared parameter, so
recovery overhead is linear in the power rather than exponential.
"""

from fractions import Fraction

from .tensors import coppersmith_winograd, polynomial_coefficient, tensor_power


def polynomial_product(a, b):
    """Convolve rational coefficient lists with our own elementary arithmetic."""
    if not a or not b or any(type(c) not in (int, Fraction) for c in (*a, *b)):
        raise ValueError("polynomials require nonempty exact rational coefficient lists")
    coefficients = [Fraction(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            coefficients[i+j] += x*y
    return tuple(coefficients)


def polynomial_tensor_product(dimensions_a, terms_a, dimensions_b, terms_b):
    """Tensor polynomial rank-one terms using the same formal parameter in both."""
    terms = []
    for first in terms_a:
        for second in terms_b:
            terms.append(tuple(tuple((i*width+j, polynomial_product(p, q))
                                     for i, p in left for j, q in right)
                               for left, right, width in zip(first, second, dimensions_b)))
    return tuple(x*y for x, y in zip(dimensions_a, dimensions_b)), tuple(terms)


def cw_family(q, power=1):
    """Generate P(e)=e^(3*power) CW_q^power + higher powers, in (q+2)^power terms.

    For one power, P is the sum of q terms
      e (x0+e xi)(y0+e yi)(z0+e zi),
    minus (x0+e^2 sum xi)(y0+e^2 sum yi)(z0+e^2 sum zi),
    plus (1-qe)(x0+e^3 xlast)(y0+e^3 ylast)(z0+e^3 zlast).
    Constant, linear, and quadratic tensor coefficients cancel. Degree is at
    most 10*power. The finite recovery bound is (10*power+1)*(q+2)^power,
    whereas the border bound is (q+2)^power; these are different quantities.
    """
    coppersmith_winograd(q)  # Validate q before allocating polynomial factors.
    if type(power) is not int or power < 1:
        raise ValueError("CW degeneration power must be a positive integer")
    terms = []
    for i in range(1, q+1):
        factor = ((0, (1,)), (i, (0, 1)))
        scaled = tuple((j, polynomial_product(poly, (0, 1))) for j, poly in factor)
        # Scaling just the third factor by e multiplies the rank-one term by e.
        # Polynomial tuples are ordered by ascending degree: (0,1) means e.
        terms.append((factor, factor, scaled))
    factor = ((0, (1,)),)+tuple((i, (0, 0, 1)) for i in range(1, q+1))
    # The next two terms cancel the unwanted low-degree coefficients. Minus
    # belongs to one factor only; changing all three signs happens to give a
    # minus too, but would obscure the general rule used by coefficient recovery.
    terms.append((factor, factor, tuple((i, tuple(-c for c in p)) for i, p in factor)))
    factor = ((0, (1,)), (q+1, (0, 0, 0, 1)))
    terms.append((factor, factor, tuple((i, polynomial_product(poly, (1, -q))) for i, poly in factor)))
    base_dimensions, base_terms = (q+2,)*3, tuple(terms)
    dimensions, powered = base_dimensions, base_terms
    for _ in range(1, power):
        # Use the SAME formal e for every factor of the tensor power. The
        # leading degree adds to 3*power and the maximum to 10*power. Independent
        # parameters would require multivariate recovery with a different cost.
        dimensions, powered = polynomial_tensor_product(dimensions, powered, base_dimensions, base_terms)
    return dimensions, powered, 10*power, 3*power


def recover_cw(q, power=1):
    """Recover the finite exact CW tensor power and check every tensor coefficient.

    The literal independent definition is the target, not an asymptotic rank
    annotation. Checking this materialized identity can be very expensive for
    larger powers, just as building the recipe is. This tensor is not itself a
    square matrix multiplication algorithm; variable extraction is still needed.
    """
    dimensions, terms, degree, leading = cw_family(q, power)
    recovered = polynomial_coefficient(dimensions, terms, degree, leading)
    expected = tensor_power(coppersmith_winograd(q), power)
    if recovered.expand() != expected.expand():
        raise AssertionError("CW coefficient recovery failed its complete tensor check")
    return recovered, {"q": q, "power": power, "polynomial_rank": len(terms),
                       "degree_bound": degree, "recovered_coefficient": leading,
                       "interpolation_points": degree+1,
                       "finite_rank_bound": (degree+1)*len(terms),
                       "finite_recipe_products": recovered.rank,
                       "literal_recipe_products": expected.rank,
                       "exact_tensor_identity_verified": True,
                       "matrix_multiplication_recipe": False,
                       "source": "https://drops.dagstuhl.de/opus/volltexte/2016/6434/pdf/LIPIcs-MFCS-2016-19.pdf"}
