package marcum

import (
	gsl "github.com/jtejido/ggsl"
	gslerr "github.com/jtejido/ggsl/err"
	sf "github.com/jtejido/ggsl/specfunc"
	"math"
)

func init() {
	gslerr.SetErrorHandlerOff()
}

func MarcumQ(mu, alpha, beta float64) float64 {
	if mu <= 0 || alpha < 0 || beta < 0 {
		panic("Invalid Inputs")
	}

	// Check for the special case of a zero alpha. This will work for
	if alpha == 0 {
		// If alpha is zero, the MarcumQ function reduces to a regularized gamma
		// function. This identity is programmed into Mathematica.
		return 1 - sf.Gamma_inc(math.Pow(beta, 2)/2, mu)
	}

	// Use Ross' algorithm if mu is of the form integer+0.5 and the precision
	// bounds are satisfied.
	fix := int(mu)
	diff := mu - float64(fix)
	if (diff == 0.5) || (diff == 0) {
		return benton(mu, alpha, beta)
	}

	// If mu is not a multiple of 0.5, then use equivalency of the Marcum Q
	// function with the noncentral chi square CDF and therefore the noncentral
	// gamma CDF to compute the result.
	return 1 - ncGammaDis(math.Pow(beta, 2), mu, 2, math.Pow(alpha, 2))
}

// This function implements the algorithm of [1], whose code is given in
// Section 7.3 as Algorithm 7.3. Some of the comments
// are taken from the paper. The algorithm will evaluate the MarcumQ
// function at integer values of mu as well as values of mu that are integers
// +0.5. Changes from the original code are the substitution of expFunc(y,M)
// from [2] for the code that computed terms of the form exp(-y)*y^M/M! to
// avoid overflow/underflow errors. Also, the error tolerance is set to
// eps(sumVal) and some code for the fact that we want Q and not 1-Q is
// added as well as the translation of the input parameters
// into the form for the noncentral Chi-Squared distribution.
//
// REFERENCES:
// [1] D. Benton and K. Krishnamoorthy, "Computing Discrete Mixtures of
//     Continuous Distributions: Noncentral Chisquare, Noncentral t and the
//     Distribution of the Square of the Sample Multiple Correlation
//     Coefficient," Computational Statistics & Data Analysis, vol. 43, no.
//     2, pp.249-26, 28 Jun. 2003.
// [2] D. A. Shnidman, "Note on "The Calculation of the Probability of
//     Detection and the Generalized Marcum Q-Function"," IEEE Transactions
//     on Information Theory, vol. 37, no. 4. pg. 1233, Jul. 1991.

func benton(mu, alpha, beta float64) float64 {
	maxIter := 5000
	n := 2 * mu
	lambda := alpha * alpha
	y := beta * beta

	x := y / 2
	del := lambda / 2
	kn := int(del)
	k := float64(kn)
	a := n/2 + k
	// Compute the gamma distribution function using (4.5) at (x; a), and assign
	// it to "gamkf" and "gamkb" so that they can be called laterforfor ward as
	// well as backward computations:
	gamkf := sf.Gamma_inc(x, a)
	gamkb := gamkf

	if lambda == 0 {
		return 1 - gamkf
	}

	// Compute the Poisson probability at (k; del) and assign it to "poikf" and
	// "poikb" so that they can be used as initial values for forward and
	// backward recursions:
	poikf := expFunc(del, k)
	poikb := poikf
	// "xtermf" is an initialization to compute the second term in (4.3)
	// recursively:
	xtermf := expFunc(x, a-1)
	// "xtermb" is an initialization to compute the second term in (4.4)
	// recursively:
	xtermb := xtermf * x / a
	sumVal := poikf * gamkf
	remain := 1 - poikf
	i := 1.0

	for {
		xtermf = xtermf * x / (a + i - 1)
		gamkf = gamkf - xtermf
		poikf = poikf * del / (k + i)
		sumVal = sumVal + poikf*gamkf
		errorVal := remain * gamkf
		remain = remain - poikf

		// Do forward and backward computations k times or until convergence:
		if i > k {
			if errorVal <= ulp(sumVal) {
				break
			} else if i > float64(maxIter) {
				// warning('Maximum number of iterations Exceeded. Unable to meet the error tolerance.')
				break
			} else {
				i = i + 1
			}
		} else {
			xtermb = xtermb * (a - i + 1) / x
			gamkb = gamkb + xtermb
			poikb = poikb * (k - i + 1) / del
			sumVal = sumVal + gamkb*poikb
			remain = remain - poikb
			if remain <= ulp(sumVal) {
				break
			} else if i > float64(maxIter) {
				// warning('Maximum number of iterations Exceeded. Unable to meet the error tolerance.')
				break
			} else {
				i = i + 1
			}
		}
	}

	P := sumVal
	// The max deals with negative values within precision bounds.
	Q := math.Max(1-P, 0)
	// The min deals with positive values within precision bounds.
	return math.Min(Q, 1)

}

// Evaluate the cumulative distribution function of the noncentral gamma distribution at desired points.
//
// This is an implementation of the distribution computations from [1]. The
// work in [1] concerns algorithms that implement the distribution
// approximations in [2]. This function implements the algorithm for
// computation of the cumulative distribution function. When lambda is zero,
// the central gamma case, a simple explicit formula is available.
//
// REFERENCES:
// [1] I. R. C. de Oliveria and D. F. Ferreira, "Computing the noncentral
//     gamma distribution, its inverse and the noncentrality parameter,"
//     Computational Statistics, vol. 28, no. 4, pp.1663-1680, 01 Aug 2013.
// [2] L. Knüsel and B. Bablok, "Computation of the noncentral gamma
//     distribution," SIAM Journal on Scientific Computing, vol. 17, no. 5,
//     pp.1224-1231, Sep. 1996.
//
// from TrackerComponentLibrary in Distributions/GammaD.m. December 2014 David A. Karnick, Naval Research Laboratory, Washington D.C.
func ncGammaDis(x, k, theta, lambda float64) float64 {
	maxIter := 5000

	x = x / theta
	m := math.Ceil(lambda)
	a := k + m
	gammap := sf.Gamma_inc_P(a, x)
	gammar := gammap
	gxr := math.Pow(x, a) * math.Exp(-x) / sf.Gamma(a+1) / theta
	var gxp float64
	if x != 0 {
		gxp = gxr * a / x
	}

	pp := math.Exp(-lambda) * math.Pow(lambda, m) / sf.Gamma(m+1)
	pr := pp
	remain := 1 - pp
	ii := 1.0
	cdf := pp * gammap

	for {
		gxp = gxp * x / (a + ii - 1)
		gammap = gammap - gxp
		pp = pp * lambda / (m + ii)
		cdf = cdf + pp*gammap
		er := remain * gammap
		remain = remain - pp
		if ii > m {
			if er <= gsl.Float64Eps || int(ii) > maxIter {
				break
			}
		} else {
			gxr = gxr * (a - ii) / x
			gammar = gammar + gxr
			pr = pr * (m - ii + 1) / lambda
			cdf = cdf + pr*gammar
			remain = remain - pr
			if remain <= gsl.Float64Eps || int(ii) > maxIter {
				break
			}

		}

		ii++
	}

	return cdf
}

// This function evaluates exp(-y)*y^M/M! while trying to avoid
// underflows. This function is used by [1] when imp[lementing the MarcumQ
// function as well as in the function AlgorithmBenton in this file.
//
// REFERENCES:
// [1] D. A. Shnidman, "Note on "The Calculation of the Probability of
//     Detection and the Generalized Marcum Q-Function"," IEEE Transactions
//     on Information Theory, vol. 37, no. 4. pg. 1233, Jul. 1991.
func expFunc(y, M float64) float64 {
	if y > 0 {
		if y < 10000 {
			return math.Exp(-y + M*math.Log(y) - sf.Lngamma(M+1))
		} else {
			// This is the expression from the 1991 Shidman paper. It is
			// supposed to be particularly accurate when y is very large.
			// Note that in the expression for A in the paper, the terms
			// z+1/2 and 1+1/(2*z) have the wrong sign in front of the z's.
			// It is not always used, because it is not as precise when y is
			// not large.
			z := M + 1
			A := (z-0.5)*((1-y/z)/(1-1/(2*z))+math.Log(y/z)) - 0.5*math.Log(2*math.Pi*y) - j(z)
			return math.Exp(A)
		}
	}
	// Deal with the zero exponent case.
	if M == 0 {
		return 1
	}

	return 0
}

// This is the Binet series from the Shidman papers. In the event
// that one moves up to higher precision arithmetic, it might be
// necessary to add more terms.
func j(z float64) float64 {
	return 1. / (12*z + 2./(5*z+53./(42*z+1170./(53*z+53./z))))
}

func ulp(x float64) float64 {
	if math.IsInf(x, 1) || math.IsInf(x, -1) {
		return math.Inf(1)
	}

	return math.Abs(x - math.Float64frombits(math.Float64bits(x)^1))
}
