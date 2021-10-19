# conquer

**Con**volution-type smoothed **qu**antil**e** **r**egression

## Description

The `conquer` library performs fast and accurate convolution-type smoothed quantile regression ([Fernandes, Guerre and Horta, 2021](https://www.tandfonline.com/doi/full/10.1080/07350015.2019.1660177), [He et al., 2021](https://doi.org/10.1016/j.jeconom.2021.07.010)) implemented via Barzilai-Borwein gradient descent ([Barzilai and Borwein, 1988](https://doi.org/10.1093/imanum/8.1.141)) with a Huber regression warm start. The package can also construct confidence intervals for regression coefficients using multiplier bootstrap.

## Installation

`conquer` is available on [CRAN](https://CRAN.R-project.org/package=conquer), and it can be installed into `R` environment:

```r
install.packages("conquer")
```

## Common errors or warnings

A collection of error / warning messages and their solutions:

* Error: smqr.cpp: 'quantile' is not a member of 'arma’. **Solution**: 'quantile' function was added into `RcppArmadillo` version 0.9.850.1.0 (2020-02-09), so reinstalling / updating the library `RcppArmadillo` will fix this issue.

* Error: unable to load shared object.. Symbol not found: _EXTPTR_PTR. **Solution**: This issue is common in some specific versions of `R` when we load Rcpp-based libraries. It is an error in R caused by a minor change about `EXTPTR_PTR`. Upgrading R to 4.0.2 will solve the problem.


## Examples

Let us illustrate conquer by a simple example. For sample size *n = 5000* and dimension *p = 70*, we generate data from a linear model *y<sub>i</sub> = &beta;<sub>0</sub> + <x<sub>i</sub>, &beta;> + &epsilon;<sub>i</sub>*, for *i = 1, 2, ... n*. Here we set *&beta;<sub>0</sub> = 1*, *&beta;* is a *p*-dimensional vector with every entry being *1*, *x<sub>i</sub>* follows *p*-dimensional standard multivariate normal distribution (available in the library `MASS`), and *&epsilon;<sub>i</sub>* is from *t<sub>2</sub>* distribution. 

```r
library(MASS)
library(quantreg)
library(conquer)
n = 5000
p = 70
beta = rep(1, p + 1)
set.seed(2020)
X = mvrnorm(n, rep(0, p), diag(p))
err = rt(n, 2)
Y = cbind(1, X) %*% beta + err
```

Then we run both quantile regression using package `quantreg`, with a Frisch-Newton approach after preprocessing ([Portnoy and Koenker, 1997](https://projecteuclid.org/euclid.ss/1030037960)), and conquer (with Gaussian kernel) on the generated data. The quantile level *&tau;* is fixed to be *0.5*. 

```r
tau = 0.5
start = Sys.time()
fit.qr = rq(Y ~ X, tau = tau, method = "pfn")
end = Sys.time()
time.qr = as.numeric(difftime(end, start, units = "secs"))
est.qr = norm(as.numeric(fit.qr$coefficients) - beta, "2")

start = Sys.time()
fit.conquer = conquer(X, Y, tau = tau)
end = Sys.time()
time.conquer = as.numeric(difftime(end, start, units = "secs"))
est.conquer = norm(fit.conquer$coeff - beta, "2")
```

It takes 0.1955 seconds to run the standard quantile regression but only 0.0255 seconds to run conquer. In the meanwhile, the estimation error is 0.1799 for quantile regression and 0.1685 for conquer. For readers’ reference, these runtimes are recorded on a Macbook Pro with 2.3 GHz 8-Core Intel Core i9 processor, and 16 GB 2667 MHz DDR4 memory.

## Getting help

Help on the functions can be accessed by typing `?`, followed by function name at the `R` command prompt. 

For example, `?conquer` will present a detailed documentation with inputs, outputs and examples of the function `conquer`.

## License

GPL-3.0

## System requirements 

C++11

## Authors

Xuming He <xmhe@umich.edu>, Xiaoou Pan <xip024@ucsd.edu>, Kean Ming Tan <keanming@umich.edu> and Wen-Xin Zhou <wez243@ucsd.edu>

## Maintainer

Xiaoou Pan <xip024@ucsd.edu>

## References

Barzilai, J. and Borwein, J. M. (1988). Two-point step size gradient methods. *IMA J. Numer. Anal.* **8** 141-148. [Paper](https://doi.org/10.1093/imanum/8.1.141)

Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist. 46 814-841. [Paper](https://projecteuclid.org/euclid.aos/1522742437)

Fernandes, M., Guerre, E. and Horta, E. (2021). Smoothing quantile regressions. *J. Bus. Econ. Statist.* **39** 338-357, [Paper](https://www.tandfonline.com/doi/full/10.1080/07350015.2019.1660177)

He, X., Pan, X., Tan, K. M., and Zhou, W.-X. (2021+). Smoothed quantile regression with large-scale inference. *J. Econometrics*, to appear. [Paper](https://doi.org/10.1016/j.jeconom.2021.07.010)

He, X., Pan, X., Tan, K. M., and Zhou, W.-X. (2021). Scalable estimation and inference for censored quantile regression process. Preprint.

Horowitz, J. L. (1998). Bootstrap methods for median regression models. *Econometrica* **66** 1327–1351. [Paper](https://www.jstor.org/stable/2999619)

Koenker, R. (2005). Quantile Regression. Cambridge Univ. Press, Cambridge. [Book](https://www.cambridge.org/core/books/quantile-regression/C18AE7BCF3EC43C16937390D44A328B1)

Koenker, R. (2019). Package "quantreg", version 5.54. [CRAN](https://CRAN.R-project.org/package=quantreg)

Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica* **46** 33-50. [Paper](https://www.jstor.org/stable/1913643?seq=1#metadata_info_tab_contents)

Peng, L. and Huang, Y. (2008). Survival analysis with quantile regression models. *J. Am. Stat. Assoc.* **103** 637–649. [Paper](https://doi.org/10.1198/016214508000000355)

Portnoy, S. and Koenker, R. (1997). The Gaussian hare and the Laplacian tortoise: Computability of squared-error versus absolute-error estimators. *Statist. Sci.* **12** 279–300. [Paper](https://projecteuclid.org/euclid.ss/1030037960)

Sanderson, C. and Curtin, R. (2016). Armadillo: A template-based C++ library for linear algebra. *J. Open Source Softw.* **1** 26. [Paper](https://joss.theoj.org/papers/10.21105/joss.00026.pdf)

Tan, K. M., Wang, L. and Zhou, W.-X. (2021). High-dimensional quantile regression: convolution smoothing and concave regularization. [arXiv](https://arxiv.org/abs/2109.05640)

