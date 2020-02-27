# conquer

**Con**volution-type smoothed **qu**antil**e** **r**egression

## Description

The `conquer` library performs convolution-type smoothed quantile regression implemented via Barzilai-Borwein gradient descent. It achieves conspicuously fast computational speed while maintaining estimation accuracy, comparing with classic quantile regression. The package can also conduct statistical inference by multiplier bootstrap for quantile regression. 

## Installation

`conquer` can be installed into `R` environment from the GitHub repository:

```r
install.packages("devtools")
library(devtools)
devtools::install_github("XiaoouPan/conquer")
library(conquer)
```

We will submit it to CRAN soon.

## Common error messages

First of all, to avoid most unexpected error messages, it is **strongly** recommended to update `R` to version >= 3.6.1.

Besides, since the library `conquer` is coded in `Rcpp` and `RcppArmadillo`, when you first install it, the following two build tools are required:

1. Rtools for Windows OS or XCode Command Line Tools for Mac OS. See [this link](https://support.rstudio.com/hc/en-us/articles/200486498-Package-Development-Prerequisites) for details.

2. gfortran binaries: see [here](https://gcc.gnu.org/wiki/GFortranBinaries#MacOS) for instructions.

`conquer` should be working well after these steps. Some common error messages along with their solutions are collected below, and we'll keep updating them based on users' feedback:

* Error: "...could not find build tools necessary to build FarmTest": Please see step 1 above.

* Error: "library not found for -lgfortran/..": Please see step 2 above.
    
* Error: "cannot remove prior installation of package 'Rcpp'": This issue happens occasionally when you have installed an old version of the package `Rcpp` before. Updating `Rcpp` with command `install.packages("Rcpp")` will solve the problem.

## Main function

The main functions of this library:

* `conquer`: Convolution-type smoothed quantile regression

## Getting help

Help on the functions can be accessed by typing `?`, followed by function name at the `R` command prompt. 

For example, `?conquer` will present a detailed documentation with inputs, outputs and examples of the function `conquer`.

## License

GPL-3.0

##  SystemRequirements 

C++11

## Authors

Xiaoou Pan, Kean Ming Tan and Wen-Xin Zhou

## Maintainer

Xiaoou Pan <xip024@ucsd.edu>

## References

Barzilai, J. and Borwein, J. M. (1988). Two-point step size gradient methods. *IMA J. Numer. Anal.*, **8** 141–148. [Paper](https://academic.oup.com/imajna/article-abstract/8/1/141/802460)

Fernandes, M., Guerre, E. and Horta, E. (2019). Smoothing quantile regressions. *J. Bus. Econ. Statist.*, in press. [Paper](https://www.tandfonline.com/doi/full/10.1080/07350015.2019.1660177)

Koenker, R. (2005). Quantile Regression. Cambridge Univ. Press, Cambridge. [Book](https://www.cambridge.org/core/books/quantile-regression/C18AE7BCF3EC43C16937390D44A328B1)

Koenker, R. (2019). Package "quantreg". [CRAN](https://cran.r-project.org/web/packages/quantreg/index.html)

Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica* **46** 33-50. [Paper](https://www.jstor.org/stable/1913643?seq=1#metadata_info_tab_contents)

