# conquer

**Con**volution-type smoothed **qu**antil**e** **r**egression

## Description

The `conquer` library performs convolution-type smoothed quantile regression ([Fernandes, Guerre and Horta, 2019](https://www.tandfonline.com/doi/full/10.1080/07350015.2019.1660177)) implemented via Barzilai-Borwein gradient descent ([Barzilai and Borwein, 1988](https://doi.org/10.1093/imanum/8.1.141)) with a Huber warm start. It achieves conspicuously fast computational speed while maintaining estimation accuracy, comparing with classical quantile regression (see [here](https://www.jstor.org/stable/1913643?seq=1#metadata_info_tab_contents) for the seminal paper, [here](https://www.cambridge.org/core/books/quantile-regression/C18AE7BCF3EC43C16937390D44A328B1) for a monograph and [here](https://CRAN.R-project.org/package=quantreg) for its software). The package can also conduct statistical inference by multiplier bootstrap for conquer. 

## Installation

`conquer` can be installed into `R` environment from the GitHub repository:

```r
install.packages("devtools")
devtools::install_github("XiaoouPan/conquer")
library(conquer)
```

## Common error messages

Since `conquer` is currently a source package, some compiling issues may appear when users install it. We will submit it to CRAN soon.

First of all, to avoid most error messages, it is **strongly** recommended to update `R` to version >= 3.6.1.

Besides, since the library `conquer` is coded in `Rcpp` and `RcppArmadillo`, when you first install it, the instructions in [this page](https://thecoatlessprofessor.com/programming/cpp/r-compiler-tools-for-rcpp-on-macos/) may be very useful. Three components to the `R 3.6.x` toolchain need to be installed following the aforementioned instructions.

1. Rtools for Windows OS or XCode Command Line Tools for Mac OS. 

2. The `clang7` binary from the [Mac OS tools page](https://cran.r-project.org/bin/macosx/tools/).

3. The `gfortran6.1` binary from the [Mac OS tools page](https://cran.r-project.org/bin/macosx/tools/).

`conquer` should be working well after these steps. 

## Main function

The main functions of this library:

* `conquer`: Convolution-type smoothed quantile regression

## Getting help

Help on the functions can be accessed by typing `?`, followed by function name at the `R` command prompt. 

For example, `?conquer` will present a detailed documentation with inputs, outputs and examples of the function `conquer`.

## License

GPL-3.0

##  System requirements 

C++11

## Authors

Xiaoou Pan <xip024@ucsd.edu>, Kean Ming Tan <keanming@umich.edu> and Wen-Xin Zhou <wez243@ucsd.edu>

## Maintainer

Xiaoou Pan <xip024@ucsd.edu>

## References

Barzilai, J. and Borwein, J. M. (1988). Two-point step size gradient methods. *IMA J. Numer. Anal.*, **8** 141–148. [Paper](https://doi.org/10.1093/imanum/8.1.141)

Fernandes, M., Guerre, E. and Horta, E. (2019). Smoothing quantile regressions. *J. Bus. Econ. Statist.*, in press. [Paper](https://www.tandfonline.com/doi/full/10.1080/07350015.2019.1660177)

Koenker, R. (2005). Quantile Regression. Cambridge Univ. Press, Cambridge. [Book](https://www.cambridge.org/core/books/quantile-regression/C18AE7BCF3EC43C16937390D44A328B1)

Koenker, R. (2019). Package "quantreg", version 5.54. [CRAN](https://CRAN.R-project.org/package=quantreg)

Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica* **46** 33-50. [Paper](https://www.jstor.org/stable/1913643?seq=1#metadata_info_tab_contents)

