#' @docType package
#' @name conquer-package
#' @title Conquer: Convolution-Type Smoothed Quantile Regression
#' @description Estimation and inference for conditional linear quantile regression models using a convolution smoothed approach. 
#' In the low-dimensional setting, efficient gradient-based methods are employed for fitting both a single model and a regression process over a quantile range. Normal-based and (multiplier) bootstrap confidence intervals for all slope coefficients are constructed. 
#' In high dimensions, the conquer methods complemented with \eqn{\ell_1}-penalization and iteratively reweighted \eqn{\ell_1}-penalization are used to fit sparse models.
#' Commonly used penalities, such as the elastic-net, group lasso and sparse group lasso, are also incorporated to deal with more complex low-dimensional structures.
#' @author Xuming He <xmhe@umich.edu>, Xiaoou Pan <xip024@ucsd.edu>, Kean Ming Tan <keanming@umich.edu>, and Wen-Xin Zhou <wez243@ucsd.edu>
#' @references Barzilai, J. and Borwein, J. M. (1988). Two-point step size gradient methods. IMA J. Numer. Anal., 8, 141–148.
#' @references Belloni, A. and Chernozhukov, V. (2011). \eqn{\ell_1} penalized quantile regression in high-dimensional sparse models. Ann. Statist., 39, 82-130.
#' @references Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist., 46, 814-841.
#' @references Fernandes, M., Guerre, E. and Horta, E. (2021). Smoothing quantile regressions. J. Bus. Econ. Statist., 39, 338-357.
#' @references He, X., Pan, X., Tan, K. M., and Zhou, W.-X. (2022+). Smoothed quantile regression for large-scale inference. J. Econometrics, in press.
#' @references Koenker, R. (2005). Quantile Regression. Cambridge University Press, Cambridge.
#' @references Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica, 46, 33-50.
#' @references Tan, K. M., Wang, L. and Zhou, W.-X. (2022). High-dimensional quantile regression: convolution smoothing and concave regularization. J. Roy. Statist. Soc. Ser. B, 84(1), 205-233.
NULL
