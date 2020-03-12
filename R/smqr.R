getPivCI = function(est, estBoot, alpha) {
  q1 = rowQuantiles(estBoot, probs = alpha / 2)
  q2 = rowQuantiles(estBoot, probs = 1 - alpha / 2)
  perCI = cbind(q1, q2)
  pivCI = cbind(2 * est - q2, 2 * est - q1)
  colnames(perCI) = colnames(pivCI) = c("lower", "upper")
  return (list(perCI = perCI, pivCI = pivCI))
}

getNormCI = function(est, sd, z) {
  lower = est - z * sd
  upper = est + z * sd
  return (cbind(lower, upper))
}

#' @title Convolution-type smoothed quantile regression
#' @description Smoothed quantile regression with fast computation and accurate estimation.
#' @param X The design matrix with dimension \eqn{n} by \eqn{p}, where \eqn{p < n}.
#' @param Y The response vector with length \eqn{n}.
#' @param tau (\strong{optional}) The desired quantile level of the regression problem. The value must be in \eqn{(0, 1)}.
#' @param kernel (\strong{optional}) A character string specifying the kernel function. Choices include "Gaussian" (default), "uniform", "parabolic" or "triangular".
#' @param h (\strong{optional}) The bandwidth of kernel smoothing. The value will be \eqn{max{((log(n) + p) / n)^0.4, 0.05}} without specific input or if the input value \eqn{< 0.05}.
#' @param standardize (\strong{optional}) A logical flag. If \code{standardize = TRUE}, then the design matrix will be standardized so that each column has \eqn{0} mean and unit standard deviation.
#' @param tol (\strong{optional}) Tolerance level of gradient descent. The gradient descent algorithm stops when the maximal entry of gradient \eqn{<} \code{tol}.
#' @param iteMax (\strong{optional}) Maximal iteration number for gradient descent. 
#' @param ci (\strong{optional}) A logical flag. If \code{ci = TRUE}, then three types of confidence intervals (percentile, pivotal and normal) will be constructed via multiplier bootstrap.
#' @param alpha (\strong{optional}) The nominal noncoverage probability for the confidence intervals. The value must be in \eqn{(0, 1)}.
#' @param B (\strong{optional}) The size of bootstrap sample.
#' @return An object containing the following items will be returned:
#' \describe{
#' \item{\code{coeff}}{The estimated coefficients including the intercept. A \eqn{(p + 1)}-vector.}
#' \item{\code{ite}}{The total number of iteration of the gradient descent algorithm.}
#' \item{\code{residual}}{The residuals of the fitted quantile regression.}
#' \item{\code{bandwidth}}{The bandwidth value.}
#' \item{\code{tau}}{The desired quantile level of the regression problem.}
#' \item{\code{kernel}}{The choice of kernel function.}
#' \item{\code{n}}{The sample size.}
#' \item{\code{p}}{The dimension.}
#' \item{\code{perCI}}{The percentile confidence intervals for regression coefficients. Not available if \code{ci = FALSE}}
#' \item{\code{pivCI}}{The pivotal confidence intervals for regression coefficients. Not available if \code{ci = FALSE}}
#' \item{\code{normCI}}{The normal-based confidence intervals for regression coefficients. Not available if \code{ci = FALSE}}
#' }
#' @references Barzilai, J. and Borwein, J. M. (1988). Two-point step size gradient methods. IMA J. Numer. Anal. 8 141–148.
#' @references Fernandes, M., Guerre, E. and Horta, E. (2019). Smoothing quantile regressions. J. Bus. Econ. Statist., in press.
#' @references Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica 46 33-50.
#' @author Xiaoou Pan <xip024@ucsd.edu>
#' @author Kean Ming Tan <keanming@umich.edu>
#' @author Wen-Xin Zhou <wez243@ucsd.edu>
#' @examples 
#' n = 500; p = 10
#' beta = rep(1, p)
#' X = matrix(rnorm(n * p), n, p)
#' Y = 1 + X %*% beta + rt(n, 2)
#' 
#' ## Smoothed quantile regression with Gaussian kernel
#' fit.Gauss = conquer(X, Y, tau = 0.5, kernel = "Gaussian")
#' beta.hat.Gauss = fit.Gauss$coeff
#' 
#' ## Smoothe quantile regression with uniform kernel
#' fit.unif = conquer(X, Y, tau = 0.5, kernel = "uniform")
#' beta.hat.unif = fit.unif$coeff
#' 
#' ## Construct three types of confidence intervals via multiplier bootstrap
#' fit = conquer(X, Y, tau = 0.5, kernel = "Gaussian", ci = TRUE)
#' ci.per = fit$perCI
#' ci.piv = fit$pivCI
#' ci.norm = fit$normCI
#' @export 
conquer = function(X, Y, tau = 0.5, kernel = c("Gaussian", "uniform", "parabolic", "triangular"), h = 0.0, standardize = TRUE, tol = 0.00001, 
                   iteMax = 5000, ci = FALSE, alpha = 0.05, B = 1000) {
  if (nrow(X) != length(Y)) {
    stop("Error: the length of Y must be the same as the number of rows of X.")
  }
  if (ncol(X) >= nrow(X)) {
    stop("Error: the number of columns of X cannot exceed the number of rows of X.")
  }
  if(tau <= 0 || tau >= 1) {
    stop("Error: the quantile level tau must be in (0, 1).")
  }
  if (alpha <= 0 || alpha >= 1) {
    stop("Error: the nominal level alpha must be in (0, 1).")
  }
  if (min(colSds(X)) == 0) {
    stop("Error: at least one column of X is constant.")
  }
  kernel = match.arg(kernel)
  if (!ci) {
    rst = NULL
    if (kernel == "Gaussian") {
      if (standardize) {
        rst = smqrGauss(X, Y, tau, h, tol = tol, iteMax = iteMax)
      } else {
        rst = smqrGaussNsd(X, Y, tau, h, tol = tol, iteMax = iteMax)
      }
    } else if (kernel == "uniform") {
      if (standardize) {
        rst = smqrUnif(X, Y, tau, h, tol = tol, iteMax = iteMax)
      } else {
        rst = smqrUnifNsd(X, Y, tau, h, tol = tol, iteMax = iteMax)
      }
    } else if (kernel == "parabolic") {
      if (standardize) {
        rst = smqrPara(X, Y, tau, h, tol = tol, iteMax = iteMax)
      } else {
        rst = smqrParaNsd(X, Y, tau, h, tol = tol, iteMax = iteMax)
      }
    } else {
      if (standardize) {
        rst = smqrTrian(X, Y, tau, h, tol = tol, iteMax = iteMax)
      } else {
        rst = smqrTrianNsd(X, Y, tau, h, tol = tol, iteMax = iteMax)
      }
    }
    return (list(coeff = as.numeric(rst$coeff), ite = rst$ite, residual = as.numeric(rst$residual), bandwidth = rst$bandwidth, tau = tau, 
                 kernel = kernel, n = nrow(X), p = ncol(X)))
  } else {
    rst = coeff = multiBeta = NULL
    if (kernel == "Gaussian") {
      rst = smqrGauss(X, Y, tau, h, tol = tol, iteMax = iteMax)
      coeff = as.numeric(rst$coeff)
      multiBeta = smqrGaussInf(X, Y, coeff, nrow(X), ncol(X), tau, B, tol, iteMax)
    } else if (kernel == "uniform") {
      rst = smqrUnif(X, Y, tau, h, tol = tol, iteMax = iteMax)
      coeff = as.numeric(rst$coeff)
      multiBeta = smqrUnifInf(X, Y, coeff, nrow(X), ncol(X), tau, B, tol, iteMax)
    } else if (kernel == "parabolic") {
      rst = smqrPara(X, Y, tau, h, tol = tol, iteMax = iteMax)
      coeff = as.numeric(rst$coeff)
      multiBeta = smqrParaInf(X, Y, coeff, nrow(X), ncol(X), tau, B, tol, iteMax)
    } else {
      rst = smqrTrian(X, Y, tau, h, tol = tol, iteMax = iteMax)
      coeff = as.numeric(rst$coeff)
      multiBeta = smqrTrianInf(X, Y, coeff, nrow(X), ncol(X), tau, B, tol, iteMax)
    }
    ciList = getPivCI(coeff, multiBeta, alpha)
    z = qnorm(1 - alpha / 2)
    normCI = as.matrix(getNormCI(coeff, rowSds(multiBeta), z))
    return (list(coeff = coeff, ite = rst$ite, residual = as.numeric(rst$residual), bandwidth = rst$bandwidth, tau = tau, kernel = kernel, 
                 n = nrow(X), p = ncol(X), perCI = as.matrix(ciList$perCI), pivCI = as.matrix(ciList$pivCI), normCI = normCI))
  }
}