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

#' @title Convolution-Type Smoothed Quantile Regression
#' @description Estimation and inference for conditional linear quantile regression models using a convolution smoothed approach. Efficient gradient-based methods are employed for fitting both a single model and a regression process over a quantile range. 
#' Normal-based and (multiplier) bootstrap confidence intervals for all slope coefficients are constructed.
#' @param X An \eqn{n} by \eqn{p} design matrix. Each row is a vector of observations with \eqn{p} covariates. Number of observations \eqn{n} must be greater than number of covariates \eqn{p}.
#' @param Y An \eqn{n}-dimensional response vector.
#' @param tau (\strong{optional}) The desired quantile level. Default is 0.5. Value must be between 0 and 1.
#' @param kernel (\strong{optional})  A character string specifying the choice of kernel function. Default is "Gaussian". Choices are "Gaussian", "logistic", "uniform", "parabolic" and "triangular".
#' @param h (\strong{optional}) Bandwidth/smoothing parameter. Default is \eqn{\max\{((log(n) + p) / n)^{0.4}, 0.05\}}. The default will be used if the input value is less than or equal to 0.
#' @param checkSing (\strong{optional}) A logical flag. Default is FALSE. If \code{checkSing = TRUE}, then it will check if the design matrix is singular before running conquer. 
#' @param tol (\strong{optional}) Tolerance level of the gradient descent algorithm. The iteration will stop when the maximum magnitude of all the elements of the gradient is less than \code{tol}. Default is 1e-04.
#' @param iteMax (\strong{optional}) Maximum number of iterations. Default is 5000.
#' @param stepBounded (\strong{optional}) A logical flag. Default is TRUE. If \code{stepBounded = TRUE}, then the step size of gradient descent is upper bounded by \code{stepMax}. If \code{stepBounded = FALSE}, then the step size is unbounded.
#' @param stepMax (\strong{optional}) Maximum bound for the gradient descent step size. Default is 100.
#' @param ci (\strong{optional}) A character string specifying methods to construct confidence intervals. Choices are "none" (default), "bootstrap", "asymptotic" and "both". If \code{ci = "none"}, then confidence intervals will not be constructed. 
#' If \code{ci = "bootstrap"}, then three types of confidence intervals (percentile, pivotal and normal) will be constructed via multiplier bootstrap. 
#' If \code{ci = "asymptotic"}, then confidence intervals will be constructed based on estimated asymptotic covariance matrix. 
#' If \code{ci = "both"}, then confidence intervals from both bootstrap and asymptotic covariance will be returned.
#' @param alpha (\strong{optional}) Miscoverage level for each confidence interval. Default is 0.05.
#' @param B (\strong{optional}) The size of bootstrap samples. Default is 1000.
#' @return An object containing the following items will be returned:
#' \describe{
#' \item{\code{coeff}}{A \eqn{(p + 1)}-vector of estimated quantile regression coefficients, including the intercept.}
#' \item{\code{ite}}{Number of iterations until convergence.}
#' \item{\code{residual}}{An \eqn{n}-vector of fitted residuals.}
#' \item{\code{bandwidth}}{Bandwidth value.}
#' \item{\code{tau}}{Quantile level.}
#' \item{\code{kernel}}{Kernel function.}
#' \item{\code{n}}{Sample size.}
#' \item{\code{p}}{Number of covariates.}
#' \item{\code{perCI}}{The percentile confidence intervals for regression coefficients. Only available if \code{ci = "bootstrap"} or \code{ci = "both"}.}
#' \item{\code{pivCI}}{The pivotal confidence intervals for regression coefficients. Only available if \code{ci = "bootstrap"} or \code{ci = "both"}.}
#' \item{\code{normCI}}{The normal-based confidence intervals for regression coefficients. Only available if \code{ci = "bootstrap"} or \code{ci = "both"}.}
#' \item{\code{asyCI}}{The asymptotic confidence intervals for regression coefficients. Only available if \code{ci = "asymptotic"} or \code{ci = "both"}.}
#' }
#' @references Barzilai, J. and Borwein, J. M. (1988). Two-point step size gradient methods. IMA J. Numer. Anal., 8, 141–148.
#' @references Fernandes, M., Guerre, E. and Horta, E. (2021). Smoothing quantile regressions. J. Bus. Econ. Statist., 39, 338-357.
#' @references He, X., Pan, X., Tan, K. M., and Zhou, W.-X. (2022+). Smoothed quantile regression for large-scale inference. J. Econometrics, in press.
#' @references Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica, 46, 33-50.
#' @seealso See \code{\link{conquer.process}} for smoothed quantile regression process.
#' @examples 
#' n = 500; p = 10
#' beta = rep(1, p)
#' X = matrix(rnorm(n * p), n, p)
#' Y = X %*% beta + rt(n, 2)
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
#' fit = conquer(X, Y, tau = 0.5, kernel = "Gaussian", ci = "bootstrap")
#' ci.per = fit$perCI
#' ci.piv = fit$pivCI
#' ci.norm = fit$normCI
#' @export 
conquer = function(X, Y, tau = 0.5, kernel = c("Gaussian", "logistic", "uniform", "parabolic", "triangular"), h = 0.0, checkSing = FALSE, tol = 0.0001, 
                   iteMax = 5000, stepBounded = TRUE, stepMax = 100.0, ci = c("none", "bootstrap", "asymptotic", "both"), alpha = 0.05, B = 1000) {
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
  if (checkSing && rankMatrix(X)[1] < ncol(X)) {
    stop("Error: the design matrix X is singular.")
  }
  kernel = match.arg(kernel)
  ci = match.arg(ci)
  if (ci == "none") {
    rst = NULL
    if (kernel == "Gaussian") {
      if (stepBounded) {
        rst = smqrGauss(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
      } else {
        rst = smqrGaussUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
      }
    } else if (kernel == "logistic") {
      if (stepBounded) {
        rst = smqrLogistic(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
      } else {
        rst = smqrLogisticUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
      }
    } else if (kernel == "uniform") {
      if (stepBounded) {
        rst = smqrUnif(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
      } else {
        rst = smqrUnifUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
      }
    } else if (kernel == "parabolic") {
      if (stepBounded) {
        rst = smqrPara(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
      } else {
        rst = smqrParaUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
      }
    } else {
      if (stepBounded) {
        rst = smqrTrian(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
      } else {
        rst = smqrTrianUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
      }
    }
    return (list(coeff = as.numeric(rst$coeff), ite = rst$ite, residual = as.numeric(rst$residual), bandwidth = rst$bandwidth, tau = tau, 
                 kernel = kernel, n = nrow(X), p = ncol(X)))
  } else if (ci == "bootstrap") {
    rst = coeff = multiBeta = NULL
    if (kernel == "Gaussian") {
      if (stepBounded) {
        rst = smqrGauss(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrGaussInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax, stepMax)
      } else {
        rst = smqrGaussUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrGaussInfUbd(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
      }
    } else if (kernel == "logistic") {
      if (stepBounded) {
        rst = smqrLogistic(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrLogisticInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax, stepMax)
      } else {
        rst = smqrLogisticUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrLogisticInfUbd(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
      }
    } else if (kernel == "uniform") {
      if (stepBounded) {
        rst = smqrUnif(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrUnifInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax, stepMax)
      } else {
        rst = smqrUnifUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrUnifInfUbd(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
      }
    } else if (kernel == "parabolic") {
      if (stepBounded) {
        rst = smqrPara(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrParaInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax, stepMax)
      } else {
        rst = smqrParaUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrParaInfUbd(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
      }
    } else {
      if (stepBounded) {
        rst = smqrTrian(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrTrianInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax, stepMax)
      } else {
        rst = smqrTrianUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrTrianInfUbd(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
      }
    }
    ciList = getPivCI(coeff, multiBeta, alpha)
    z = qnorm(1 - alpha / 2)
    normCI = as.matrix(getNormCI(coeff, rowSds(multiBeta), z))
    return (list(coeff = coeff, ite = rst$ite, residual = as.numeric(rst$residual), bandwidth = rst$bandwidth, tau = tau, kernel = kernel, 
                 n = nrow(X), p = ncol(X), perCI = as.matrix(ciList$perCI), pivCI = as.matrix(ciList$pivCI), normCI = normCI))
  } else if (ci == "asymptotic") {
    rst = coeff = NULL
    if (kernel == "Gaussian") {
      if (stepBounded) {
        rst = smqrGauss(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
      } else {
        rst = smqrGaussUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
      }
    } else if (kernel == "logistic") {
      if (stepBounded) {
        rst = smqrLogistic(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
      } else {
        rst = smqrLogisticUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
      }
    } else if (kernel == "uniform") {
      if (stepBounded) {
        rst = smqrUnif(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
      } else {
        rst = smqrUnifUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
      }
    } else if (kernel == "parabolic") {
      if (stepBounded) {
        rst = smqrPara(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
      } else {
        rst = smqrParaUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
      }
    } else {
      if (stepBounded) {
        rst = smqrTrian(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
      } else {
        rst = smqrTrianUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
      }
    }
    res = as.numeric(rst$residual)
    h = rst$bandwidth
    n = nrow(X)
    z = qnorm(1 - alpha / 2)
    asyCI = asymptoticCI(X, res, coeff, tau, n, h, z)
    return (list(coeff = coeff, ite = rst$ite, residual = res, bandwidth = h, tau = tau, kernel = kernel, n = n, p = ncol(X), asyCI = asyCI))
  } else {
    rst = coeff = multiBeta = NULL
    if (kernel == "Gaussian") {
      if (stepBounded) {
        rst = smqrGauss(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrGaussInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax, stepMax)
      } else {
        rst = smqrGaussUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrGaussInfUbd(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
      }
    } else if (kernel == "logistic") {
      if (stepBounded) {
        rst = smqrLogistic(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrLogisticInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax, stepMax)
      } else {
        rst = smqrLogisticUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrLogisticInfUbd(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
      }
    } else if (kernel == "uniform") {
      if (stepBounded) {
        rst = smqrUnif(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrUnifInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax, stepMax)
      } else {
        rst = smqrUnifUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrUnifInfUbd(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
      }
    } else if (kernel == "parabolic") {
      if (stepBounded) {
        rst = smqrPara(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrParaInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax, stepMax)
      } else {
        rst = smqrParaUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrParaInfUbd(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
      }
    } else {
      if (stepBounded) {
        rst = smqrTrian(X, Y, tau, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrTrianInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax, stepMax)
      } else {
        rst = smqrTrianUbd(X, Y, tau, h, tol = tol, iteMax = iteMax)
        coeff = as.numeric(rst$coeff)
        multiBeta = smqrTrianInfUbd(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
      }
    }
    ciList = getPivCI(coeff, multiBeta, alpha)
    z = qnorm(1 - alpha / 2)
    normCI = as.matrix(getNormCI(coeff, rowSds(multiBeta), z))
    res = as.numeric(rst$residual)
    h = rst$bandwidth
    n = nrow(X)
    asyCI = asymptoticCI(X, res, coeff, tau, n, h, z)
    return (list(coeff = coeff, ite = rst$ite, residual = res, bandwidth = h, tau = tau, kernel = kernel, n = n, p = ncol(X), 
                 perCI = as.matrix(ciList$perCI), pivCI = as.matrix(ciList$pivCI), normCI = normCI, asyCI = asyCI))
  }
}

#' @title Convolution-Type Smoothed Quantile Regression Process
#' @description Fit a smoothed quantile regression process over a quantile range. The algorithm is essentially the same as \code{\link{conquer}}.
#' @param X An \eqn{n} by \eqn{p} design matrix. Each row is a vector of observations with \eqn{p} covariates. Number of observations \eqn{n} must be greater than number of covariates \eqn{p}.
#' @param Y An \eqn{n}-dimensional response vector.
#' @param tauSeq (\strong{optional}) A sequence of quantile values (between 0 and 1). Default is \eqn{\{0.1, 0.15, 0.2, ..., 0.85, 0.9\}}.
#' @param kernel (\strong{optional})  A character string specifying the choice of kernel function. Default is "Gaussian". Choices are "Gaussian", "logistic", "uniform", "parabolic" and "triangular".
#' @param h (\strong{optional}) The bandwidth/smoothing parameter. Default is \eqn{\max\{((log(n) + p) / n)^{0.4}, 0.05\}}. The default will be used if the input value is less than or equal to 0.
#' @param checkSing (\strong{optional}) A logical flag. Default is FALSE. If \code{checkSing = TRUE}, then it will check if the design matrix is singular before running conquer. 
#' @param tol (\strong{optional}) Tolerance level of the gradient descent algorithm. The iteration will stop when the maximum magnitude of all the elements of the gradient is less than \code{tol}. Default is 1e-04.
#' @param iteMax (\strong{optional}) Maximum number of iterations. Default is 5000.
#' @param stepBounded (\strong{optional}) A logical flag. Default is TRUE.  If \code{stepBounded = TRUE}, then the step size of gradient descent is upper bounded by \code{stepMax}. If \code{stepBounded = FALSE}, then the step size is unbounded.
#' @param stepMax (\strong{optional}) Maximum bound for the gradient descent step size. Default is 100.
#' @return An object containing the following items will be returned:
#' \describe{
#' \item{\code{coeff}}{A \eqn{(p + 1)} by \eqn{m} matrix of estimated quantile regression process coefficients, including the intercept. m is the length of \code{tauSeq}.}
#' \item{\code{bandwidth}}{Bandwidth value.}
#' \item{\code{tauSeq}}{The sequence of quantile levels.}
#' \item{\code{kernel}}{The choice of kernel function.}
#' \item{\code{n}}{Sample size.}
#' \item{\code{p}}{Number the covariates.}
#' }
#' @references Barzilai, J. and Borwein, J. M. (1988). Two-point step size gradient methods. IMA J. Numer. Anal., 8, 141–148.
#' @references Fernandes, M., Guerre, E. and Horta, E. (2021). Smoothing quantile regressions. J. Bus. Econ. Statist., 39, 338-357.
#' @references He, X., Pan, X., Tan, K. M., and Zhou, W.-X. (2022+). Smoothed quantile regression for large-scale inference. J. Econometrics, in press.
#' @references Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica, 46, 33-50.
#' @seealso See \code{\link{conquer}} for single-index smoothed quantile regression.
#' @examples 
#' n = 500; p = 10
#' beta = rep(1, p)
#' X = matrix(rnorm(n * p), n, p)
#' Y = X %*% beta + rt(n, 2)
#' 
#' ## Smoothed quantile regression process with Gaussian kernel
#' fit.Gauss = conquer.process(X, Y, tauSeq = seq(0.2, 0.8, by = 0.05), kernel = "Gaussian")
#' beta.hat.Gauss = fit.Gauss$coeff
#' 
#' ## Smoothe quantile regression with uniform kernel
#' fit.unif = conquer.process(X, Y, tauSeq = seq(0.2, 0.8, by = 0.05), kernel = "uniform")
#' beta.hat.unif = fit.unif$coeff
#' @export 
conquer.process = function(X, Y, tauSeq = seq(0.1, 0.9, by = 0.05), kernel = c("Gaussian", "logistic", "uniform", "parabolic", "triangular"), h = 0.0, 
                           checkSing = FALSE, tol = 0.0001, iteMax = 5000, stepBounded = TRUE, stepMax = 100.0) {
  if (nrow(X) != length(Y)) {
    stop("Error: the length of Y must be the same as the number of rows of X.")
  }
  if (ncol(X) >= nrow(X)) {
    stop("Error: the number of columns of X cannot exceed the number of rows of X.")
  }
  if(min(tauSeq) <= 0 || max(tauSeq) >= 1) {
    stop("Error: every quantile level must be in (0, 1).")
  }
  if (min(colSds(X)) == 0) {
    stop("Error: at least one column of X is constant.")
  }
  if (checkSing && rankMatrix(X)[1] < ncol(X)) {
    stop("Error: the design matrix X is singular.")
  }
  kernel = match.arg(kernel)
  rst = NULL
  if (kernel == "Gaussian") {
    if (stepBounded) {
      rst = smqrGaussProc(X, Y, tauSeq, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
    } else {
      rst = smqrGaussProcUbd(X, Y, tauSeq, h, tol = tol, iteMax = iteMax)
    }
  } else if (kernel == "logistic") {
    if (stepBounded) {
      rst = smqrLogisticProc(X, Y, tauSeq, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
    } else {
      rst = smqrLogisticProcUbd(X, Y, tauSeq, h, tol = tol, iteMax = iteMax)
    }
  } else if (kernel == "uniform") {
    if (stepBounded) {
      rst = smqrUnifProc(X, Y, tauSeq, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
    } else {
      rst = smqrUnifProcUbd(X, Y, tauSeq, h, tol = tol, iteMax = iteMax)
    }
  } else if (kernel == "parabolic") {
    if (stepBounded) {
      rst = smqrParaProc(X, Y, tauSeq, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
    } else {
      rst = smqrParaProcUbd(X, Y, tauSeq, h, tol = tol, iteMax = iteMax)
    }
  } else {
    if (stepBounded) {
      rst = smqrTrianProc(X, Y, tauSeq, h, tol = tol, iteMax = iteMax, stepMax = stepMax)
    } else {
      rst = smqrTrianProcUbd(X, Y, tauSeq, h, tol = tol, iteMax = iteMax)
    }
  }
  return (list(coeff = rst$coeff, bandwidth = rst$bandwidth, tauSeq = tauSeq, kernel = kernel, n = nrow(X), p = ncol(X)))
}

#' @title Penalized Convolution-Type Smoothed Quantile Regression
#' @description Fit sparse quantile regression models in high dimensions via regularized conquer methods with "lasso", "elastic-net", "group lasso", "sparse group lasso", "scad" and "mcp" penalties. 
#' For "scad" and "mcp", the iteratively reweighted \eqn{\ell_1}-penalized algorithm is complemented with a local adpative majorize-minimize algorithm.
#' @param X An \eqn{n} by \eqn{p} design matrix. Each row is a vector of observations with \eqn{p} covariates. 
#' @param Y An \eqn{n}-dimensional response vector.
#' @param lambda (\strong{optional}) Regularization parameter. Can be a scalar or a sequence. If the input is a sequence, the function will sort it in ascending order, and run the regression accordingly. Default is 0.2.
#' @param tau (\strong{optional}) Quantile level (between 0 and 1). Default is 0.5.
#' @param kernel (\strong{optional}) A character string specifying the choice of kernel function. Default is "Gaussian". Choices are "Gaussian", "logistic", "uniform", "parabolic" and "triangular".
#' @param h (\strong{optional}) Bandwidth/smoothing parameter. Default is \eqn{\max\{0.5 * (log(p) / n)^{0.25}, 0.05\}}. The default will be used if the input value is less than or equal to 0.
#' @param penalty (\strong{optional}) A character string specifying the penalty. Default is "lasso" (Tibshirani, 1996). The other options are "elastic" for elastic-net (Zou and Hastie, 2005), "group" for group lasso (Yuan and Lin, 2006), "sparse-group" for sparse group lasso (Simon et al., 2013), "scad" (Fan and Li, 2001) and "mcp" (Zhang, 2010).
#' @param para.elastic (\strong{optional}) The mixing parameter between 0 and 1 (usually noted as \eqn{\alpha}) for elastic-net. The penalty is defined as \eqn{\alpha ||\beta||_1 + (1 - \alpha) ||\beta||_2^2}. Default is 0.5.
#' Setting \code{para.elastic = 1} gives the lasso penalty, and setting \code{para.elastic = 0} yields the ridge penalty. Only specify it when \code{penalty = "elastic"}.
#' @param group (\strong{optional}) A \eqn{p}-dimensional vector specifying group indices. Only specify it if \code{penalty = "group"} or \code{penalty = "sparse-group"}. 
#' For example, if \eqn{p = 10}, and we assume the first 3 coefficients belong to the first group, and the last 7 coefficients belong to the second group, then the argument should be \code{group = c(rep(1, 3), rep(2, 7))}. If not specified, then the penalty will be the classical lasso.
#' @param weights (\strong{optional}) A vector specifying groups weights for group Lasso and sparse group Lasso. The length must be equal to the number of groups. If not specified, the default weights are square roots of group sizes. 
#' For example , if \code{group = c(rep(1, 3), rep(2, 7))}, then the default weights are \eqn{\sqrt{3}} for the first group, and \eqn{\sqrt{7}} for the second group.
#' @param para.scad (\strong{optional}) The constant parameter for "scad". Default value is 3.7. Only specify it if \code{penalty = "scad"}.
#' @param para.mcp (\strong{optional}) The constant parameter for "mcp". Default value is 3. Only specify it if \code{penalty = "mcp"}.
#' @param epsilon (\strong{optional}) A tolerance level for the stopping rule. The iteration will stop when the maximum magnitude of the change of coefficient updates is less than \code{epsilon}. Default is 0.001.
#' @param iteMax (\strong{optional}) Maximum number of iterations. Default is 500.
#' @param phi0 (\strong{optional}) The initial quadratic coefficient parameter in the local adaptive majorize-minimize algorithm. Default is 0.01.
#' @param gamma (\strong{optional}) The adaptive search parameter (greater than 1) in the local adaptive majorize-minimize algorithm. Default is 1.2.
#' @param iteTight (\strong{optional}) Maximum number of tightening iterations in the iteratively reweighted \eqn{\ell_1}-penalized algorithm. Only specify it if the penalty is scad or mcp. Default is 3.
#' @return An object containing the following items will be returned:
#' \describe{
#' \item{\code{coeff}}{If the input \code{lambda} is a scalar, then \code{coeff} returns a \eqn{(p + 1)} vector of estimated coefficients, including the intercept. If the input \code{lambda} is a sequence, then \code{coeff} returns a \eqn{(p + 1)} by \eqn{nlambda} matrix, where \eqn{nlambda} refers to the length of \code{lambda} sequence.}
#' \item{\code{bandwidth}}{Bandwidth value.}
#' \item{\code{tau}}{Quantile level.}
#' \item{\code{kernel}}{Kernel function.}
#' \item{\code{penalty}}{Penalty type.}
#' \item{\code{lambda}}{Regularization parameter(s).}
#' \item{\code{n}}{Sample size.}
#' \item{\code{p}}{Number of the covariates.}
#' }
#' @references Belloni, A. and Chernozhukov, V. (2011). \eqn{\ell_1} penalized quantile regression in high-dimensional sparse models. Ann. Statist., 39, 82-130.
#' @references Fan, J. and Li, R. (2001). Variable selection via nonconcave regularized likelihood and its oracle properties. J. Amer. Statist. Assoc., 96, 1348-1360.
#' @references Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist., 46, 814-841.
#' @references Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica, 46, 33-50.
#' @references Simon, N., Friedman, J., Hastie, T. and Tibshirani, R. (2013). A sparse-group lasso. J. Comp. Graph. Statist., 22, 231-245.
#' @references Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. J. R. Statist. Soc. Ser. B, 58, 267–288.
#' @references Tan, K. M., Wang, L. and Zhou, W.-X. (2022). High-dimensional quantile regression: convolution smoothing and concave regularization. J. Roy. Statist. Soc. Ser. B, 84, 205-233.
#' @references Yuan, M. and Lin, Y. (2006). Model selection and estimation in regression with grouped variables., J. Roy. Statist. Soc. Ser. B, 68, 49-67.
#' @references Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. Ann. Statist., 38, 894-942.
#' @references Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. J. R. Statist. Soc. Ser. B, 67, 301-320.
#' @seealso See \code{\link{conquer.cv.reg}} for regularized quantile regression with cross-validation.
#' @examples 
#' n = 200; p = 500; s = 10
#' beta = c(rep(1.5, s), rep(0, p - s))
#' X = matrix(rnorm(n * p), n, p)
#' Y = X %*% beta + rt(n, 2)
#' 
#' ## Regularized conquer with lasso penalty at tau = 0.7
#' fit.lasso = conquer.reg(X, Y, lambda = 0.05, tau = 0.7, penalty = "lasso")
#' beta.lasso = fit.lasso$coeff
#' 
#' ## Regularized conquer with elastic-net penalty at tau = 0.7
#' fit.elastic = conquer.reg(X, Y, lambda = 0.1, tau = 0.7, penalty = "elastic", para.elastic = 0.7)
#' beta.elastic = fit.elastic$coeff
#' 
#' ## Regularized conquer with scad penalty at tau = 0.7
#' fit.scad = conquer.reg(X, Y, lambda = 0.13, tau = 0.7, penalty = "scad")
#' beta.scad = fit.scad$coeff
#' 
#' ## Regularized conquer with group lasso at tau = 0.7
#' beta = c(rep(1.3, 5), rep(1.5, 5), rep(0, p - s))
#' err = rt(n, 2)
#' Y = X %*% beta + err
#' group = c(rep(1, 5), rep(2, 5), rep(3, p - s))
#' fit.group = conquer.reg(X, Y, lambda = 0.05, tau = 0.7, penalty = "group", group = group)
#' beta.group = fit.group$coeff
#' @export 
conquer.reg = function(X, Y, lambda = 0.2, tau = 0.5, kernel = c("Gaussian", "logistic", "uniform", "parabolic", "triangular"), h = 0.0, 
                       penalty = c("lasso", "elastic", "group", "sparse-group", "scad", "mcp"), para.elastic = 0.5, group = NULL, weights = NULL, 
                       para.scad = 3.7, para.mcp = 3.0, epsilon = 0.001, iteMax = 500, phi0 = 0.01, gamma = 1.2, iteTight = 3) {
  n = nrow(X)
  p = ncol(X)
  if (length(Y) != n) {
    stop("Error: the length of Y must be the same as the number of rows of X.")
  }
  if (tau <= 0 || tau >= 1) {
    stop("Error: the quantile level tau must be in (0, 1).")
  }
  if (min(lambda) <= 0) {
    stop("Error: lambda must be positive.")
  }
  if (min(colSds(X)) == 0) {
    stop("Error: at least one column of X is constant.")
  }
  kernel = match.arg(kernel)
  penalty = match.arg(penalty)
  if (h <= 0.0) {
    h = max(0.5 * (log(p) / n)^(0.25), 0.05);
  }
  rst = NULL
  lambda = sort(lambda)
  if (penalty == "lasso" || (penalty == "group" && is.null(group)) || (penalty == "sparse-group" && is.null(group))) {
    if (kernel == "Gaussian") {
      if (length(lambda) == 1) {
        rst = conquerGaussLasso(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerGaussLassoSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "logistic") {
      if (length(lambda) == 1) {
        rst = conquerLogisticLasso(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerLogisticLassoSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "uniform") {
      if (length(lambda) == 1) {
        rst = conquerUnifLasso(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerUnifLassoSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "parabolic") {
      if (length(lambda) == 1) {
        rst = conquerParaLasso(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerParaLassoSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax)
      }
    } else {
      if (length(lambda) == 1) {
        rst = conquerTrianLasso(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerTrianLassoSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax)
      }
    }
  } else if (penalty == "elastic") {
    if (para.elastic < 0 || para.elastic > 1) {
      stop("Error: the elastic net parameter must be in [0, 1].")
    }
    if (kernel == "Gaussian") {
      if (length(lambda) == 1) {
        rst = conquerGaussElastic(X, Y, lambda, tau, para.elastic, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerGaussElasticSeq(X, Y, lambda, tau, para.elastic, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "logistic") {
      if (length(lambda) == 1) {
        rst = conquerLogisticElastic(X, Y, lambda, tau, para.elastic, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerLogisticElasticSeq(X, Y, lambda, tau, para.elastic, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "uniform") {
      if (length(lambda) == 1) {
        rst = conquerUnifElastic(X, Y, lambda, tau, para.elastic, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerUnifElasticSeq(X, Y, lambda, tau, para.elastic, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "parabolic") {
      if (length(lambda) == 1) {
        rst = conquerParaElastic(X, Y, lambda, tau, para.elastic, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerParaElasticSeq(X, Y, lambda, tau, para.elastic, h, phi0, gamma, epsilon, iteMax)
      }
    } else {
      if (length(lambda) == 1) {
        rst = conquerTrianElastic(X, Y, lambda, tau, para.elastic, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerTrianElasticSeq(X, Y, lambda, tau, para.elastic, h, phi0, gamma, epsilon, iteMax)
      }
    }
  } else if (penalty == "group") {
    if (length(group) != p) {
      stop("Error: the argument group refers to the group indices, and its length must be the same as the number of columns of X.")
    }
    G = length(unique(group))
    group = c(0, group - 1)
    if (!is.null(weights) && length(weights) != G) {
      stop("Error: the length of weights must be equal to the number of groups.")
    } else if (is.null(weights)) {
      weights = sqrt(as.numeric(table(group)))
    }
    if (kernel == "Gaussian") {
      if (length(lambda) == 1) {
        rst = conquerGaussGroupLasso(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerGaussGroupLassoSeq(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "logistic") {
      if (length(lambda) == 1) {
        rst = conquerLogisticGroupLasso(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerLogisticGroupLassoSeq(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "uniform") {
      if (length(lambda) == 1) {
        rst = conquerUnifGroupLasso(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerUnifGroupLassoSeq(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "parabolic") {
      if (length(lambda) == 1) {
        rst = conquerParaGroupLasso(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerParaGroupLassoSeq(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
      }
    } else {
      if (length(lambda) == 1) {
        rst = conquerTrianGroupLasso(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerTrianGroupLassoSeq(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
      }
    }
  } else if (penalty == "sparse-group") {
    if (length(group) != p) {
      stop("Error: the argument group refers to the group indices, and its length must be the same as the number of columns of X.")
    }
    G = length(unique(group))
    group = c(0, group - 1)
    if (!is.null(weights) && length(weights) != G) {
      stop("Error: the length of weights must be equal to the number of groups.")
    } else if (is.null(weights)) {
      weights = sqrt(as.numeric(table(group)))
    }
    if (kernel == "Gaussian") {
      if (length(lambda) == 1) {
        rst = conquerGaussSparseGroupLasso(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerGaussSparseGroupLassoSeq(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "logistic") {
      if (length(lambda) == 1) {
        rst = conquerLogisticSparseGroupLasso(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerLogisticSparseGroupLassoSeq(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "uniform") {
      if (length(lambda) == 1) {
        rst = conquerUnifSparseGroupLasso(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerUnifSparseGroupLassoSeq(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
      }
    } else if (kernel == "parabolic") {
      if (length(lambda) == 1) {
        rst = conquerParaSparseGroupLasso(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerParaSparseGroupLassoSeq(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
      }
    } else {
      if (length(lambda) == 1) {
        rst = conquerTrianSparseGroupLasso(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
        rst = as.numeric(rst)
      } else {
        rst = conquerTrianSparseGroupLassoSeq(X, Y, lambda, tau, group, weights, G, h, phi0, gamma, epsilon, iteMax)
      }
    }
  } else if (penalty == "scad") {
    if (para.scad <= 0) {
      stop("Error: the scad parameter must be positive.")
    }
    if (kernel == "Gaussian") {
      if (length(lambda) == 1) {
        rst = conquerGaussScad(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
        rst = as.numeric(rst)
      } else {
        rst = conquerGaussScadSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
      }
    } else if (kernel == "logistic") {
      if (length(lambda) == 1) {
        rst = conquerLogisticScad(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
        rst = as.numeric(rst)
      } else {
        rst = conquerLogisticScadSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
      }
    } else if (kernel == "uniform") {
      if (length(lambda) == 1) {
        rst = conquerUnifScad(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
        rst = as.numeric(rst)
      } else {
        rst = conquerUnifScadSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
      }
    } else if (kernel == "parabolic") {
      if (length(lambda) == 1) {
        rst = conquerParaScad(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
        rst = as.numeric(rst)
      } else {
        rst = conquerParaScadSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
      }
    } else {
      if (length(lambda) == 1) {
        rst = conquerTrianScad(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
        rst = as.numeric(rst)
      } else {
        rst = conquerTrianScadSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
      }
    }
  } else {
    if (para.mcp <= 0) {
      stop("Error: the mcp parameter must be positive.")
    }
    if (kernel == "Gaussian") {
      if (length(lambda) == 1) {
        rst = conquerGaussMcp(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
        rst = as.numeric(rst)
      } else {
        rst = conquerGaussMcpSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
      }
    } else if (kernel == "logistic") {
      if (length(lambda) == 1) {
        rst = conquerLogisticMcp(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
        rst = as.numeric(rst)
      } else {
        rst = conquerLogisticMcpSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
      }
    } else if (kernel == "uniform") {
      if (length(lambda) == 1) {
        rst = conquerUnifMcp(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
        rst = as.numeric(rst)
      } else {
        rst = conquerUnifMcpSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
      }
    } else if (kernel == "parabolic") {
      if (length(lambda) == 1) {
        rst = conquerParaMcp(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
        rst = as.numeric(rst)
      } else {
        rst = conquerParaMcpSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
      }
    } else {
      if (length(lambda) == 1) {
        rst = conquerTrianMcp(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
        rst = as.numeric(rst)
      } else {
        rst = conquerTrianMcpSeq(X, Y, lambda, tau, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
      }
    }
  } 
  return (list(coeff = rst, bandwidth = h, tau = tau, kernel = kernel, penalty = penalty, lambda = lambda, n = n, p = p))
}

#' @title Cross-Validated Penalized Convolution-Type Smoothed Quantile Regression
#' @description Fit sparse quantile regression models via regularized conquer methods with "lasso", "elastic-net", "group lasso", "sparse group lasso", "scad" and "mcp" penalties. The regularization parameter \eqn{\lambda} is selected via cross-validation.
#' @param X An \eqn{n} by \eqn{p} design matrix. Each row is a vector of observations with \eqn{p} covariates. 
#' @param Y An \eqn{n}-dimensional response vector.
#' @param lambdaSeq (\strong{optional}) A sequence of candidate regularization parameters. If unspecified, the sequence will be generated by a simulated pivotal quantity approach proposed in Belloni and Chernozhukov (2011).
#' @param tau (\strong{optional}) Quantile level (between 0 and 1). Default is 0.5.
#' @param kernel (\strong{optional}) A character string specifying the choice of kernel function. Default is "Gaussian". Choices are "Gaussian", "logistic", "uniform", "parabolic" and "triangular".
#' @param h (\strong{optional}) The bandwidth parameter for kernel smoothing. Default is \eqn{\max\{0.5 * (log(p) / n)^{0.25}, 0.05\}}. The default will be used if the input value is less than or equal to 0.
#' @param penalty (\strong{optional}) A character string specifying the penalty. Default is "lasso" (Tibshirani, 1996). The other options are "elastic" for elastic-net (Zou and Hastie, 2005), "group" for group lasso (Yuan and Lin, 2006), "sparse-group" for sparse group lasso (Simon et al., 2013), "scad" (Fan and Li, 2001) and "mcp" (Zhang, 2010).
#' @param kfolds (\strong{optional}) Number of folds for cross-validation. Default is 5.
#' @param numLambda (\strong{optional}) Number of \eqn{\lambda} values for cross-validation if \code{lambdaSeq} is unspeficied. Default is 50.
#' @param para.elastic (\strong{optional}) The mixing parameter between 0 and 1 (usually noted as \eqn{\alpha}) for elastic net. The penalty is defined as \eqn{\alpha ||\beta||_1 + (1 - \alpha) ||\beta||_2^2}. Default is 0.5.
#' Setting \code{para.elastic = 1} gives the lasso penalty, and setting \code{para.elastic = 0} yields the ridge penalty. Only specify it when \code{penalty = "elastic"}.
#' @param group (\strong{optional}) A \eqn{p}-dimensional vector specifying group indices. Only specify it if \code{penalty = "group"} or \code{penalty = "sparse-group"}. 
#' For example, if \eqn{p = 10}, and we assume the first 3 coefficients belong to the first group, and the last 7 coefficients belong to the second group, then the argument should be \code{group = c(rep(1, 3), rep(2, 7))}. If not specified, then the penalty will be the classical lasso.
#' @param weights (\strong{optional}) A vector specifying groups weights for group Lasso and sparse group Lasso. The length must be equal to the number of groups. If not specified, the default weights are square roots of group sizes. 
#' For example , if \code{group = c(rep(1, 3), rep(2, 7))}, then the default weights are \eqn{\sqrt{3}} for the first group, and \eqn{\sqrt{7}} for the second group.
#' @param para.scad (\strong{optional}) The constant parameter for "scad". Default value is 3.7. Only specify it if \code{penalty = "scad"}.
#' @param para.mcp (\strong{optional}) The constant parameter for "mcp". Default value is 3. Only specify it if \code{penalty = "mcp"}.
#' @param epsilon (\strong{optional}) A tolerance level for the stopping rule. The iteration will stop when the maximum magnitude of the change of coefficient updates is less than \code{epsilon}. Default is 0.001.
#' @param iteMax (\strong{optional}) Maximum number of iterations. Default is 500.
#' @param phi0 (\strong{optional}) The initial quadratic coefficient parameter in the local adaptive majorize-minimize algorithm. Default is 0.01.
#' @param gamma (\strong{optional}) The adaptive search parameter (greater than 1) in the local adaptive majorize-minimize algorithm. Default is 1.2.
#' @param iteTight (\strong{optional}) Maximum number of tightening iterations in the iteratively reweighted \eqn{\ell_1}-penalized algorithm. Only specify it if the penalty is scad or mcp. Default is 3.
#' @return An object containing the following items will be returned:
#' \describe{
#' \item{\code{coeff.min}}{A \eqn{(p + 1)} vector of estimated coefficients including the intercept selected by minimizing the cross-validation errors.}
#' \item{\code{coeff.1se}}{A \eqn{(p + 1)} vector of estimated coefficients including the intercept. The corresponding \eqn{\lambda} is the largest \eqn{\lambda} such that the cross-validation error is within 1 standard error of the minimum.}
#' \item{\code{lambdaSeq}}{The sequence of regularization parameter candidates for cross-validation.}
#' \item{\code{lambda.min}}{Regularization parameter selected by minimizing the cross-validation errors. This is the corresponding \eqn{\lambda} of \code{coeff.min}.}
#' \item{\code{lambda.1se}}{The largest regularization parameter such that the cross-validation error is within 1 standard error of the minimum. This is the corresponding \eqn{\lambda} of \code{coeff.1se}.}
#' \item{\code{deviance}}{Cross-validation errors based on the quantile loss. The length is equal to the length of \code{lambdaSeq}.}
#' \item{\code{deviance.se}}{Estimated standard errors of \code{deviance}. The length is equal to the length of \code{lambdaSeq}.}
#' \item{\code{bandwidth}}{Bandwidth value.}
#' \item{\code{tau}}{Quantile level.}
#' \item{\code{kernel}}{Kernel function.}
#' \item{\code{penalty}}{Penalty type.}
#' \item{\code{n}}{Sample size.}
#' \item{\code{p}}{Number of covariates.}
#' }
#' @references Belloni, A. and Chernozhukov, V. (2011). \eqn{\ell_1} penalized quantile regression in high-dimensional sparse models. Ann. Statist., 39, 82-130.
#' @references Fan, J. and Li, R. (2001). Variable selection via nonconcave regularized likelihood and its oracle properties. J. Amer. Statist. Assoc., 96, 1348-1360.
#' @references Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist., 46, 814-841.
#' @references Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica, 46, 33-50.
#' @references Simon, N., Friedman, J., Hastie, T. and Tibshirani, R. (2013). A sparse-group lasso. J. Comp. Graph. Statist., 22, 231-245.
#' @references Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. J. R. Statist. Soc. Ser. B, 58, 267–288.
#' @references Tan, K. M., Wang, L. and Zhou, W.-X. (2022). High-dimensional quantile regression: convolution smoothing and concave regularization. J. Roy. Statist. Soc. Ser. B, 84, 205-233.
#' @references Yuan, M. and Lin, Y. (2006). Model selection and estimation in regression with grouped variables., J. Roy. Statist. Soc. Ser. B, 68, 49-67.
#' @references Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. Ann. Statist., 38, 894-942.
#' @references Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. J. R. Statist. Soc. Ser. B, 67, 301-320.
#' @seealso See \code{\link{conquer.reg}} for regularized quantile regression with a prescribed \eqn{lambda}.
#' @examples 
#' n = 100; p = 200; s = 5
#' beta = c(rep(1.5, s), rep(0, p - s))
#' X = matrix(rnorm(n * p), n, p)
#' Y = X %*% beta + rt(n, 2)
#' 
#' ## Cross-validated regularized conquer with lasso penalty at tau = 0.7
#' fit.lasso = conquer.cv.reg(X, Y, tau = 0.7, penalty = "lasso")
#' beta.lasso = fit.lasso$coeff.min
#' 
#' ## Cross-validated regularized conquer with elastic-net penalty at tau = 0.7
#' fit.elastic = conquer.cv.reg(X, Y, tau = 0.7, penalty = "elastic", para.elastic = 0.7)
#' beta.elastic = fit.elastic$coeff.min
#' 
#' ## Cross-validated regularized conquer with scad penalty at tau = 0.7
#' fit.scad = conquer.cv.reg(X, Y, tau = 0.7, penalty = "scad")
#' beta.scad = fit.scad$coeff.min
#' 
#' ## Regularized conquer with group lasso at tau = 0.7
#' beta = c(rep(1.3, 2), rep(1.5, 3), rep(0, p - s))
#' err = rt(n, 2)
#' Y = X %*% beta + err
#' group = c(rep(1, 2), rep(2, 3), rep(3, p - s))
#' fit.group = conquer.cv.reg(X, Y,tau = 0.7, penalty = "group", group = group)
#' beta.group = fit.group$coeff.min
#' @export 
conquer.cv.reg = function(X, Y, lambdaSeq = NULL, tau = 0.5, kernel = c("Gaussian", "logistic", "uniform", "parabolic", "triangular"), h = 0.0, 
                          penalty = c("lasso", "elastic", "group", "sparse-group", "scad", "mcp"), para.elastic = 0.5, group = NULL, weights = NULL,
                          para.scad = 3.7, para.mcp = 3.0, kfolds = 5, numLambda = 50, epsilon = 0.001, iteMax = 500, phi0 = 0.01, gamma = 1.2, iteTight = 3) {
  n = nrow(X)
  p = ncol(X)
  if (length(Y) != n) {
    stop("Error: the length of Y must be the same as the number of rows of X.")
  }
  if(tau <= 0 || tau >= 1) {
    stop("Error: the quantile level tau must be in (0, 1).")
  }
  if (!is.null(lambdaSeq) && min(lambdaSeq) <= 0) {
    stop("Error: all lambda's must be positive.")
  }
  if (min(colSds(X)) == 0) {
    stop("Error: at least one column of X is constant.")
  }
  kernel = match.arg(kernel)
  penalty = match.arg(penalty)
  if (h <= 0.0) {
    h = max(0.5 * (log(p) / n)^(0.25), 0.05);
  }
  if (is.null(lambdaSeq)) {
    if (numLambda == 1) {
      stop("Error: numLambda must be greater than 1 for cross-validation.")
    }
    nsim = 200
    U = matrix(runif(nsim * n), nsim, n)
    pivot = tau - (U <= tau)
    lambda0 = quantile(rowMaxs(abs(pivot %*% scale(X))), 0.9) / n
    lambdaSeq = seq(0.05, 2, length.out = numLambda) * lambda0
  } else if (length(lambdaSeq) == 1) {
    stop("Error: lambdaSeq must be a sequence. Please use conquer.reg instead for a specific lambda.")
  } else {
    lambdaSeq = sort(lambdaSeq)
  }
  folds = sample(rep(1:kfolds, ceiling(n / kfolds)), n)
  rst = NULL
  if (penalty == "lasso" || (penalty == "group" && is.null(group)) || (penalty == "sparse-group" && is.null(group))) {
    if (kernel == "Gaussian") {
      rst = cvGaussLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "logistic") {
      rst = cvLogisticLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "uniform") {
      rst = cvUnifLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "parabolic") {
      rst = cvParaLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else {
      rst = cvTrianLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax)
    }
  } else if (penalty == "elastic") {
    if (para.elastic < 0 || para.elastic > 1) {
      stop("Error: the elastic net parameter must be in [0, 1].")
    }
    if (kernel == "Gaussian") {
      rst = cvGaussElasticWarm(X, Y, lambdaSeq, folds, tau, para.elastic, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "logistic") {
      rst = cvLogisticElasticWarm(X, Y, lambdaSeq, folds, tau, para.elastic, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "uniform") {
      rst = cvUnifElasticWarm(X, Y, lambdaSeq, folds, tau, para.elastic, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "parabolic") {
      rst = cvParaElasticWarm(X, Y, lambdaSeq, folds, tau, para.elastic, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else {
      rst = cvTrianElasticWarm(X, Y, lambdaSeq, folds, tau, para.elastic, kfolds, h, phi0, gamma, epsilon, iteMax)
    }
  } else if (penalty == "group") {
    if (length(group) != p) {
      stop("Error: the argument group refers to the group indices, and its length must be the same as the number of columns of X.")
    }
    G = length(unique(group))
    group = c(0, group - 1)
    if (!is.null(weights) && length(weights) != G) {
      stop("Error: the length of weights must be equal to the number of groups.")
    } else if (is.null(weights)) {
      weights = sqrt(as.numeric(table(group)))
    }
    if (kernel == "Gaussian") {
      rst = cvGaussGroupLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, group, weights, G, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "logistic") {
      rst = cvLogisticGroupLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, group, weights, G, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "uniform") {
      rst = cvUnifGroupLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, group, weights, G, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "parabolic") {
      rst = cvParaGroupLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, group, weights, G, h, phi0, gamma, epsilon, iteMax)
    } else {
      rst = cvTrianGroupLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, group, weights, G, h, phi0, gamma, epsilon, iteMax)
    }
  } else if (penalty == "sparse-group") {
    if (length(group) != p) {
      stop("Error: the argument group refers to the group indices, and its length must be the same as the number of columns of X.")
    }
    G = length(unique(group))
    group = c(0, group - 1)
    if (!is.null(weights) && length(weights) != G) {
      stop("Error: the length of weights must be equal to the number of groups.")
    } else if (is.null(weights)) {
      weights = sqrt(as.numeric(table(group)))
    }
    if (kernel == "Gaussian") {
      rst = cvGaussSparseGroupLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, group, weights, G, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "logistic") {
      rst = cvLogisticSparseGroupLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, group, weights, G, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "uniform") {
      rst = cvUnifSparseGroupLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, group, weights, G, h, phi0, gamma, epsilon, iteMax)
    } else if (kernel == "parabolic") {
      rst = cvParaSparseGroupLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, group, weights, G, h, phi0, gamma, epsilon, iteMax)
    } else {
      rst = cvTrianSparseGroupLassoWarm(X, Y, lambdaSeq, folds, tau, kfolds, group, weights, G, h, phi0, gamma, epsilon, iteMax)
    }
  } else if (penalty == "scad") {
    if (para.scad <= 0) {
      stop("Error: the scad parameter must be positive.")
    }
    if (kernel == "Gaussian") {
      rst = cvGaussScadWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
    } else if (kernel == "logistic") {
      rst = cvLogisticScadWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
    } else if (kernel == "uniform") {
      rst = cvUnifScadWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
    } else if (kernel == "parabolic") {
      rst = cvParaScadWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
    } else {
      rst = cvTrianScadWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para.scad)
    }
  } else {
    if (para.mcp <= 0) {
      stop("Error: the mcp parameter must be positive.")
    }
    if (kernel == "Gaussian") {
      rst = cvGaussMcpWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
    } else if (kernel == "logistic") {
      rst = cvLogisticMcpWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
    } else if (kernel == "uniform") {
      rst = cvUnifMcpWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
    } else if (kernel == "parabolic") {
      rst = cvParaMcpWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
    } else {
      rst = cvTrianMcpWarm(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para.mcp)
    }
  } 
  return (list(coeff.min = as.numeric(rst$coeff), coeff.1se = as.numeric(rst$coeffSe), lambdaSeq = lambdaSeq, lambda.min = rst$lambdaMin, 
               lambda.1se = rst$lambdaSe, deviance = as.numeric(rst$deviance), deviance.se = as.numeric(rst$devianceSd), bandwidth = h, tau = tau, 
               kernel = kernel, penalty = penalty, n = n, p = p))
}

