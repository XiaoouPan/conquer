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
#' @description Fit a smoothed quantile regression via convolution-type smoothing method. The solution is computed using gradient descent with Barzilai-Borwein step size. Constructs (1-\eqn{\alpha}) confidence intervals with multiplier bootstrap.
#' @param X A \eqn{n} by \eqn{p} design matrix. Each row is a vector of observation with \eqn{p} covariates. Number of observations \eqn{n} must be greater than number of covariates \eqn{p}.
#' @param Y An \eqn{n}-dimensional response vector.
#' @param tau (\strong{optional}) The desired quantile level. Default is 0.5. Value must be between 0 and 1.
#' @param kernel (\strong{optional})  A character string specifying the choice of kernel function. Default is "Gaussian". Choices are "Gaussian", "logistic", "uniform", "parabolic" or "triangular".
#' @param h (\strong{optional}) The bandwidth parameter for kernel smoothing. Default is \eqn{max{((log(n) + p) / n)^{0.4}, 0.05}}. The default will be used if the input value is less than 0.05.
#' @param checkSing (\strong{optional}) A logical flag. Default is FALSE. If \code{checkSing = TRUE}, then it will check if the design matrix is singular before running conquer. 
#' @param tol (\strong{optional}) Tolerance level of the gradient descent algorithm. The gradient descent algorithm terminates when the maximal entry of the gradient is less than \code{tol}. Default is 1e-04. 
#' @param iteMax (\strong{optional}) Maximum number of iterations. Default is 5000.
#' @param ci (\strong{optional}) A logical flag. Default is FALSE. If \code{ci = TRUE}, then three types of confidence intervals (percentile, pivotal and normal) will be constructed via multiplier bootstrap.
#' @param alpha (\strong{optional}) The nominal level for (1-\eqn{\alpha})-confidence intervals. Default is 0.05. The input value must be in \eqn{(0, 1)}.
#' @param B (\strong{optional}) The size of bootstrap samples. Default is 1000.
#' @return An object containing the following items will be returned:
#' \describe{
#' \item{\code{coeff}}{A \eqn{(p + 1)}-vector of estimated quantile regression coefficients, including the intercept.}
#' \item{\code{ite}}{The number of iterations of the gradient descent algorithm for convergence.}
#' \item{\code{residual}}{The residuals of the quantile regression fit.}
#' \item{\code{bandwidth}}{The value of smoothing bandwidth.}
#' \item{\code{tau}}{The desired quantile level.}
#' \item{\code{kernel}}{The choice of kernel function.}
#' \item{\code{n}}{The sample size.}
#' \item{\code{p}}{The dimension of the covariates.}
#' \item{\code{perCI}}{The percentile confidence intervals for regression coefficients. Not available if \code{ci = FALSE}.}
#' \item{\code{pivCI}}{The pivotal confidence intervals for regression coefficients. Not available if \code{ci = FALSE}}
#' \item{\code{normCI}}{The normal-based confidence intervals for regression coefficients. Not available if \code{ci = FALSE}}
#' }
#' @references Barzilai, J. and Borwein, J. M. (1988). Two-point step size gradient methods. IMA J. Numer. Anal. 8 141–148.
#' @references Fernandes, M., Guerre, E. and Horta, E. (2019). Smoothing quantile regressions. J. Bus. Econ. Statist., in press.
#' @references He, X., Pan, X., Tan, K. M., and Zhou, W.-X. (2021+). Smoothed quantile regression for large-scale inference. J. Econometrics, in press.
#' @references Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica 46 33-50.
#' @author Xuming He <xmhe@umich.edu>, Xiaoou Pan <xip024@ucsd.edu>, Kean Ming Tan <keanming@umich.edu>, and Wen-Xin Zhou <wez243@ucsd.edu>
#' @seealso See \code{\link{conquer.process}} for smoothed quantile regression process.
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
conquer = function(X, Y, tau = 0.5, kernel = c("Gaussian", "logistic", "uniform", "parabolic", "triangular"), h = 0.0, checkSing = FALSE, tol = 0.0001, 
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
  if (checkSing && rankMatrix(X)[1] < ncol(X)) {
    stop("Error: the design matrix X is singular.")
  }
  kernel = match.arg(kernel)
  if (!ci) {
    rst = NULL
    if (kernel == "Gaussian") {
      rst = smqrGauss(X, Y, tau, h, tol = tol, iteMax = iteMax)
    } else if (kernel == "logistic") {
      rst = smqrLogistic(X, Y, tau, h, tol = tol, iteMax = iteMax)
    } else if (kernel == "uniform") {
      rst = smqrUnif(X, Y, tau, h, tol = tol, iteMax = iteMax)
    } else if (kernel == "parabolic") {
      rst = smqrPara(X, Y, tau, h, tol = tol, iteMax = iteMax)
    } else {
      rst = smqrTrian(X, Y, tau, h, tol = tol, iteMax = iteMax)
    }
    return (list(coeff = as.numeric(rst$coeff), ite = rst$ite, residual = as.numeric(rst$residual), bandwidth = rst$bandwidth, tau = tau, 
                 kernel = kernel, n = nrow(X), p = ncol(X)))
  } else {
    rst = coeff = multiBeta = NULL
    if (kernel == "Gaussian") {
      rst = smqrGauss(X, Y, tau, h, tol = tol, iteMax = iteMax)
      coeff = as.numeric(rst$coeff)
      multiBeta = smqrGaussInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
    } else if (kernel == "logistic") {
      rst = smqrLogistic(X, Y, tau, h, tol = tol, iteMax = iteMax)
      coeff = as.numeric(rst$coeff)
      multiBeta = smqrLogisticInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
    } else if (kernel == "uniform") {
      rst = smqrUnif(X, Y, tau, h, tol = tol, iteMax = iteMax)
      coeff = as.numeric(rst$coeff)
      multiBeta = smqrUnifInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
    } else if (kernel == "parabolic") {
      rst = smqrPara(X, Y, tau, h, tol = tol, iteMax = iteMax)
      coeff = as.numeric(rst$coeff)
      multiBeta = smqrParaInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
    } else {
      rst = smqrTrian(X, Y, tau, h, tol = tol, iteMax = iteMax)
      coeff = as.numeric(rst$coeff)
      multiBeta = smqrTrianInf(X, Y, coeff, nrow(X), ncol(X), h, tau, B, tol, iteMax)
    }
    ciList = getPivCI(coeff, multiBeta, alpha)
    z = qnorm(1 - alpha / 2)
    normCI = as.matrix(getNormCI(coeff, rowSds(multiBeta), z))
    return (list(coeff = coeff, ite = rst$ite, residual = as.numeric(rst$residual), bandwidth = rst$bandwidth, tau = tau, kernel = kernel, 
                 n = nrow(X), p = ncol(X), perCI = as.matrix(ciList$perCI), pivCI = as.matrix(ciList$pivCI), normCI = normCI))
  }
}

#' @title Convolution-Type Smoothed Quantile Regression Process
#' @description Fit a smoothed quantile regression process over a quantile range. The algorithm is essentially the same as \code{\link{conquer}}.
#' @param X A \eqn{n} by \eqn{p} design matrix. Each row is a vector of observation with \eqn{p} covariates. Number of observations \eqn{n} must be greater than number of covariates \eqn{p}.
#' @param Y An \eqn{n}-dimensional response vector.
#' @param tauSeq (\strong{optional}) The desired quantile grid. Default is \eqn{{0.1, 0.15, 0.2, ..., 0.85, 0.9}}. All values must be between 0 and 1.
#' @param kernel (\strong{optional})  A character string specifying the choice of kernel function. Default is "Gaussian". Choices are "Gaussian", "logistic", "uniform", "parabolic" or "triangular".
#' @param h (\strong{optional}) The bandwidth parameter for kernel smoothing. Default is \eqn{max{((log(n) + p) / n)^{0.4}, 0.05}}. The default will be used if the input value is less than 0.05.
#' @param checkSing (\strong{optional}) A logical flag. Default is FALSE. If \code{checkSing = TRUE}, then it will check if the design matrix is singular before running conquer. 
#' @param tol (\strong{optional}) Tolerance level of the gradient descent algorithm. The gradient descent algorithm terminates when the maximal entry of the gradient is less than \code{tol}. Default is 1e-04. 
#' @param iteMax (\strong{optional}) Maximum number of iterations. Default is 5000.
#' @return An object containing the following items will be returned:
#' \describe{
#' \item{\code{coeff}}{A \eqn{(p + 1)} by \eqn{m} matrix of estimated quantile regression process coefficients, including the intercept. m is the length of \code{tauSeq}.}
#' \item{\code{bandwidth}}{The value of smoothing bandwidth.}
#' \item{\code{tauSeq}}{The desired quantile levels for the process.}
#' \item{\code{kernel}}{The choice of kernel function.}
#' \item{\code{n}}{The sample size.}
#' \item{\code{p}}{The dimension of the covariates.}
#' }
#' @references Barzilai, J. and Borwein, J. M. (1988). Two-point step size gradient methods. IMA J. Numer. Anal. 8 141–148.
#' @references Fernandes, M., Guerre, E. and Horta, E. (2019). Smoothing quantile regressions. J. Bus. Econ. Statist., in press.
#' @references He, X., Pan, X., Tan, K. M., and Zhou, W.-X. (2021+). Smoothed quantile regression for large-scale inference. J. Econometrics, in press.
#' @references Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica 46 33-50.
#' @author Xuming He <xmhe@umich.edu>, Xiaoou Pan <xip024@ucsd.edu>, Kean Ming Tan <keanming@umich.edu>, and Wen-Xin Zhou <wez243@ucsd.edu>
#' @seealso See \code{\link{conquer}} for single-index smoothed quantile regression.
#' @examples 
#' n = 500; p = 10
#' beta = rep(1, p)
#' X = matrix(rnorm(n * p), n, p)
#' Y = 1 + X %*% beta + rt(n, 2)
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
                           checkSing = FALSE, tol = 0.0001, iteMax = 5000) {
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
    rst = smqrGaussProc(X, Y, tauSeq, h, tol = tol, iteMax = iteMax)
  } else if (kernel == "logistic") {
    rst = smqrLogisticProc(X, Y, tauSeq, h, tol = tol, iteMax = iteMax)
  } else if (kernel == "uniform") {
    rst = smqrUnifProc(X, Y, tauSeq, h, tol = tol, iteMax = iteMax)
  } else if (kernel == "parabolic") {
    rst = smqrParaProc(X, Y, tauSeq, h, tol = tol, iteMax = iteMax)
  } else {
    rst = smqrTrianProc(X, Y, tauSeq, h, tol = tol, iteMax = iteMax)
  }
  return (list(coeff = rst$coeff, bandwidth = rst$bandwidth, tauSeq = tauSeq, kernel = kernel, n = nrow(X), p = ncol(X)))
}

#' @title Convolution-Type Smoothed Quantile Regression with Regularization
#' @description Fit a smoothed quantile regression with Lasso, scad and mcp penalties and a prescribed regularization parameter \eqn{\lambda} using a local majorize-minimize algorithm.
#' @param X A \eqn{n} by \eqn{p} design matrix. Each row is a vector of observation with \eqn{p} covariates. 
#' @param Y An \eqn{n}-dimensional response vector.
#' @param lambda (\strong{optional}) A prescribed value for the regularization parameter. Default is 0.2.
#' @param tau (\strong{optional}) The desired quantile level. Default is 0.5. Value must be between 0 and 1.
#' @param kernel (\strong{optional}) A character string specifying the choice of kernel function. Default is "Gaussian". Choices are "Gaussian", "logistic", "uniform", "parabolic" or "triangular".
#' @param h (\strong{optional}) The bandwidth parameter for kernel smoothing. Default is \eqn{max{0.5 * (log(p) / n)^(1/4), 0.05}}.
#' @param penalty (\strong{optional}) A character string specifying the penalty. Default is "lasso". Choices are "lasso", "scad" or "mcp".
#' @param para (\strong{optional}) A parameter for concave penalties scad and mcp. Do not need to specify if the penalty is lasso. The default values are 3.7 for scand and 3 for mcp.
#' @param epsilon (\strong{optional}) A tolerance level for the optimization stopping rule. The algorithm terminates when the maximal entry of the change of coefficients is less than \code{epsilon}. Default is 0.001.
#' @param iteMax (\strong{optional}) Maximum number of iterations. Default is 500.
#' @param phi0 (\strong{optional}) A parameter for the local majorize-minimize algorithm. It is the initial value to search the largest eigen value of the covariance matrix. Default is 0.01.
#' @param gamma (\strong{optional}) A parameter for the local majorize-minimize algorithm. It is the inflating parameter to search the largest eigen value of the covariance matrix. Default is 1.2.
#' @param iteTight (\strong{optional}) Maximum number of tightening iterations. Do not need to specify if the penalty is lasso. Default is 3.
#' @return An object containing the following items will be returned:
#' \describe{
#' \item{\code{coeff}}{A \eqn{(p + 1)} vector of estimated coefficients, including the intercept.}
#' \item{\code{bandwidth}}{The value of smoothing bandwidth.}
#' \item{\code{tau}}{The desired quantile level for the regularized regression.}
#' \item{\code{kernel}}{The choice of kernel function.}
#' \item{\code{penalty}}{The choice of penalty type.}
#' \item{\code{lambda}}{The value of regularization parameter.}
#' \item{\code{n}}{The sample size.}
#' \item{\code{p}}{The dimension of the covariates.}
#' }
#' @references Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist. 46 814-841.
#' @references Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica 46 33-50.
#' @references Tan, K. M., Wang, L. and Zhou, W.-X. (2021). High-dimensional quantile regression: convolution smoothing and concave regularization. Preprint.
#' @author Xuming He <xmhe@umich.edu>, Xiaoou Pan <xip024@ucsd.edu>, Kean Ming Tan <keanming@umich.edu>, and Wen-Xin Zhou <wez243@ucsd.edu>
#' @seealso See \code{\link{conquer.cv.regularized}} for regularized quantile regression with cross-validation.
#' @examples 
#' n = 200; p = 500; s = 10
#' beta = c(rep(1.5, s), rep(0, p - s))
#' X = matrix(rnorm(n * p), n, p)
#' Y = X %*% beta + rt(n, 2)
#' 
#' ## Regularized conquer with lasso penalty at tau = 0.8
#' fit.lasso = conquer.regularized(X, Y, lambda = 0.05, tau = 0.8, kernel = "Gaussian", penalty = "lasso")
#' beta.lasso = fit.lasso$coeff
#' 
#' #' ## Regularized conquer with scad penalty at tau = 0.8
#' fit.scad = conquer.regularized(X, Y, lambda = 0.13, tau = 0.8, kernel = "Gaussian", penalty = "scad")
#' beta.scad = fit.scad$coeff
#' @export 
conquer.regularized = function(X, Y, lambda = 0.2, tau = 0.5, kernel = c("Gaussian", "logistic", "uniform", "parabolic", "triangular"), h = 0.0, 
                               penalty = c("lasso", "scad", "mcp"), para = NULL, epsilon = 0.001, iteMax = 500, phi0 = 0.01, gamma = 1.2, iteTight = 3) {
  if (nrow(X) != length(Y)) {
    stop("Error: the length of Y must be the same as the number of rows of X.")
  }
  if(tau <= 0 || tau >= 1) {
    stop("Error: the quantile level tau must be in (0, 1).")
  }
  if (lambda <= 0) {
    stop("Error: lambda must be positive.")
  }
  if (min(colSds(X)) == 0) {
    stop("Error: at least one column of X is constant.")
  }
  kernel = match.arg(kernel)
  penalty = match.arg(penalty)
  n = nrow(X)
  p = ncol(X)
  if (h <= 0.05) {
    h = max(0.5 * (log(p) / n)^(0.25), 0.05);
  }
  type = 1
  if (penalty == "lasso") {
    para = 1.0
  } else if (penalty == "scad") {
    type = 2
    if (is.null(para) || para <= 0) {
      para = 3.7
    }
  } else if (penalty == "mcp") {
    type = 3
    if (is.null(para) || para <= 0) {
      para = 3.0
    }
  }
  rst = NULL
  if (kernel == "Gaussian") {
    rst = conquerHdGauss(X, Y, lambda, tau, h, type, phi0, gamma, epsilon, iteMax, iteTight, para)
  } else if (kernel == "logistic") {
    rst = conquerHdLogistic(X, Y, lambda, tau, h, type, phi0, gamma, epsilon, iteMax, iteTight, para)
  } else if (kernel == "uniform") {
    rst = conquerHdUnif(X, Y, lambda, tau, h, type, phi0, gamma, epsilon, iteMax, iteTight, para)
  } else if (kernel == "parabolic") {
    rst = conquerHdPara(X, Y, lambda, tau, h, type, phi0, gamma, epsilon, iteMax, iteTight, para)
  } else {
    rst = conquerHdTrian(X, Y, lambda, tau, h, type, phi0, gamma, epsilon, iteMax, iteTight, para)
  }
  return (list(coeff = as.numeric(rst), bandwidth = h, tau = tau, kernel = kernel, penalty = penalty, lambda = lambda, n = n, p = p))
}

#' @title Cross-Validated Convolution-Type Smoothed Quantile Regression with Regularization
#' @description Fit a smoothed quantile regression with Lasso, scad and mcp penalties using a local majorize-minimize algorithm. The regularization parameter \eqn{\lambda} is calibrated via cross-validation.
#' @param X A \eqn{n} by \eqn{p} design matrix. Each row is a vector of observation with \eqn{p} covariates. 
#' @param Y An \eqn{n}-dimensional response vector.
#' @param lambdaSeq (\strong{optional}) A sequence of regularization parameter candidates. if not specified, the sequence will be generated based on a simulated pivotal quantity proposed in Belloni and Chernozhukov (2011).
#' @param tau (\strong{optional}) The desired quantile level. Default is 0.5. Value must be between 0 and 1.
#' @param kernel (\strong{optional}) A character string specifying the choice of kernel function. Default is "Gaussian". Choices are "Gaussian", "logistic", "uniform", "parabolic" or "triangular".
#' @param h (\strong{optional}) The bandwidth parameter for kernel smoothing. Default is \eqn{max{0.5 * (log(p) / n)^(1/4), 0.05}}.
#' @param penalty (\strong{optional}) A character string specifying the penalty. Default is "lasso". Choices are "lasso", "scad" or "mcp".
#' @param kfolds (\strong{optional}) The number of folds for cross-validation. Default is 5.
#' @param numLambda (\strong{optional}) The number of \eqn{lambda}'s for cross-validation if \code{lambdaSeq} is not speficied. Default is 50.
#' @param para (\strong{optional}) A parameter for concave penalties scad and mcp. Do not need to specify if the penalty is lasso. The default values are 3.7 for scand and 3 for mcp.
#' @param epsilon (\strong{optional}) A tolerance level for the optimization stopping rule. The algorithm terminates when the maximal entry of the change of coefficients is less than \code{epsilon}. Default is 0.001.
#' @param iteMax (\strong{optional}) Maximum number of iterations. Default is 500.
#' @param phi0 (\strong{optional}) A parameter for the local majorize-minimize algorithm. It is the initial value to search the largest eigen value of the covariance matrix. Default is 0.01.
#' @param gamma (\strong{optional}) A parameter for the local majorize-minimize algorithm. It is the inflating parameter to search the largest eigen value of the covariance matrix. Default is 1.2.
#' @param iteTight (\strong{optional}) Maximum number of tightening iterations. Do not need to specify if the penalty is lasso. Default is 3.
#' @return An object containing the following items will be returned:
#' \describe{
#' \item{\code{coeff}}{A \eqn{(p + 1)} vector of estimated coefficients, including the intercept.}
#' \item{\code{lambda}}{The optimal value of regularization parameter selected by cross-validation.}
#' \item{\code{bandwidth}}{The value of smoothing bandwidth.}
#' \item{\code{tau}}{The desired quantile level for the regularized regression.}
#' \item{\code{kernel}}{The choice of kernel function.}
#' \item{\code{penalty}}{The choice of penalty type.}
#' \item{\code{n}}{The sample size.}
#' \item{\code{p}}{The dimension of the covariates.}
#' }
#' @references Belloni, A. and Chernozhukov, V. (2011). l_1 penalized quantile regression in high-dimensional sparse models. Ann. Statist. 39 82-130.
#' @references Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist. 46 814-841.
#' @references Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica 46 33-50.
#' @references Tan, K. M., Wang, L. and Zhou, W.-X. (2021). High-dimensional quantile regression: convolution smoothing and concave regularization. Preprint.
#' @author Xuming He <xmhe@umich.edu>, Xiaoou Pan <xip024@ucsd.edu>, Kean Ming Tan <keanming@umich.edu>, and Wen-Xin Zhou <wez243@ucsd.edu>
#' @seealso See \code{\link{conquer.regularized}} for regularized quantile regression with a prescribed \eqn{lambda}.
#' @examples 
#' n = 200; p = 500; s = 10
#' beta = c(rep(1.5, s), rep(0, p - s))
#' X = matrix(rnorm(n * p), n, p)
#' Y = X %*% beta + rt(n, 2)
#' 
#' ## Cross-validated regularized conquer with lasso penalty at tau = 0.8
#' fit.lasso = conquer.cv.regularized(X, Y, tau = 0.8, kernel = "Gaussian", penalty = "lasso")
#' beta.lasso = fit.lasso$coeff
#' 
#' #' ## Cross-validated regularized conquer with scad penalty at tau = 0.8
#' fit.scad = conquer.cv.regularized(X, Y,tau = 0.8, kernel = "Gaussian", penalty = "scad")
#' beta.scad = fit.scad$coeff
#' @export 
conquer.cv.regularized = function(X, Y, lambdaSeq = NULL, tau = 0.5, kernel = c("Gaussian", "logistic", "uniform", "parabolic", "triangular"), h = 0.0, 
                                  penalty = c("lasso", "scad", "mcp"), kfolds = 5, numLambda = 50, para = NULL, epsilon = 0.001, iteMax = 500, phi0 = 0.01, 
                                  gamma = 1.2, iteTight = 3) {
  if (nrow(X) != length(Y)) {
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
  n = nrow(X)
  p = ncol(X)
  if (h <= 0.05) {
    h = max(0.5 * (log(p) / n)^(0.25), 0.05);
  }
  if (is.null(lambdaSeq)) {
    nsim = 200
    U = matrix(runif(nsim * n), nsim, n)
    pivot = tau - (U <= tau)
    lambda0 = quantile(rowMaxs(abs(pivot %*% scale(X))), 0.9) / n
    lambdaSeq = seq(0.5, 1.5, length.out = numLambda) * lambda0
  } 
  if (penalty == "scad" && is.null(para)) {
    para = 3.7
  } else if (penalty == "mcp" && is.null(para)) {
    para = 3.0
  }
  folds = createFolds(Y, kfolds, FALSE)
  rst = NULL
  if (kernel == "Gaussian") {
    if (penalty == "lasso") {
      rst = cvSmqrLassoGauss(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (penalty == "scad") {
      rst = cvSmqrScadGauss(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para)
    } else {
      rst = cvSmqrMcpGauss(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para)
    }
  } else if (kernel == "logistic") {
    if (penalty == "lasso") {
      rst = cvSmqrLassoLassoLogistic(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (penalty == "scad") {
      rst = cvSmqrScadLassoLogistic(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para)
    } else {
      rst = cvSmqrMcpLassoLogistic(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para)
    }
  } else if (kernel == "uniform") {
    if (penalty == "lasso") {
      rst = cvSmqrLassoUnif(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (penalty == "scad") {
      rst = cvSmqrScadUnif(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para)
    } else {
      rst = cvSmqrMcpUnif(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para)
    }
  } else if (kernel == "parabolic") {
    if (penalty == "lasso") {
      rst = cvSmqrLassoPara(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (penalty == "scad") {
      rst = cvSmqrScadPara(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para)
    } else {
      rst = cvSmqrMcpPara(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para)
    }
  } else {
    if (penalty == "lasso") {
      rst = cvSmqrLassoTrian(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax)
    } else if (penalty == "scad") {
      rst = cvSmqrScadTrian(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para)
    } else {
      rst = cvSmqrMcpTrian(X, Y, lambdaSeq, folds, tau, kfolds, h, phi0, gamma, epsilon, iteMax, iteTight, para)
    }
  }
  return (list(coeff = as.numeric(rst$coeff), lambda = rst$lambda, bandwidth = h, tau = tau, kernel = kernel, penalty = penalty, n = n, p = p))
}




