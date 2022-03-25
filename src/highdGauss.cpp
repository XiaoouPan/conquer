# include <RcppArmadillo.h>
# include <cmath>
# include "basicOp.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
double lossGaussHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau, const double h, const double h1, const double h2) {
  arma::vec res = Y - Z * beta;
  arma::vec temp = 0.3989423 * h  * arma::exp(-0.5 * h2 * arma::square(res)) + tau * res - res % arma::normcdf(-h1 * res);
  return arma::mean(temp);
}

// [[Rcpp::export]]
double updateGaussHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, arma::vec& grad, const double tau, const double n1, const double h, 
                     const double h1, const double h2) {
  arma::vec res = Y - Z * beta;
  arma::vec der = arma::normcdf(-h1 * res) - tau;
  grad = n1 * Z.t() * der;
  arma::vec temp = 0.3989423 * h  * arma::exp(-0.5 * h2 * arma::square(res)) + tau * res - res % arma::normcdf(-h1 * res);
  return arma::mean(temp);
}

// [[Rcpp::export]]
double lammGaussLasso(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double tau, const double phi, 
                      const double gamma, const int p, const double h, const double n1, const double h1, const double h2) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateGaussHd(Z, Y, beta, grad, tau, n1, h, h1, h2);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = lossGaussHd(Z, Y, betaNew, tau, h, h1, h2);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) + 0.5 * phiNew * arma::as_scalar(diff.t() * diff);
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  beta = betaNew;
  return phiNew;
}

// [[Rcpp::export]]
double lammGaussElastic(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double tau, const double alpha, 
                        const double phi, const double gamma, const int p, const double h, const double n1, const double h1, const double h2) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateGaussHd(Z, Y, beta, grad, tau, n1, h, h1, h2);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = alpha * Lambda / phiNew;
    betaNew = softThresh(first, second, p) / (1.0 + (2.0 - 2 * alpha) * Lambda / phiNew);
    double fVal = lossGaussHd(Z, Y, betaNew, tau, h, h1, h2);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) + 0.5 * phiNew * arma::as_scalar(diff.t() * diff);
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  beta = betaNew;
  return phiNew;
}

// [[Rcpp::export]]
double lammGaussGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, arma::vec& beta, const double tau, const arma::vec& group, 
                           const arma::vec& weight, const double phi, const double gamma, const int p, const int G, const double h, const double n1, 
                           const double h1, const double h2) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateGaussHd(Z, Y, beta, grad, tau, n1, h, h1, h2);
  while (true) {
    arma::vec subNorm = arma::zeros(G);
    betaNew = beta - grad / phiNew;
    for (int i = 1; i <= p; i++) {
      subNorm(group(i)) += betaNew(i) * betaNew(i);
    }
    subNorm = arma::max(1.0 - lambda * weight / (phiNew * arma::sqrt(subNorm)), arma::zeros(G));
    for (int i = 1; i <= p; i++) {
      betaNew(i) *= subNorm(group(i));
    }
    double fVal = lossGaussHd(Z, Y, betaNew, tau, h, h1, h2);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) + 0.5 * phiNew * arma::as_scalar(diff.t() * diff);
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  beta = betaNew;
  return phiNew;
}

// [[Rcpp::export]]
double lammGaussSparseGroupLasso(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, const double lambda, arma::vec& beta, const double tau, 
                                 const arma::vec& group, const arma::vec& weight, const double phi, const double gamma, const int p, const int G, 
                                 const double h, const double n1, const double h1, const double h2) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateGaussHd(Z, Y, beta, grad, tau, n1, h, h1, h2);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    arma::vec subNorm = arma::zeros(G);
    for (int i = 1; i <= p; i++) {
      subNorm(group(i)) += betaNew(i) * betaNew(i);
    }
    subNorm = arma::max(1.0 - lambda * weight / (phiNew * arma::sqrt(subNorm)), arma::zeros(G));
    for (int i = 1; i <= p; i++) {
      betaNew(i) *= subNorm(group(i));
    }
    double fVal = lossGaussHd(Z, Y, betaNew, tau, h, h1, h2);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) + 0.5 * phiNew * arma::as_scalar(diff.t() * diff);
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  beta = betaNew;
  return phiNew;
}

// High-dim conquer with a standardized design matrix and a given lambda
// [[Rcpp::export]]
arma::vec gaussLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                     const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                     const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                         const double n1, const double h, const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, 
                         const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussElastic(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const double alpha, const int p, const double n1, 
                       const double h, const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, 
                       const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussElastic(Z, Y, Lambda, betaNew, tau, alpha, phi, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussElasticWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const double alpha, 
                           const int p, const double n1, const double h, const double h1, const double h2, const double phi0 = 0.01, 
                           const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = lambda * arma::ones(p + 1);
  Lambda(0) = 0;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussElastic(Z, Y, Lambda, betaNew, tau, alpha, phi, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const arma::vec& group, 
                          const arma::vec& weight, const int p, const int G, const double n1, const double h, const double h1, const double h2, 
                          const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec betaNew = beta;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussGroupLasso(Z, Y, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussGroupLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, 
                              const arma::vec& group, const arma::vec& weight, const int p, const int G, const double n1, const double h, 
                              const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                              const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussGroupLasso(Z, Y, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussSparseGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const arma::vec& group, 
                                const arma::vec& weight, const int p, const int G, const double n1, const double h, const double h1, const double h2, 
                                const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec betaNew = beta;
  arma::vec Lambda = lambda * arma::ones(p + 1);
  Lambda(0) = 0;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussSparseGroupLasso(Z, Y, Lambda, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussSparseGroupLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const arma::vec& group, 
                                    const arma::vec& weight, const int p, const int G, const double n1, const double h, const double h1, const double h2, 
                                    const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = lambda * arma::ones(p + 1);
  Lambda(0) = 0;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussSparseGroupLasso(Z, Y, Lambda, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussScad(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                    const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                    const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 1;
  // Tightening
  arma::vec beta0(p + 1);
  while (iteT <= iteTight) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaSCAD(beta, lambda, p, para);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      phi = lammGaussLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h2);
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussScadWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                        const double n1, const double h, const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, 
                        const double epsilon = 0.001, const int iteMax = 500, const double para = 3.7) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussMcp(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                   const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                   const int iteMax = 500, const int iteTight = 3, const double para = 3) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 1;
  // Tightening
  arma::vec beta0(p + 1);
  while (iteT <= iteTight) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaMCP(beta, lambda, p, para);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      phi = lammGaussLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h2);
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec gaussMcpWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                       const double n1, const double h, const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, 
                       const double epsilon = 0.001, const int iteMax = 500, const double para = 3) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammGaussLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// high-dim conquer with a specified lambda or a sequence of lambda
// [[Rcpp::export]]
arma::vec conquerGaussLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                            const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = gaussLasso(Z, Y, lambda, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerGaussLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, const double phi0 = 0.01, 
                               const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = gaussLasso(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = gaussLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, h2, phi0, gamma, epsilon, iteMax);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerGaussElastic(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double alpha, const double h, 
                              const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = gaussElastic(Z, Y, lambda, tau, alpha, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerGaussElasticSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double alpha, const double h, 
                                 const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = gaussElastic(Z, Y, lambdaSeq(0), tau, alpha, p, n1, h, h1, h2, phi0, gamma, epsilon, iteMax);
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = gaussElasticWarm(Z, Y, lambdaSeq(i), betaWarm, tau, alpha, p, n1, h, h1, h2, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerGaussGroupLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const arma::vec& group, const arma::vec& weight, const int G, 
                                 const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = gaussGroupLasso(Z, Y, lambda, tau, group, weight, p, G, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerGaussGroupLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const arma::vec& group, const arma::vec& weight, const int G, 
                                    const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = gaussGroupLasso(Z, Y, lambdaSeq(0), tau, group, weight, p, G, n1, h, h1, h2, phi0, gamma, epsilon, iteMax);
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = gaussGroupLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1, h, h1, h2, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerGaussSparseGroupLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const arma::vec& group, const arma::vec& weight, const int G, 
                                       const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                                       const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = gaussSparseGroupLasso(Z, Y, lambda, tau, group, weight, p, G, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerGaussSparseGroupLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const arma::vec& group, const arma::vec& weight, 
                                          const int G, const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                                          const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = gaussSparseGroupLasso(Z, Y, lambdaSeq(0), tau, group, weight, p, G, n1, h, h1, h2, phi0, gamma, epsilon, iteMax);
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = gaussSparseGroupLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1, h, h1, h2, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerGaussScad(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                           const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = gaussScad(Z, Y, lambda, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerGaussScadSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, 
                              const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, 
                              const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = gaussScad(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = gaussScadWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, h2, phi0, gamma, epsilon, iteMax, para);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerGaussMcp(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                          const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.0) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = gaussMcp(Z, Y, lambda, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerGaussMcpSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, 
                             const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, 
                             const int iteTight = 3, const double para = 3.0) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = gaussMcp(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = gaussMcpWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, h2, phi0, gamma, epsilon, iteMax, para);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// cross-validation, the range of lambda is guided by the simulation-based mathod in Belloni & Chernozhukov (2011), AOS
// [[Rcpp::export]]
Rcpp::List cvGaussLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                            const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = gaussLasso(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = gaussLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = gaussLasso(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = gaussLasso(Z, Y, lambdaSeq(seIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvGaussElasticWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const double alpha, 
                              const int kfolds, const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                              const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = gaussElastic(trainZ, trainY, lambdaSeq(0), tau, alpha, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = gaussElasticWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, alpha, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = gaussElastic(Z, Y, lambdaSeq(cvIdx), tau, alpha, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = gaussElastic(Z, Y, lambdaSeq(seIdx), tau, alpha, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvGaussGroupLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                                 const arma::vec& group, const arma::vec& weight, const int G, const double h, const double phi0 = 0.01, const double gamma = 1.2, 
                                 const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = gaussGroupLasso(trainZ, trainY, lambdaSeq(0), tau, group, weight, p, G, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = gaussGroupLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = gaussGroupLasso(Z, Y, lambdaSeq(cvIdx), tau, group, weight, p, G, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = gaussGroupLasso(Z, Y, lambdaSeq(seIdx), tau, group, weight, p, G, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvGaussSparseGroupLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, 
                                       const int kfolds, const arma::vec& group, const arma::vec& weight, const int G, const double h, const double phi0 = 0.01, 
                                       const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = gaussSparseGroupLasso(trainZ, trainY, lambdaSeq(0), tau, group, weight, p, G, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = gaussSparseGroupLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = gaussSparseGroupLasso(Z, Y, lambdaSeq(cvIdx), tau, group, weight, p, G, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = gaussSparseGroupLasso(Z, Y, lambdaSeq(seIdx), tau, group, weight, p, G, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx),
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvGaussScadWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                           const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500,
                           const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = gaussScad(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = gaussScadWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax, para);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = gaussScad(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = gaussScad(Z, Y, lambdaSeq(seIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx),
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvGaussMcpWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                          const double h, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500,
                          const int iteTight = 3, const double para = 3) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = gaussMcp(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = gaussMcpWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax, para);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = gaussMcp(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = gaussMcp(Z, Y, lambdaSeq(seIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx),
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

