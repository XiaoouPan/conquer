# include <RcppArmadillo.h>
# include <cmath>
# include "basicOp.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
double lossUnifHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau, const double h, const double h1) {
  arma::vec res = Y - Z * beta;
  arma::vec temp = (tau - 0.5) * res;
  for (int i = 0; i < res.size(); i++) {
    double cur = std::abs(res(i));
    temp(i) += cur <= h ? (0.25 * h1 * cur * cur + 0.25 * h) : 0.5 * cur;
  }
  return arma::mean(temp);
}

// [[Rcpp::export]]
double updateUnifHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, arma::vec& grad, const double tau, const double n1, const double h, 
                    const double h1) {
  arma::vec res = Y - Z * beta;
  arma::vec temp = (tau - 0.5) * res;
  arma::vec der(res.size());
  for (int i = 0; i < res.size(); i++) {
    double cur = res(i);
    if (cur <= -h) {
      der(i) = 1 - tau;
      temp(i) -= 0.5 * cur;
    } else if (cur < h) {
      der(i) = 0.5 - tau - 0.5 * h1 * cur;
      temp(i) += 0.25 * h1 * cur * cur + 0.25 * h;
    } else {
      der(i) = -tau;
      temp(i) += 0.5 * cur;
    }
  }
  grad = n1 * Z.t() * der;
  return arma::mean(temp);
}

// [[Rcpp::export]]
double lammUnifLasso(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double tau, const double phi, 
                     const double gamma, const int p, const double h, const double n1, const double h1) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateUnifHd(Z, Y, beta, grad, tau, n1, h, h1);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = lossUnifHd(Z, Y, betaNew, tau, h, h1);
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
double lammUnifElastic(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double tau, const double alpha, 
                       const double phi, const double gamma, const int p, const double h, const double n1, const double h1) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateUnifHd(Z, Y, beta, grad, tau, n1, h, h1);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = alpha * Lambda / phiNew;
    betaNew = softThresh(first, second, p) / (1.0 + (1.0 - alpha) * Lambda / phiNew);
    double fVal = lossUnifHd(Z, Y, betaNew, tau, h, h1);
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
double lammUnifGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, arma::vec& beta, const double tau, const arma::vec& group, 
                          const arma::vec& weight, const double phi, const double gamma, const int p, const int G, const double h, const double n1, 
                          const double h1) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateUnifHd(Z, Y, beta, grad, tau, n1, h, h1);
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
    double fVal = lossUnifHd(Z, Y, betaNew, tau, h, h1);
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
double lammUnifSparseGroupLasso(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, const double lambda, arma::vec& beta, const double tau, 
                                const arma::vec& group, const arma::vec& weight, const double phi, const double gamma, const int p, const int G, 
                                const double h, const double n1, const double h1) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateUnifHd(Z, Y, beta, grad, tau, n1, h, h1);
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
    double fVal = lossUnifHd(Z, Y, betaNew, tau, h, h1);
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
arma::vec unifLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                    const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammUnifLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec unifLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                        const double n1, const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, 
                        const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammUnifLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec unifElastic(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const double alpha, const int p, const double n1, 
                      const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
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
    phi = lammUnifElastic(Z, Y, Lambda, betaNew, tau, alpha, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec unifElasticWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const double alpha, 
                          const int p, const double n1, const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, 
                          const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = lambda * arma::ones(p + 1);
  Lambda(0) = 0;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammUnifElastic(Z, Y, Lambda, betaNew, tau, alpha, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec unifGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const arma::vec& group, 
                         const arma::vec& weight, const int p, const int G, const double n1, const double h, const double h1,
                         const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec betaNew = beta;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammUnifGroupLasso(Z, Y, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec unifGroupLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, 
                             const arma::vec& group, const arma::vec& weight, const int p, const int G, const double n1, const double h, 
                             const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                             const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammUnifGroupLasso(Z, Y, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec unifSparseGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const arma::vec& group, 
                               const arma::vec& weight, const int p, const int G, const double n1, const double h, const double h1, 
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
    phi = lammUnifSparseGroupLasso(Z, Y, Lambda, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec unifSparseGroupLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, 
                                   const arma::vec& group, const arma::vec& weight, const int p, const int G, const double n1, const double h, 
                                   const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = lambda * arma::ones(p + 1);
  Lambda(0) = 0;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammUnifSparseGroupLasso(Z, Y, Lambda, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec unifScad(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                   const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, 
                   const int iteTight = 3, const double para = 3.7) {
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
    phi = lammUnifLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
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
      phi = lammUnifLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
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
arma::vec unifScadWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                       const double n1, const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, 
                       const double epsilon = 0.001, const int iteMax = 500, const double para = 3.7) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammUnifLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec unifMcp(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                  const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, 
                  const int iteTight = 3, const double para = 3) {
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
    phi = lammUnifLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
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
      phi = lammUnifLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
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
arma::vec unifMcpWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                      const double n1, const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, 
                      const double epsilon = 0.001, const int iteMax = 500, const double para = 3) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammUnifLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec conquerUnifLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                           const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = unifLasso(Z, Y, lambda, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerUnifLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, const double phi0 = 0.01, 
                              const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = unifLasso(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = unifLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerUnifElastic(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double alpha, const double h, 
                             const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = unifElastic(Z, Y, lambda, tau, alpha, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerUnifElasticSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double alpha, const double h, 
                                const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = unifElastic(Z, Y, lambdaSeq(0), tau, alpha, p, n1, h, h1, phi0, gamma, epsilon, iteMax);
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = unifElasticWarm(Z, Y, lambdaSeq(i), betaWarm, tau, alpha, p, n1, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerUnifGroupLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const arma::vec& group, const int G, 
                                const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec weight = arma::zeros(G);
  for (int i = 1; i <= p; i++) {
    weight(group(i)) += 1;
  }
  weight = arma::sqrt(weight);
  arma::vec betaHat = unifGroupLasso(Z, Y, lambda, tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerUnifGroupLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const arma::vec& group, const int G, 
                                   const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec weight = arma::zeros(G);
  for (int i = 1; i <= p; i++) {
    weight(group(i)) += 1;
  }
  weight = arma::sqrt(weight);
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = unifGroupLasso(Z, Y, lambdaSeq(0), tau, group, weight, p, G, n1, h, h1, phi0, gamma, epsilon, iteMax);
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = unifGroupLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerUnifSparseGroupLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const arma::vec& group, const int G, 
                                      const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                                      const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec weight = arma::zeros(G);
  for (int i = 1; i <= p; i++) {
    weight(group(i)) += 1;
  }
  weight = arma::sqrt(weight);
  arma::vec betaHat = unifSparseGroupLasso(Z, Y, lambda, tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerUnifSparseGroupLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const arma::vec& group, 
                                         const int G, const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                                         const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec weight = arma::zeros(G);
  for (int i = 1; i <= p; i++) {
    weight(group(i)) += 1;
  }
  weight = arma::sqrt(weight);
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = unifSparseGroupLasso(Z, Y, lambdaSeq(0), tau, group, weight, p, G, n1, h, h1, phi0, gamma, epsilon, iteMax);
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = unifSparseGroupLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerUnifScad(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                          const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = unifScad(Z, Y, lambda, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerUnifScadSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, 
                             const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, 
                             const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = unifScad(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = unifScadWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax, para);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerUnifMcp(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                         const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.0) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = unifMcp(Z, Y, lambda, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerUnifMcpSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, 
                            const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, 
                            const int iteTight = 3, const double para = 3.0) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = unifMcp(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = unifMcpWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax, para);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
Rcpp::List cvUnifLasso(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                       const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1);
  arma::vec mse = arma::zeros(nlambda);
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
    for (int i = 0; i < nlambda; i++) {
      betaHat = unifLasso(trainZ, trainY, lambdaSeq(i), tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifLasso(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                           const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec mse = arma::zeros(nlambda);
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
    betaHat = unifLasso(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    mse(0) += lossQr(testZ, testY, betaHat, tau);
    for (int i = 1; i < nlambda; i++) {
      betaHat = unifLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
      betaWarm = betaHat;
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifLasso(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifElastic(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const double alpha, 
                         const int kfolds, const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                         const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1);
  arma::vec mse = arma::zeros(nlambda);
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
    for (int i = 0; i < nlambda; i++) {
      betaHat = unifElastic(trainZ, trainY, lambdaSeq(i), tau, alpha, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifElastic(Z, Y, lambdaSeq(cvIdx), tau, alpha, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifElasticWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const double alpha, 
                             const int kfolds, const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                             const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec mse = arma::zeros(nlambda);
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
    betaHat = unifElastic(trainZ, trainY, lambdaSeq(0), tau, alpha, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    mse(0) += lossQr(testZ, testY, betaHat, tau);
    for (int i = 1; i < nlambda; i++) {
      betaHat = unifElasticWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, alpha, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
      betaWarm = betaHat;
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifElastic(Z, Y, lambdaSeq(cvIdx), tau, alpha, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifGroupLasso(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                            const arma::vec& group, const int G, const double h, const double phi0 = 0.01, const double gamma = 1.2, 
                            const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1);
  arma::vec mse = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec weight = arma::zeros(G);
  for (int i = 1; i <= p; i++) {
    weight(group(i)) += 1;
  }
  weight = arma::sqrt(weight);
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    for (int i = 0; i < nlambda; i++) {
      betaHat = unifGroupLasso(trainZ, trainY, lambdaSeq(i), tau, group, weight, p, G, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifGroupLasso(Z, Y, lambdaSeq(cvIdx), tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifGroupLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                                const arma::vec& group, const int G, const double h, const double phi0 = 0.01, const double gamma = 1.2, 
                                const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec mse = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec weight = arma::zeros(G);
  for (int i = 1; i <= p; i++) {
    weight(group(i)) += 1;
  }
  weight = arma::sqrt(weight);
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = unifGroupLasso(trainZ, trainY, lambdaSeq(0), tau, group, weight, p, G, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    mse(0) += lossQr(testZ, testY, betaHat, tau);
    for (int i = 1; i < nlambda; i++) {
      betaHat = unifGroupLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
      betaWarm = betaHat;
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifGroupLasso(Z, Y, lambdaSeq(cvIdx), tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifSparseGroupLasso(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, 
                                  const int kfolds, const arma::vec& group, const int G, const double h, const double phi0 = 0.01, 
                                  const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1);
  arma::vec mse = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec weight = arma::zeros(G);
  for (int i = 1; i <= p; i++) {
    weight(group(i)) += 1;
  }
  weight = arma::sqrt(weight);
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    for (int i = 0; i < nlambda; i++) {
      betaHat = unifSparseGroupLasso(trainZ, trainY, lambdaSeq(i), tau, group, weight, p, G, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifSparseGroupLasso(Z, Y, lambdaSeq(cvIdx), tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifSparseGroupLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, 
                                      const int kfolds, const arma::vec& group, const int G, const double h, const double phi0 = 0.01, 
                                      const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec mse = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec weight = arma::zeros(G);
  for (int i = 1; i <= p; i++) {
    weight(group(i)) += 1;
  }
  weight = arma::sqrt(weight);
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = unifSparseGroupLasso(trainZ, trainY, lambdaSeq(0), tau, group, weight, p, G, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    mse(0) += lossQr(testZ, testY, betaHat, tau);
    for (int i = 1; i < nlambda; i++) {
      betaHat = unifSparseGroupLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
      betaWarm = betaHat;
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifSparseGroupLasso(Z, Y, lambdaSeq(cvIdx), tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifScad(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                      const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500,
                      const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1);
  arma::vec mse = arma::zeros(nlambda);
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
    for (int i = 0; i < nlambda; i++) {
      betaHat = unifScad(trainZ, trainY, lambdaSeq(i), tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifScad(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifScadWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                          const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500,
                          const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec mse = arma::zeros(nlambda);
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
    betaHat = unifScad(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
    betaWarm = betaHat;
    mse(0) += lossQr(testZ, testY, betaHat, tau);
    for (int i = 1; i < nlambda; i++) {
      betaHat = unifScadWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, para);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
      betaWarm = betaHat;
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifScad(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifMcp(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                     const double h, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500,
                     const int iteTight = 3, const double para = 3) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1);
  arma::vec mse = arma::zeros(nlambda);
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
    for (int i = 0; i < nlambda; i++) {
      betaHat = unifMcp(trainZ, trainY, lambdaSeq(i), tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifMcp(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvUnifMcpWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                         const double h, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500,
                         const int iteTight = 3, const double para = 3) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p + 1), betaWarm(p + 1);
  arma::vec mse = arma::zeros(nlambda);
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
    betaHat = unifMcp(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
    betaWarm = betaHat;
    mse(0) += lossQr(testZ, testY, betaHat, tau);
    for (int i = 1; i < nlambda; i++) {
      betaHat = unifMcpWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, para);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
      betaWarm = betaHat;
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = unifMcp(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}


