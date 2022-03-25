# include <RcppArmadillo.h>
# include <cmath>
# include "basicOp.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
double lossParaHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau, const double h, const double h1, const double h3) {
  arma::vec res = Y - Z * beta;
  arma::vec temp = (tau - 0.5) * res;
  for (int i = 0; i < res.size(); i++) {
    double cur = std::abs(res(i));
    temp(i) += cur <= h ? (0.375 * h1 * cur * cur - 0.0625 * h3 * cur * cur * cur * cur + 0.1875 * h) : 0.5 * cur;
  }
  return arma::mean(temp);
}

// [[Rcpp::export]]
double updateParaHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, arma::vec& grad, const double tau, const double n1, const double h, 
                    const double h1, const double h3) {
  arma::vec res = Y - Z * beta;
  arma::vec temp = (tau - 0.5) * res;
  arma::vec der(res.size());
  for (int i = 0; i < res.size(); i++) {
    double cur = res(i);
    if (cur <= -h) {
      der(i) = 1 - tau;
      temp(i) -= 0.5 * cur;
    } else if (cur < h) {
      der(i) =  0.5 - tau - 0.75 * h1 * cur + 0.25 * h3 * cur * cur * cur;
      temp(i) += 0.375 * h1 * cur * cur - 0.0625 * h3 * cur * cur * cur * cur + 0.1875 * h;
    } else {
      der(i) = -tau;
      temp(i) += 0.5 * cur;
    }
  }
  grad = n1 * Z.t() * der;
  return arma::mean(temp);
}

// [[Rcpp::export]]
double lammParaLasso(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double tau, const double phi, 
                     const double gamma, const int p, const double h, const double n1, const double h1, const double h3) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateParaHd(Z, Y, beta, grad, tau, n1, h, h1, h3);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = lossParaHd(Z, Y, betaNew, tau, h, h1, h3);
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
double lammParaElastic(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double tau, const double alpha, 
                       const double phi, const double gamma, const int p, const double h, const double n1, const double h1, const double h3) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateParaHd(Z, Y, beta, grad, tau, n1, h, h1, h3);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = alpha * Lambda / phiNew;
    betaNew = softThresh(first, second, p) / (1.0 + (2.0 - 2 * alpha) * Lambda / phiNew);
    double fVal = lossParaHd(Z, Y, betaNew, tau, h, h1, h3);
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
double lammParaGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, arma::vec& beta, const double tau, const arma::vec& group, 
                          const arma::vec& weight, const double phi, const double gamma, const int p, const int G, const double h, const double n1, 
                          const double h1, const double h3) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateParaHd(Z, Y, beta, grad, tau, n1, h, h1, h3);
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
    double fVal = lossParaHd(Z, Y, betaNew, tau, h, h1, h3);
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
double lammParaSparseGroupLasso(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, const double lambda, arma::vec& beta, const double tau, 
                                const arma::vec& group, const arma::vec& weight, const double phi, const double gamma, const int p, const int G, 
                                const double h, const double n1, const double h1, const double h3) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateParaHd(Z, Y, beta, grad, tau, n1, h, h1, h3);
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
    double fVal = lossParaHd(Z, Y, betaNew, tau, h, h1, h3);
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
arma::vec paraLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                    const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
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
    phi = lammParaLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec paraLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                        const double n1, const double h, const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, 
                        const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammParaLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec paraElastic(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const double alpha, const int p, const double n1, 
                      const double h, const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, 
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
    phi = lammParaElastic(Z, Y, Lambda, betaNew, tau, alpha, phi, gamma, p, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec paraElasticWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const double alpha, 
                          const int p, const double n1, const double h, const double h1, const double h3, const double phi0 = 0.01, 
                          const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = lambda * arma::ones(p + 1);
  Lambda(0) = 0;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammParaElastic(Z, Y, Lambda, betaNew, tau, alpha, phi, gamma, p, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec paraGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const arma::vec& group, 
                         const arma::vec& weight, const int p, const int G, const double n1, const double h, const double h1, const double h3, 
                         const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec betaNew = beta;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammParaGroupLasso(Z, Y, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec paraGroupLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, 
                             const arma::vec& group, const arma::vec& weight, const int p, const int G, const double n1, const double h, 
                             const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                             const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammParaGroupLasso(Z, Y, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec paraSparseGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const arma::vec& group, 
                               const arma::vec& weight, const int p, const int G, const double n1, const double h, const double h1, const double h3, 
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
    phi = lammParaSparseGroupLasso(Z, Y, Lambda, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec paraSparseGroupLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const arma::vec& group, 
                                   const arma::vec& weight, const int p, const int G, const double n1, const double h, const double h1, const double h3, 
                                   const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = lambda * arma::ones(p + 1);
  Lambda(0) = 0;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammParaSparseGroupLasso(Z, Y, Lambda, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec paraScad(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                   const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
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
    phi = lammParaLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h3);
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
      phi = lammParaLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h3);
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
arma::vec paraScadWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                       const double n1, const double h, const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, 
                       const double epsilon = 0.001, const int iteMax = 500, const double para = 3.7) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammParaLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec paraMcp(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                  const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
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
    phi = lammParaLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h3);
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
      phi = lammParaLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h3);
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
arma::vec paraMcpWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                      const double n1, const double h, const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, 
                      const double epsilon = 0.001, const int iteMax = 500, const double para = 3) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammParaLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec conquerParaLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                           const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = paraLasso(Z, Y, lambda, tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerParaLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, const double phi0 = 0.01, 
                              const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = paraLasso(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = paraLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, h3, phi0, gamma, epsilon, iteMax);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerParaElastic(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double alpha, const double h, 
                             const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = paraElastic(Z, Y, lambda, tau, alpha, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerParaElasticSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double alpha, const double h, 
                                const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = paraElastic(Z, Y, lambdaSeq(0), tau, alpha, p, n1, h, h1, h3, phi0, gamma, epsilon, iteMax);
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = paraElasticWarm(Z, Y, lambdaSeq(i), betaWarm, tau, alpha, p, n1, h, h1, h3, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerParaGroupLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const arma::vec& group, const arma::vec& weight, const int G, 
                                const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = paraGroupLasso(Z, Y, lambda, tau, group, weight, p, G, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerParaGroupLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const arma::vec& group, const arma::vec& weight, const int G, 
                                   const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = paraGroupLasso(Z, Y, lambdaSeq(0), tau, group, weight, p, G, n1, h, h1, h3, phi0, gamma, epsilon, iteMax);
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = paraGroupLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1, h, h1, h3, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerParaSparseGroupLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const arma::vec& group, const arma::vec& weight, const int G, 
                                      const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                                      const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = paraSparseGroupLasso(Z, Y, lambda, tau, group, weight, p, G, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerParaSparseGroupLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const arma::vec& group, const arma::vec& weight, 
                                         const int G, const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                                         const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = paraSparseGroupLasso(Z, Y, lambdaSeq(0), tau, group, weight, p, G, n1, h, h1, h3, phi0, gamma, epsilon, iteMax);
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = paraSparseGroupLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1, h, h1, h3, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerParaScad(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                          const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = paraScad(Z, Y, lambda, tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerParaScadSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, const double phi0 = 0.01, 
                             const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, 
                             const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = paraScad(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = paraScadWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, h3, phi0, gamma, epsilon, iteMax, para);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerParaMcp(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                         const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.0) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat = paraMcp(Z, Y, lambda, tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerParaMcpSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, const double phi0 = 0.01, 
                            const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.0) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h), n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::mat betaSeq(p + 1, nlambda);
  arma::vec betaHat = paraMcp(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = paraMcpWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, h3, phi0, gamma, epsilon, iteMax, para);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.rows(1, p).each_col() %= sx1;
  betaSeq.row(0) += my - mx * betaSeq.rows(1, p);
  return betaSeq;
}

// [[Rcpp::export]]
Rcpp::List cvParaLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                           const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
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
    betaHat = paraLasso(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = paraLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = paraLasso(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = paraLasso(Z, Y, lambdaSeq(seIdx), tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvParaElasticWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const double alpha, 
                             const int kfolds, const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                             const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
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
    betaHat = paraElastic(trainZ, trainY, lambdaSeq(0), tau, alpha, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = paraElasticWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, alpha, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = paraElastic(Z, Y, lambdaSeq(cvIdx), tau, alpha, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = paraElastic(Z, Y, lambdaSeq(seIdx), tau, alpha, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvParaGroupLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                                const arma::vec& group, const arma::vec& weight, const int G, const double h, const double phi0 = 0.01, const double gamma = 1.2, 
                                const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
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
    betaHat = paraGroupLasso(trainZ, trainY, lambdaSeq(0), tau, group, weight, p, G, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = paraGroupLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = paraGroupLasso(Z, Y, lambdaSeq(cvIdx), tau, group, weight, p, G, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = paraGroupLasso(Z, Y, lambdaSeq(seIdx), tau, group, weight, p, G, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                                        Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvParaSparseGroupLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, 
                                      const int kfolds, const arma::vec& group, const arma::vec& weight, const int G, const double h, const double phi0 = 0.01, 
                                      const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
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
    betaHat = paraSparseGroupLasso(trainZ, trainY, lambdaSeq(0), tau, group, weight, p, G, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = paraSparseGroupLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = paraSparseGroupLasso(Z, Y, lambdaSeq(cvIdx), tau, group, weight, p, G, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = paraSparseGroupLasso(Z, Y, lambdaSeq(seIdx), tau, group, weight, p, G, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvParaScadWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                          const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500,
                          const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
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
    betaHat = paraScad(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = paraScadWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax, para);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = paraScad(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = paraScad(Z, Y, lambdaSeq(seIdx), tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvParaMcpWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                         const double h, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500,
                         const int iteTight = 3, const double para = 3) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
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
    betaHat = paraMcp(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = paraMcpWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax, para);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = paraMcp(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = paraMcp(Z, Y, lambdaSeq(seIdx), tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHatSe.rows(1, p) %= sx1;
  betaHatSe(0) += my - arma::as_scalar(mx * betaHatSe.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

