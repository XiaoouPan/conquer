# include <RcppArmadillo.h>
# include <cmath>
# include "basicOp.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
int sgn(const double x) {
  return (x > 0) - (x < 0);
}

// [[Rcpp::export]]
double mad(const arma::vec& x) {
  return 1.482602 * arma::median(arma::abs(x - arma::median(x)));
}

// [[Rcpp::export]]
arma::mat standardize(arma::mat X, const arma::rowvec& mx, const arma::vec& sx1, const int p) {
  for (int i = 0; i < p; i++) {
    X.col(i) = (X.col(i) - mx(i)) * sx1(i);
  }
  return X;
}

// [[Rcpp::export]]
arma::vec softThresh(const arma::vec& x, const arma::vec& Lambda, const int p) {
  return arma::sign(x) % arma::max(arma::abs(x) - Lambda, arma::zeros(p + 1));
}

// [[Rcpp::export]]
void lossQr(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau, const int i, arma::vec& dev, arma::vec& devsq) {
  arma::vec res = Y - Z * beta;
  for (int k = 0; k < res.size(); k++) {
    double temp =  res(k) >= 0 ? tau * res(k) : (tau - 1) * res(k);
    dev(i) += temp;
    devsq(i) += temp * temp;
  }
}

// [[Rcpp::export]]
arma::vec cmptLambdaLasso(const double lambda, const int p) {
  arma::vec rst = lambda * arma::ones(p + 1);
  rst(0) = 0;
  return rst;
}

// [[Rcpp::export]]
double lossL2(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double n1, const double tau) {
  arma::vec res = Y - Z * beta;
  double rst = 0.0;
  for (int i = 0; i < Y.size(); i++) {
    rst += (res(i) > 0) ? (tau * res(i) * res(i)) : ((1 - tau) * res(i) * res(i));
  }
  return 0.5 * n1 * rst;
}

// [[Rcpp::export]]
double updateL2(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, arma::vec& grad, const double n1, const double tau) {
  arma::vec res = Y - Z * beta;
  double rst = 0.0;
  grad = arma::zeros(grad.size());
  for (int i = 0; i < Y.size(); i++) {
    double temp = res(i) > 0 ? tau : (1 - tau);
    grad -= temp * res(i) * Z.row(i).t();
    rst += temp * res(i) * res(i);
  }
  grad *= n1;
  return 0.5 * n1 * rst;
}

// LAMM code for initialization
// [[Rcpp::export]]
double lammL2(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double tau, const double phi, const double gamma, 
              const int p, const double n1) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateL2(Z, Y, beta, grad, n1, tau);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = lossL2(Z, Y, betaNew, n1, tau);
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
arma::vec lasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double phi0, 
                const double gamma, const double epsilon, const int iteMax) {
  arma::vec beta = arma::zeros(p + 1);
  arma::vec betaNew = arma::zeros(p + 1);
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammL2(Z, Y, Lambda, betaNew, tau, phi, gamma, p, n1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec cmptLambdaSCAD(const arma::vec& beta, const double lambda, const int p, const double para) {
  arma::vec rst = arma::zeros(p + 1);
  for (int i = 1; i <= p; i++) {
    double abBeta = std::abs(beta(i));
    if (abBeta <= lambda) {
      rst(i) = lambda;
    } else if (abBeta <= para * lambda) {
      rst(i) = (para * lambda - abBeta) / (para - 1);
    } 
  }
  return rst;
}

// [[Rcpp::export]]
arma::vec cmptLambdaMCP(const arma::vec& beta, const double lambda, const int p, const double para) {
  arma::vec rst = arma::zeros(p + 1);
  for (int i = 1; i <= p; i++) {
    double abBeta = std::abs(beta(i));
    if (abBeta <= para * lambda) {
      rst(i) = lambda - abBeta / para;
    }
  }
  return rst;
}

