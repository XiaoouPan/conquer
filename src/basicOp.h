#ifndef BASICOP_H
#define BASICOP_H

# include <RcppArmadillo.h>
# include <cmath>
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
double lossQr(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau) {
  arma::vec res = Y - Z * beta;
  double rst = 0.0;
  for (int i = 0; i < res.size(); i++) {
    rst += res(i) >= 0 ? tau * res(i) : (tau - 1) * res(i);
  }
  return rst;
}

#endif