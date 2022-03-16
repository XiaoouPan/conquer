#ifndef BASICOP_H
#define BASICOP_H

# include <RcppArmadillo.h>
# include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

int sgn(const double x);

double mad(const arma::vec& x);

arma::mat standardize(arma::mat X, const arma::rowvec& mx, const arma::vec& sx1, const int p);

arma::vec softThresh(const arma::vec& x, const arma::vec& Lambda, const int p);

double lossQr(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau);

arma::vec cmptLambdaLasso(const double lambda, const int p);

double lossL2(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double n1, const double tau);

double updateL2(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, arma::vec& grad, const double n1, const double tau);

double lammL2(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double tau, const double phi, const double gamma, 
              const int p, const double n1);

arma::vec lasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double phi0 = 0.1, 
                const double gamma = 1.2, const double epsilon = 0.01, const int iteMax = 500);

arma::vec cmptLambdaSCAD(const arma::vec& beta, const double lambda, const int p, const double para = 3.7);

arma::vec cmptLambdaMCP(const arma::vec& beta, const double lambda, const int p, const double para = 3.0);

#endif