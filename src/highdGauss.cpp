# include <RcppArmadillo.h>
# include <cmath>
# include "basicOp.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
arma::vec cmptLambdaSCAD(const arma::vec& beta, const double lambda, const int p, const double para = 3.7) {
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
arma::vec cmptLambdaMCP(const arma::vec& beta, const double lambda, const int p, const double para = 3.0) {
  arma::vec rst = arma::zeros(p + 1);
  for (int i = 1; i <= p; i++) {
    double abBeta = std::abs(beta(i));
    if (abBeta <= para * lambda) {
      rst(i) = lambda - abBeta / para;
    }
  }
  return rst;
}

// [[Rcpp::export]]
double lammGaussLasso(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double phi, const double tau, 
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
    phi = lammGaussLasso(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
    phi = lammGaussLasso(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
      phi = lammGaussLasso(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
    phi = lammGaussLasso(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
      phi = lammGaussLasso(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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

// high-dim conquer with a specified lambda
// [[Rcpp::export]]
arma::vec conquerHdGauss(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const int type = 1, const double phi0 = 0.01, 
                         const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat(p + 1);
  if (type == 1) {
    betaHat = gaussLasso(Z, Y, lambda, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  } else if (type == 2) {
    betaHat = gaussScad(Z, Y, lambda, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  } else {
    betaHat = gaussMcp(Z, Y, lambda, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  }
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// cross-validation, the range of lambda is guided by the simulation-based mathod in Belloni & Chernozhukov (2011), AOS
// [[Rcpp::export]]
Rcpp::List cvGaussLasso(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                        const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
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
      betaHat = gaussLasso(trainZ, trainY, lambdaSeq(i), tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = gaussLasso(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvGaussScad(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                       const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500,
                       const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
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
      betaHat = gaussScad(trainZ, trainY, lambdaSeq(i), tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = gaussScad(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}

// [[Rcpp::export]]
Rcpp::List cvGaussMcp(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                          const double h, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500,
                          const int iteTight = 3, const double para = 3) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
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
      betaHat = gaussMcp(trainZ, trainY, lambdaSeq(i), tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += lossQr(testZ, testY, betaHat, tau);
    }
  }
  mse /= n;
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = gaussMcp(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx), Rcpp::Named("deviance") = mse);
}
