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

// Asymmetric huber regression adjusted to quantile tau for initialization 
// [[Rcpp::export]]
void updateHuber(const arma::mat& Z, const arma::vec& res, const double tau, arma::vec& der, arma::vec& grad, const int n, const double rob, const double n1) {
  for (int i = 0; i < n; i++) {
    double cur = res(i);
    if (cur > rob) {
      der(i) = -tau * rob;
    } else if (cur > 0) {
      der(i) = -tau * cur;
    } else if (cur > -rob) {
      der(i) = (tau - 1) * cur;
    } else {
      der(i) = (1 - tau) * rob;
    }
  }
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
arma::vec huberReg(const arma::mat& Z, const arma::vec& Y, const double tau, arma::vec& der, arma::vec& gradOld, arma::vec& gradNew, const int n, const int p, 
                   const double n1, const double tol = 0.0001, const double constTau = 1.345, const int iteMax = 5000) {
  double rob = constTau * mad(Y);
  updateHuber(Z, Y, tau, der, gradOld, n, rob, n1);
  arma::vec beta = -gradOld, betaDiff = -gradOld;
  arma::vec res = Y - Z * beta;
  rob = constTau * mad(res);
  updateHuber(Z, res, tau, der, gradNew, n, rob, n1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    rob = constTau * mad(res);
    updateHuber(Z, res, tau, der, gradNew, n, rob, n1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  return beta;
}

// [[Rcpp::export]]
arma::mat standardize(arma::mat X, const arma::rowvec& mx, const arma::vec& sx1, const int p) {
  for (int i = 0; i < p; i++) {
    X.col(i) = (X.col(i) - mx(i)) * sx1(i);
  }
  return X;
}

// Different kernels for low-dimensional conquer 
// [[Rcpp::export]]
void updateGauss(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const double tau, const double n1, const double h1) {
  der = arma::normcdf(-res * h1) - tau;
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
void updateGaussWeight(const arma::mat& Z, const arma::vec& weight, const arma::vec& res, arma::vec& der, arma::vec& grad, const double tau, 
                       const double n1, const double h1) {
  der = weight % (arma::normcdf(-res * h1) - tau);
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
void updateLogistic(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const double tau, const double n1, const double h1) {
  der = 1.0 / (1 + arma::exp(res * h1)) - tau;
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
void updateUnif(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const int n, const double tau, const double h, 
                const double n1, const double h1) {
  for (int i = 0; i < n; i++) {
    double cur = res(i);
    if (cur <= -h) {
      der(i) = 1 - tau;
    } else if (cur < h) {
      der(i) = 0.5 - tau - 0.5 * h1 * cur;
    } else {
      der(i) = -tau;
    }
  }
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
void updatePara(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const int n, const double tau, const double h, 
                const double n1, const double h1, const double h3) {
  for (int i = 0; i < n; i++) {
    double cur = res(i);
    if (cur <= -h) {
      der(i) = 1 - tau;
    } else if (cur < h) {
      der(i) = 0.5 - tau - 0.75 * h1 * cur + 0.25 * h3 * cur * cur * cur;
    } else {
      der(i) = -tau;
    }
  }
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
void updateTrian(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const int n, const double tau, const double h, 
                 const double n1, const double h1, const double h2) {
  for (int i = 0; i < n; i++) {
    double cur = res(i);
    if (cur <= -h) {
      der(i) = 1 - tau;
    } else if (cur < 0) {
      der(i) = 0.5 - tau - h1 * cur - 0.5 * h2 * cur * cur;
    } else if (cur < h) {
      der(i) = 0.5 - tau - h1 * cur + 0.5 * h2 * cur * cur;
    } else {
      der(i) = -tau;
    }
  }
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
Rcpp::List smqrGauss(const arma::mat& X, arma::vec Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                     const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  updateGauss(Z, res, der, gradOld, tau, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateGauss(Z, res, der, gradNew, tau, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateGauss(Z, res, der, gradNew, tau, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
Rcpp::List smqrGaussNsd(const arma::mat& Z, const arma::vec& Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                        const double tol = 0.0001, const int iteMax = 5000) {
  const int n = Z.n_rows;
  const int p = Z.n_cols - 1;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  updateGauss(Z, res, der, gradOld, tau, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateGauss(Z, res, der, gradNew, tau, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateGauss(Z, res, der, gradNew, tau, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
arma::vec smqrGaussIni(const arma::mat& X, arma::vec Y, const arma::vec& betaHat, const int p, const double tau = 0.5, double h = 0.0,
                       const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = betaHat;
  arma::vec res = Y - Z * beta;
  updateGauss(Z, res, der, gradOld, tau, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateGauss(Z, res, der, gradNew, tau, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateGauss(Z, res, der, gradNew, tau, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return beta;
}

// [[Rcpp::export]]
arma::vec smqrGaussIniWeight(const arma::mat& X, arma::vec Y, const arma::vec& weight, const arma::vec& betaHat, const int p, const double tau = 0.5, 
                             double h = 0.0, const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = betaHat;
  arma::vec res = Y - Z * beta;
  updateGaussWeight(Z, weight, res, der, gradOld, tau, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateGaussWeight(Z, weight, res, der, gradNew, tau, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateGaussWeight(Z, weight, res, der, gradNew, tau, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return beta;
}

// [[Rcpp::export]]
Rcpp::List smqrLogistic(const arma::mat& X, arma::vec Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                        const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  updateLogistic(Z, res, der, gradOld, tau, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateLogistic(Z, res, der, gradNew, tau, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateLogistic(Z, res, der, gradNew, tau, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
Rcpp::List smqrLogisticNsd(const arma::mat& Z, const arma::vec& Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                           const double tol = 0.0001, const int iteMax = 5000) {
  const int n = Z.n_rows;
  const int p = Z.n_cols - 1;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  updateLogistic(Z, res, der, gradOld, tau, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateLogistic(Z, res, der, gradNew, tau, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateLogistic(Z, res, der, gradNew, tau, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
arma::vec smqrLogisticIni(const arma::mat& X, arma::vec Y, const arma::vec& betaHat, const int p, const double tau = 0.5, double h = 0.0,
                          const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = betaHat;
  arma::vec res = Y - Z * beta;
  updateLogistic(Z, res, der, gradOld, tau, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateLogistic(Z, res, der, gradNew, tau, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateLogistic(Z, res, der, gradNew, tau, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return beta;
}

// [[Rcpp::export]]
Rcpp::List smqrUnif(const arma::mat& X, arma::vec Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                    const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  updateUnif(Z, res, der, gradOld, n, tau, h, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateUnif(Z, res, der, gradNew, n, tau, h, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateUnif(Z, res, der, gradNew, n, tau, h, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
Rcpp::List smqrUnifNsd(const arma::mat& Z, const arma::vec& Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                       const double tol = 0.0001, const int iteMax = 5000) {
  const int n = Z.n_rows;
  const int p = Z.n_cols - 1;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  updateUnif(Z, res, der, gradOld, n, tau, h, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateUnif(Z, res, der, gradNew, n, tau, h, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateUnif(Z, res, der, gradNew, n, tau, h, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
arma::vec smqrUnifIni(const arma::mat& X, arma::vec Y, const arma::vec& betaHat, const int p, const double tau = 0.5, double h = 0.0,
                      const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = betaHat;
  arma::vec res = Y - Z * beta;
  updateUnif(Z, res, der, gradOld, n, tau, h, n1, h1);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateUnif(Z, res, der, gradNew, n, tau, h, n1, h1);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateUnif(Z, res, der, gradNew, n, tau, h, n1, h1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return beta;
}
  
// [[Rcpp::export]]
Rcpp::List smqrPara(const arma::mat& X, arma::vec Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                    const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  updatePara(Z, res, der, gradOld, n, tau, h, n1, h1, h3);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updatePara(Z, res, der, gradNew, n, tau, h, n1, h1, h3);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updatePara(Z, res, der, gradNew, n, tau, h, n1, h1, h3);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
Rcpp::List smqrParaNsd(const arma::mat& Z, const arma::vec& Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                       const double tol = 0.0001, const int iteMax = 5000) {
  const int n = Z.n_rows;
  const int p = Z.n_cols - 1;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  updatePara(Z, res, der, gradOld, n, tau, h, n1, h1, h3);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updatePara(Z, res, der, gradNew, n, tau, h, n1, h1, h3);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updatePara(Z, res, der, gradNew, n, tau, h, n1, h1, h3);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
arma::vec smqrParaIni(const arma::mat& X, arma::vec Y, const arma::vec& betaHat, const int p, const double tau = 0.5, double h = 0.0,
                      const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = betaHat;
  arma::vec res = Y - Z * beta;
  updatePara(Z, res, der, gradOld, n, tau, h, n1, h1, h3);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updatePara(Z, res, der, gradNew, n, tau, h, n1, h1, h3);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updatePara(Z, res, der, gradNew, n, tau, h, n1, h1, h3);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return beta;
}

// [[Rcpp::export]]
Rcpp::List smqrTrian(const arma::mat& X, arma::vec Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                     const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  updateTrian(Z, res, der, gradOld, n, tau, h, n1, h1, h2);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateTrian(Z, res, der, gradNew, n, tau, h, n1, h1, h2);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateTrian(Z, res, der, gradNew, n, tau, h, n1, h1, h2);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
Rcpp::List smqrTrianNsd(const arma::mat& Z, const arma::vec& Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                        const double tol = 0.0001, const int iteMax = 5000) {
  const int n = Z.n_rows;
  const int p = Z.n_cols - 1;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  updateTrian(Z, res, der, gradOld, n, tau, h, n1, h1, h2);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateTrian(Z, res, der, gradNew, n, tau, h, n1, h1, h2);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateTrian(Z, res, der, gradNew, n, tau, h, n1, h1, h2);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
arma::vec smqrTrianIni(const arma::mat& X, arma::vec Y, const arma::vec& betaHat, const int p, const double tau = 0.5, double h = 0.0,
                       const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = betaHat;
  arma::vec res = Y - Z * beta;
  updateTrian(Z, res, der, gradOld, n, tau, h, n1, h1, h2);
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res -= Z * betaDiff;
  updateTrian(Z, res, der, gradNew, n, tau, h, n1, h1, h2);
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (arma::norm(gradNew, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    betaDiff = -alpha * gradNew;
    beta += betaDiff;
    res -= Z * betaDiff;
    updateTrian(Z, res, der, gradNew, n, tau, h, n1, h1, h2);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) %= sx1;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return beta;
}

// Codes for bootstrap inference
// [[Rcpp::export]]
arma::mat smqrGaussInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, double h = 0.0, const double tau = 0.5, 
                       const int B = 1000, const double tol = 0.0001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrGaussIni(mbX, mbY, betaHat, p, tau, h, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrGaussInfWeight(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, const double tau = 0.5, 
                             const int B = 1000, const double tol = 0.0001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  double h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  for (int b = 0; b < B; b++) {
    arma::vec weight = 2 * arma::randu(n);
    rst.col(b) = smqrGaussIniWeight(X, Y, weight, betaHat, p, tau, h, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrLogisticInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, double h = 0.0, const double tau = 0.5, 
                          const int B = 1000, const double tol = 0.0001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrLogisticIni(mbX, mbY, betaHat, p, tau, h, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrUnifInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, double h = 0.0, const double tau = 0.5, 
                      const int B = 1000, const double tol = 0.0001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrUnifIni(mbX, mbY, betaHat, p, tau, h, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrParaInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, double h = 0.0, const double tau = 0.5, 
                      const int B = 1000, const double tol = 0.0001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrParaIni(mbX, mbY, betaHat, p, tau, h, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrTrianInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, double h = 0.0, const double tau = 0.5, 
                       const int B = 1000, const double tol = 0.0001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrTrianIni(mbX, mbY, betaHat, p, tau, h, tol, iteMax);
  }
  return rst;
}

// High-dimensional conquer via an iterative local MM algorithm




