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
      der(i) = -2 * tau * rob;
    } else if (cur > 0) {
      der(i) = -2 * tau * cur;
    } else if (cur > -rob) {
      der(i) = 2 * (tau - 1) * cur;
    } else {
      der(i) = 2 * (1 - tau) * rob;
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
  der = 1.0 / (1.0 + arma::exp(res * h1)) - tau;
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

// Low-dimensional conquer: estimation
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

// Global conquer process with a quantile grid
// [[Rcpp::export]]
Rcpp::List smqrGaussProc(const arma::mat& X, arma::vec Y, const arma::vec tauSeq, double h = 0.0, const double constTau = 1.345, 
                         const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  const int m = tauSeq.size();
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
  arma::mat betaProc(p + 1, m);
  for (int i = 0; i < m; i++) {
    double tau = tauSeq(i);
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
    betaProc.col(i) = beta;
  }
  betaProc.rows(1, p).each_col() %= sx1;
  betaProc.row(0) += my - mx * betaProc.rows(1, p);
  return Rcpp::List::create(Rcpp::Named("coeff") = betaProc, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
Rcpp::List smqrLogisticProc(const arma::mat& X, arma::vec Y, const arma::vec tauSeq, double h = 0.0, const double constTau = 1.345, 
                            const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  const int m = tauSeq.size();
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
  arma::mat betaProc(p + 1, m);
  for (int i = 0; i < m; i++) {
    double tau = tauSeq(i);
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
    betaProc.col(i) = beta;
  }
  betaProc.rows(1, p).each_col() %= sx1;
  betaProc.row(0) += my - mx * betaProc.rows(1, p);
  return Rcpp::List::create(Rcpp::Named("coeff") = betaProc, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
Rcpp::List smqrUnifProc(const arma::mat& X, arma::vec Y, const arma::vec tauSeq, double h = 0.0, const double constTau = 1.345, 
                        const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  const int m = tauSeq.size();
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
  arma::mat betaProc(p + 1, m);
  for (int i = 0; i < m; i++) {
    double tau = tauSeq(i);
    arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
    arma::vec quant = {tau};
    beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
    arma::vec res = Y - Z * beta;
    updateUnif(Z, res, der, gradOld, n, tau, h, n1, h1);
    beta -= gradOld;
    arma::vec betaDiff = -gradOld;
    res -= Z * betaDiff;
    updateUnif(Z, res, der, gradOld, n, tau, h, n1, h1);
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
      updateUnif(Z, res, der, gradOld, n, tau, h, n1, h1);
      gradDiff = gradNew - gradOld;
      ite++;
    }
    betaProc.col(i) = beta;
  }
  betaProc.rows(1, p).each_col() %= sx1;
  betaProc.row(0) += my - mx * betaProc.rows(1, p);
  return Rcpp::List::create(Rcpp::Named("coeff") = betaProc, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
Rcpp::List smqrParaProc(const arma::mat& X, arma::vec Y, const arma::vec tauSeq, double h = 0.0, const double constTau = 1.345, 
                        const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  const int m = tauSeq.size();
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
  arma::mat betaProc(p + 1, m);
  for (int i = 0; i < m; i++) {
    double tau = tauSeq(i);
    arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
    arma::vec quant = {tau};
    beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
    arma::vec res = Y - Z * beta;
    updatePara(Z, res, der, gradOld, n, tau, h, n1, h1, h3);
    beta -= gradOld;
    arma::vec betaDiff = -gradOld;
    res -= Z * betaDiff;
    updatePara(Z, res, der, gradOld, n, tau, h, n1, h1, h3);
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
      updatePara(Z, res, der, gradOld, n, tau, h, n1, h1, h3);
      gradDiff = gradNew - gradOld;
      ite++;
    }
    betaProc.col(i) = beta;
  }
  betaProc.rows(1, p).each_col() %= sx1;
  betaProc.row(0) += my - mx * betaProc.rows(1, p);
  return Rcpp::List::create(Rcpp::Named("coeff") = betaProc, Rcpp::Named("bandwidth") = h);
}

// [[Rcpp::export]]
Rcpp::List smqrTrianProc(const arma::mat& X, arma::vec Y, const arma::vec tauSeq, double h = 0.0, const double constTau = 1.345, 
                         const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  const int m = tauSeq.size();
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
  arma::mat betaProc(p + 1, m);
  for (int i = 0; i < m; i++) {
    double tau = tauSeq(i);
    arma::vec beta = huberReg(Z, Y, tau, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
    arma::vec quant = {tau};
    beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
    arma::vec res = Y - Z * beta;
    updateTrian(Z, res, der, gradOld, n, tau, h, n1, h1, h2);
    beta -= gradOld;
    arma::vec betaDiff = -gradOld;
    res -= Z * betaDiff;
    updateTrian(Z, res, der, gradOld, n, tau, h, n1, h1, h2);
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
      updateTrian(Z, res, der, gradOld, n, tau, h, n1, h1, h2);
      gradDiff = gradNew - gradOld;
      ite++;
    }
    betaProc.col(i) = beta;
  }
  betaProc.rows(1, p).each_col() %= sx1;
  betaProc.row(0) += my - mx * betaProc.rows(1, p);
  return Rcpp::List::create(Rcpp::Named("coeff") = betaProc, Rcpp::Named("bandwidth") = h);
}

// Conquer with bootstrap inference
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

// The following code is high-dimensional conquer via an iterative local majorize-minimize algorithm
// [[Rcpp::export]]
arma::vec softThresh(const arma::vec& x, const arma::vec& lambda, const int p) {
  return arma::sign(x) % arma::max(arma::abs(x) - lambda, arma::zeros(p + 1));
}

// [[Rcpp::export]]
arma::vec cmptLambdaLasso(const double lambda, const int p) {
  arma::vec rst = lambda * arma::ones(p + 1);
  rst(0) = 0;
  return rst;
}

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

// Expectile regression (asymmetric l_2 loss) as an initial value for high-dim regression
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

// Smoothed quantile loss with different kernels
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
double lossLogisticHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau, const double h, const double h1) {
  arma::vec res = Y - Z * beta;
  arma::vec temp = tau * res + h * arma::log(1.0 + arma::exp(-h1 * res));
  return arma::mean(temp);
}

// [[Rcpp::export]]
double updateLogisticHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, arma::vec& grad, const double tau, const double n1, const double h, 
                        const double h1) {
  arma::vec res = Y - Z * beta;
  arma::vec der = 1.0 / (1.0 + arma::exp(res * h1)) - tau;
  grad = n1 * Z.t() * der;
  arma::vec temp = tau * res + h * arma::log(1.0 + arma::exp(-h1 * res));
  return arma::mean(temp);
}

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
double lossTrianHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau, const double h, const double h1, const double h2) {
  arma::vec res = Y - Z * beta;
  arma::vec temp = (tau - 0.5) * res;
  for (int i = 0; i < res.size(); i++) {
    double cur = std::abs(res(i));
    temp(i) += cur <= h ? (0.5 * h1 * cur * cur - 0.1666667 * h2 * cur * cur * cur + 0.1666667 * h) : 0.5 * cur;
  }
  return arma::mean(temp);
}

// [[Rcpp::export]]
double updateTrianHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, arma::vec& grad, const double tau, const double n1, const double h, 
                     const double h1, const double h2) {
  arma::vec res = Y - Z * beta;
  arma::vec temp = (tau - 0.5) * res;
  arma::vec der(res.size());
  for (int i = 0; i < res.size(); i++) {
    double cur = res(i);
    if (cur <= -h) {
      der(i) = 1 - tau;
      temp(i) -= 0.5 * cur;
    } else if (cur < 0) {
      der(i) = 0.5 - tau - h1 * cur - 0.5 * h2 * cur * cur;
      temp(i) += 0.5 * h1 * cur * cur + 0.1666667 * h2 * cur * cur * cur + 0.1666667 * h;
    } else if (cur < h) {
      der(i) = 0.5 - tau - h1 * cur + 0.5 * h2 * cur * cur;
      temp(i) += 0.5 * h1 * cur * cur - 0.1666667 * h2 * cur * cur * cur + 0.1666667 * h;
    } else {
      der(i) = -tau;
      temp(i) += 0.5 * cur;
    }
  }
  grad = n1 * Z.t() * der;
  return arma::mean(temp);
}

// LAMM core code for different loss functions, update beta, return phi
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
arma::vec lasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double phi0 = 0.1, 
                const double gamma = 1.2, const double epsilon = 0.01, const int iteMax = 500) {
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
double lammSmqrGauss(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double phi, const double tau, 
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
double lammSmqrLogistic(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double phi, const double tau, 
                        const double gamma, const int p, const double h, const double n1, const double h1) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateLogisticHd(Z, Y, beta, grad, tau, n1, h, h1);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = lossLogisticHd(Z, Y, betaNew, tau, h, h1);
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
double lammSmqrUnif(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double phi, const double tau, 
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
double lammSmqrPara(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double phi, const double tau, 
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
double lammSmqrTrian(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double phi, const double tau, 
                     const double gamma, const int p, const double h, const double n1, const double h1, const double h2) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateTrianHd(Z, Y, beta, grad, tau, n1, h, h1, h2);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = lossTrianHd(Z, Y, betaNew, tau, h, h1, h2);
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
arma::vec smqrLassoGauss(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                         const double h, const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                         const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrGauss(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec smqrLassoLogistic(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                            const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                            const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrLogistic(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec smqrLassoUnif(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                        const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                        const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrUnif(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec smqrLassoPara(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                        const double h, const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                        const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrPara(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h3);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec smqrLassoTrian(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                         const double h, const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                         const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrTrian(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec smqrScadGauss(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                        const double h, const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                        const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrGauss(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
      phi = lammSmqrGauss(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
arma::vec smqrScadLogistic(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                           const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                           const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrLogistic(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1);
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
      phi = lammSmqrLogistic(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1);
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
arma::vec smqrScadUnif(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                       const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                       const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrUnif(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1);
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
      phi = lammSmqrUnif(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1);
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
arma::vec smqrScadPara(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                       const double h, const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                       const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrPara(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h3);
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
      phi = lammSmqrPara(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h3);
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
arma::vec smqrScadTrian(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                        const double h, const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                        const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrTrian(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
      phi = lammSmqrTrian(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
arma::vec smqrMcpGauss(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                        const double h, const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                        const int iteMax = 500, const int iteTight = 3, const double para = 3) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrGauss(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
      phi = lammSmqrGauss(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
arma::vec smqrMcpLogistic(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                          const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                          const int iteMax = 500, const int iteTight = 3, const double para = 3) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrLogistic(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1);
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
      phi = lammSmqrLogistic(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1);
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
arma::vec smqrMcpUnif(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                      const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                      const int iteMax = 500, const int iteTight = 3, const double para = 3) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrUnif(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1);
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
      phi = lammSmqrUnif(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1);
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
arma::vec smqrMcpPara(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                      const double h, const double h1, const double h3, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                      const int iteMax = 500, const int iteTight = 3, const double para = 3) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrPara(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h3);
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
      phi = lammSmqrPara(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h3);
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
arma::vec smqrMcpTrian(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& sx1, const double tau, const int p, const double n1, 
                       const double h, const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                       const int iteMax = 500, const int iteTight = 3, const double para = 3) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSmqrTrian(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
      phi = lammSmqrTrian(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
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
    betaHat = smqrLassoGauss(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  } else if (type == 2) {
    betaHat = smqrScadGauss(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  } else {
    betaHat = smqrMcpGauss(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  }
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::vec conquerHdLogistic(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const int type = 1, const double phi0 = 0.01, 
                            const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat(p + 1);
  if (type == 1) {
    betaHat = smqrLassoLogistic(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  } else if (type == 2) {
    betaHat = smqrScadLogistic(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  } else {
    betaHat = smqrMcpLogistic(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  }
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::vec conquerHdUnif(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const int type = 1, const double phi0 = 0.01, 
                        const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat(p + 1);
  if (type == 1) {
    betaHat = smqrLassoUnif(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  } else if (type == 2) {
    betaHat = smqrScadUnif(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  } else {
    betaHat = smqrMcpUnif(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  }
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::vec conquerHdPara(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const int type = 1, const double phi0 = 0.01, 
                        const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx1, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec betaHat(p + 1);
  if (type == 1) {
    betaHat = smqrLassoPara(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  } else if (type == 2) {
    betaHat = smqrScadPara(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  } else {
    betaHat = smqrMcpPara(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  }
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// [[Rcpp::export]]
arma::vec conquerHdTrian(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const int type = 1, const double phi0 = 0.01, 
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
    betaHat = smqrLassoTrian(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  } else if (type == 2) {
    betaHat = smqrScadTrian(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  } else {
    betaHat = smqrMcpTrian(Z, Y, lambda, sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  }
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return betaHat;
}

// cross-validation, no warm-start, the range of lambda is guided by the simulation-based mathod of Belloni & Chernozhukov (2011), AOS
// [[Rcpp::export]]
double lossQr(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau) {
  arma::vec res = Y - Z * beta;
  double rst = 0.0;
  for (int i = 0; i < res.size(); i++) {
    rst += res(i) >= 0 ? tau * res(i) : (tau - 1) * res(i);
  }
  return rst;
}

// [[Rcpp::export]]
Rcpp::List cvSmqrLassoGauss(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrLassoGauss(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrLassoGauss(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrScadGauss(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrScadGauss(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrScadGauss(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrMcpGauss(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrMcpGauss(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrMcpGauss(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrLassoLogistic(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrLassoLogistic(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrLassoLogistic(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrScadLogistic(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrScadLogistic(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrScadLogistic(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrMcpLogistic(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrMcpLogistic(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrMcpLogistic(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrLassoUnif(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrLassoUnif(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrLassoUnif(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrScadUnif(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrScadUnif(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrScadUnif(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrMcpUnif(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrMcpUnif(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrMcpUnif(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrLassoPara(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                           const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
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
      betaHat = smqrLassoPara(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrLassoPara(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrScadPara(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                          const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500,
                          const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
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
      betaHat = smqrScadPara(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrScadPara(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrMcpPara(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                         const double h, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500,
                         const int iteTight = 3, const double para = 3) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h3 = 1.0 / (h * h * h);
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
      betaHat = smqrMcpPara(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrMcpPara(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, h3, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrLassoTrian(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrLassoTrian(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrLassoTrian(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrScadTrian(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrScadTrian(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrScadTrian(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}

// [[Rcpp::export]]
Rcpp::List cvSmqrMcpTrian(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
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
      betaHat = smqrMcpTrian(trainZ, trainY, lambdaSeq(i), sx1, tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
      mse(i) += arma::accu(lossQr(testZ, testY, betaHat, tau));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  betaHat = smqrMcpTrian(Z, Y, lambdaSeq(cvIdx), sx1, tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat.rows(1, p) %= sx1;
  betaHat(0) += my - arma::as_scalar(mx * betaHat.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("lambda") = lambdaSeq(cvIdx));
}


