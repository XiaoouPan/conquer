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
void updateHuber(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const int n, const double tau, const double n1) {
  for (int i = 0; i < n; i++) {
    double cur = res(i);
    if (std::abs(cur) <= tau) {
      der(i) = -cur;
    } else {
      der(i) = -tau * sgn(cur);
    }
  }
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
arma::vec huberReg(const arma::mat& Z, const arma::vec& Y, arma::vec& der, arma::vec& gradOld, arma::vec& gradNew, const int n, const int p, 
                   const double n1, const double tol = 0.0001, const double constTau = 1.345, const int iteMax = 5000) {
  double tau = constTau * mad(Y);
  updateHuber(Z, Y, der, gradOld, n, tau, n1);
  arma::vec beta = -gradOld, betaDiff = -gradOld;
  arma::vec res = Y - Z * beta;
  tau = constTau * mad(res);
  updateHuber(Z, res, der, gradNew, n, tau, n1);
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
    tau = constTau * mad(res);
    updateHuber(Z, res, der, gradNew, n, tau, n1);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  return beta;
}

// [[Rcpp::export]]
arma::mat standardize(arma::mat X, const arma::rowvec& mx, const arma::vec& sx, const int p) {
  for (int i = 0; i < p; i++) {
    X.col(i) = (X.col(i) - mx(i)) / sx(i);
  }
  return X;
}

// [[Rcpp::export]]
void updateGauss(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const double tau, const double n1, const double h1) {
  der = arma::normcdf(-res * h1) - tau;
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
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
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
  beta.rows(1, p) /= sx;
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
  arma::vec beta = huberReg(Z, Y, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
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
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
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
  beta.rows(1, p) /= sx;
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
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
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
  beta.rows(1, p) /= sx;
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
  arma::vec beta = huberReg(Z, Y, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
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
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
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
  beta.rows(1, p) /= sx;
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
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
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
  beta.rows(1, p) /= sx;
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
  arma::vec beta = huberReg(Z, Y, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
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
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
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
  beta.rows(1, p) /= sx;
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
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
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
  beta.rows(1, p) /= sx;
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
  arma::vec beta = huberReg(Z, Y, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
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
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
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
  beta.rows(1, p) /= sx;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return beta;
}

// [[Rcpp::export]]
arma::mat smqrGaussInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, const double tau = 0.5, 
                       const int B = 1000, const double tol = 0.0001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  double h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrGaussIni(mbX, mbY, betaHat, p, tau, h, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrUnifInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, const double tau = 0.5, 
                      const int B = 1000, const double tol = 0.0001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  double h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrUnifIni(mbX, mbY, betaHat, p, tau, h, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrParaInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, const double tau = 0.5, 
                      const int B = 1000, const double tol = 0.0001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  double h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrParaIni(mbX, mbY, betaHat, p, tau, h, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrTrianInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, const double tau = 0.5, 
                       const int B = 1000, const double tol = 0.0001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  double h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrTrianIni(mbX, mbY, betaHat, p, tau, h, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
double sqLossHoro(const arma::vec& u, const double tau, const double h) {
  return arma::mean(tau * u - u % arma::normcdf(-u / h));
}

// [[Rcpp::export]]
arma::vec sqDerHoro(const arma::vec& u, const double tau) {
  return arma::normcdf(u) + u % arma::normpdf(u) - tau;
}

// [[Rcpp::export]]
Rcpp::List smqrHoro(const arma::mat& Z, const arma::vec& Y, const double tau = 0.5, const double tol = 0.0001, const int iteMax = 5000) {
  int n = Z.n_rows;
  int p = Z.n_cols - 1;
  double h = std::pow((std::log(n) + p) / n, 2.0 / 5);
  double alpha = 1.0;
  arma::vec betaOld = arma::mvnrnd(arma::zeros(p + 1), arma::eye(p + 1, p + 1), 1);
  arma::vec resOld = Y - Z * betaOld;
  double lossOld = sqLossHoro(resOld, tau, h);
  arma::vec grad = Z.t() * sqDerHoro(-resOld / h, tau) / n;
  arma::vec betaNew = betaOld - alpha * grad;
  arma::vec resNew = Y - Z * betaNew;
  double lossNew = sqLossHoro(resNew, tau, h);
  int ite = 1;
  while (arma::norm(grad, "inf") > tol && ite <= iteMax) {
    resOld = resNew;
    grad = Z.t() * sqDerHoro(-resOld / h, tau) / n;
    betaOld = betaNew;
    lossOld = lossNew;
    betaNew = betaOld - alpha * grad;
    resNew = resOld + alpha * Z * grad;
    lossNew = sqLossHoro(resNew, tau, h);
    if (lossNew > lossOld - alpha * arma::as_scalar(grad.t() * grad) / 2) {
      alpha *= 0.8;
      betaNew = betaOld - alpha * grad;
      resNew = resOld + alpha * Z * grad;
      lossNew = sqLossHoro(resNew, tau, h);
    }
    ite++;
  }
  return Rcpp::List::create(Rcpp::Named("coeff") = betaNew, Rcpp::Named("ite") = ite);
}

// [[Rcpp::export]]
arma::mat hoHessGauss(const arma::mat& X, const arma::vec& resH) {
  arma::vec diagHes = 0.5 * (3 - resH % resH) % arma::normpdf(resH);
  return X.t() * arma::diagmat(diagHes) * X;
}

// [[Rcpp::export]]
arma::vec hoGradGauss(const arma::mat& X, const arma::vec& resH, const double tau) {
  return X.t() * (0.5 * resH % arma::normpdf(resH) + arma::normcdf(resH) + tau - 1);
}

// [[Rcpp::export]]
Rcpp::List osSmqrGauss(const arma::mat& X, arma::vec Y, const double tau = 0.5, double h = 0.0, const double constTau = 1.345, 
                       const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  if (h <= 0.05) {
    h = std::max(std::pow((std::log(n) + p) / n, 0.4), 0.05);
  }
  const double n1 = 1.0 / n;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec der(n);
  arma::vec gradOld(p + 1), gradNew(p + 1);
  arma::vec beta = huberReg(Z, Y, der, gradOld, gradNew, n, p, n1, tol, constTau, iteMax);
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
  beta.rows(1, p) /= sx;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  Y += my;
  Z = arma::join_rows(arma::ones(n), X);
  res = Y - Z * beta;
  h = std::max(std::pow((std::log(n) + p) / n, 2.0 / 9), 0.05);
  arma::mat hoHess = hoHessGauss(Z, res / h);
  arma::vec hoGrad = hoGradGauss(Z, res / h, tau);
  beta += arma::solve(hoHess, hoGrad * h);
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite, Rcpp::Named("residual") = res, Rcpp::Named("bandwidth") = h);
}
