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
  return arma::median(arma::abs(x - arma::median(x))) / 0.6744898;
}

// [[Rcpp::export]]
arma::vec huberDer(const arma::vec& x, const int n, const double tau) {
  arma::vec w(n);
  for (int i = 0; i < n; i++) {
    w(i) = std::abs(x(i)) <= tau ? -x(i) : -tau * sgn(x(i));
  }
  return w;
}

// [[Rcpp::export]]
double huberLoss(const arma::vec& x, const int n, const double tau) {
  double loss = 0;
  for (int i = 0; i < n; i++) {
    double cur = x(i);
    loss += std::abs(cur) <= tau ? (cur * cur / 2) : (tau * std::abs(cur) - tau * tau / 2);
  }
  return loss / n;
}

// [[Rcpp::export]]
arma::vec huberReg(const arma::mat& Z, const arma::vec& Y, const int n, const int p, const double tol = 0.0000001, const double constTau = 1.345, 
                   const int iteMax = 5000) {
  double tau = constTau * mad(Y);
  arma::vec gradOld = Z.t() * huberDer(Y, n, tau) / n;
  double lossOld = huberLoss(Y, n, tau);
  arma::vec beta = -gradOld;
  arma::vec betaDiff = -gradOld;
  arma::vec res = Y - Z * beta;
  tau = constTau * mad(res);
  double lossNew = huberLoss(res, n, tau);
  arma::vec gradNew = Z.t() * huberDer(res, n, tau) / n;
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaDiff, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    lossOld = lossNew;
    beta -= alpha * gradNew;
    betaDiff = -alpha * gradNew;
    res += alpha * Z * gradNew;
    tau = constTau * mad(res);
    gradNew = Z.t() * huberDer(res, n, tau) / n;
    lossNew = huberLoss(res, n, tau);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  return beta;
}

// [[Rcpp::export]]
arma::vec sqDerGauss(const arma::vec& u, const double tau) {
  return arma::normcdf(u) - tau;
}

// [[Rcpp::export]]
arma::vec sqDerUnif(const arma::vec& u, const int n, const double tau) {
  arma::vec rst(n);
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur > 1) {
      rst(i) = 1 - tau;
    } else if (cur > -1) {
      rst(i) = (1 + cur) / 2 - tau;
    } else {
      rst(i) = -tau;
    }
  }
  return rst;
}

// [[Rcpp::export]]
arma::vec sqDerPara(const arma::vec& u, const int n, const double tau) {
  arma::vec rst(n);
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur > 1) {
      rst(i) = 1 - tau;
    } else if (cur > -1) {
      rst(i) = (3 * cur - cur * cur * cur + 2) / 4 - tau;
    } else {
      rst(i) = -tau;
    }
  }
  return rst;
}

// [[Rcpp::export]]
arma::vec sqDerTrian(const arma::vec& u, const int n, const double tau) {
  arma::vec rst(n);
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur > 1) {
      rst(i) = 1 - tau;
    } else if (cur > 0) {
      rst(i) = (2 * cur - cur * cur + 1) / 2 - tau;
    } else if (cur > -1) {
      rst(i) = (1 + cur * cur + 2 * cur) / 2 - tau;
    } else {
      rst(i) = -tau;
    }
  }
  return rst;
}

// [[Rcpp::export]]
arma::vec sqDerHoro(const arma::vec& u, const double tau) {
  return arma::normcdf(u) + u % arma::normpdf(u) - tau;
}

// [[Rcpp::export]]
double sqLossGauss(const arma::vec& u, const double tau, const double h) {
  arma::vec temp = h * 0.79788 * arma::exp(-arma::square(u) / (2 * h * h)) + u - 2 * u % arma::normcdf(-u / h);
  return arma::mean(temp / 2 + (tau - 0.5) * u);
}

// [[Rcpp::export]]
double sqLossUnif(const arma::vec& u, const int n, const double tau, const double h) {
  double rst = 0;
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur <= -h) {
      rst += (tau - 1) * cur;
    } else if (cur < h) {
      rst += cur * cur / (4 * h) + h / 4 + (tau - 0.5) * cur;
    } else {
      rst += tau * cur;
    }
  }
  return rst / n;
}

// [[Rcpp::export]]
double sqLossPara(const arma::vec& u, const int n, const double tau, const double h) {
  double rst = 0;
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur <= -h) {
      rst += (tau - 1) * cur;
    } else if (cur < h) {
      rst += 3 * h / 16 + 3 * cur * cur / (8 * h) - cur * cur * cur * cur / (16 * h * h * h) + (tau - 0.5) * cur;
    } else {
      rst += tau * cur;
    }
  }
  return rst / n;
}

// [[Rcpp::export]]
double sqLossTrian(const arma::vec& u, const int n, const double tau, const double h) {
  double rst = 0;
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur <= -h) {
      rst += (tau - 1) * cur;
    } else if (cur < 0) {
      rst += cur * cur * cur / (6 * h * h) + cur * cur / (2 * h) + h / 6 + (tau - 0.5) * cur;
    } else if (cur < h) {
      rst += -cur * cur * cur / (6 * h * h) + cur * cur / (2 * h) + h / 6 + (tau - 0.5) * cur;
    } else {
      rst += tau * cur;
    }
  }
  return rst / n;
}

// [[Rcpp::export]]
double sqLossHoro(const arma::vec& u, const double tau, const double h) {
  return arma::mean(tau * u - u % arma::normcdf(-u / h));
}

// [[Rcpp::export]]
arma::mat standardize(arma::mat X, const int p) {
  for (int i = 0; i < p; i++) {
    X.col(i) = (X.col(i) - arma::mean(X.col(i))) / arma::stddev(X.col(i));
  }
  return X;
}

// [[Rcpp::export]]
Rcpp::List smqrGauss(const arma::mat& X, const arma::vec& Y, const double tau = 0.5, const double constTau = 1.345, const double tol = 0.0000001, 
                     const int iteMax = 5000) {
  int n = X.n_rows;
  int p = X.n_cols;
  double h = std::pow((std::log(n) + p) / n, 0.4);
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, p));
  arma::vec beta = huberReg(Z, Y, n, p, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  double lossOld = sqLossGauss(res, tau, h);
  arma::vec gradOld = Z.t() * sqDerGauss(-res / h, tau) / n;
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res = Y - Z * beta;
  double lossNew = sqLossGauss(res, tau, h);
  arma::vec gradNew = Z.t() * sqDerGauss(-res / h, tau) / n;
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaDiff, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    lossOld = lossNew;
    beta -= alpha * gradNew;
    betaDiff = -alpha * gradNew;
    res += alpha * Z * gradNew;
    gradNew = Z.t() * sqDerGauss(-res / h, tau) / n;
    lossNew = sqLossGauss(res, tau, h);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) /= arma::stddev(X, 0, 0).t();
  beta(0) -= arma::as_scalar(arma::mean(X, 0) * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite);
}

// [[Rcpp::export]]
arma::vec smqrGaussIni(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int p, const double tau = 0.5, 
                       const double tol = 0.0000001, const int iteMax = 5000) {
  int n = X.n_rows;
  double h = std::pow((std::log(n) + p) / n, 0.4);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  arma::vec beta = betaHat;
  arma::vec res = Y - Z * beta;
  double lossOld = sqLossGauss(res, tau, h);
  arma::vec gradOld = Z.t() * sqDerGauss(-res / h, tau) / n;
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res = Y - Z * beta;
  double lossNew = sqLossGauss(res, tau, h);
  arma::vec gradNew = Z.t() * sqDerGauss(-res / h, tau) / n;
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaDiff, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    lossOld = lossNew;
    beta -= alpha * gradNew;
    betaDiff = -alpha * gradNew;
    res += alpha * Z * gradNew;
    gradNew = Z.t() * sqDerGauss(-res / h, tau) / n;
    lossNew = sqLossGauss(res, tau, h);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) /= arma::stddev(X, 0, 0).t();
  beta(0) -= arma::as_scalar(arma::mean(X, 0) * beta.rows(1, p));
  return beta;
}

// [[Rcpp::export]]
Rcpp::List smqrUnif(const arma::mat& X, const arma::vec& Y, const double tau = 0.5, const double constTau = 1.345, const double tol = 0.0000001, 
                    const int iteMax = 5000) {
  int n = X.n_rows;
  int p = X.n_cols;
  double h = std::pow((std::log(n) + p) / n, 0.4);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  arma::vec beta = huberReg(Z, Y, n, p, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  double lossOld = sqLossUnif(res, n, tau, h);
  arma::vec gradOld = Z.t() * sqDerUnif(-res / h, n, tau) / n;
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res = Y - Z * beta;
  double lossNew = sqLossUnif(res, n, tau, h);
  arma::vec gradNew = Z.t() * sqDerUnif(-res / h, n, tau) / n;
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaDiff, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    lossOld = lossNew;
    beta -= alpha * gradNew;
    betaDiff = -alpha * gradNew;
    res += alpha * Z * gradNew;
    gradNew = Z.t() * sqDerUnif(-res / h, n, tau) / n;
    lossNew = sqLossUnif(res, n, tau, h);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) /= arma::stddev(X, 0, 0).t();
  beta(0) -= arma::as_scalar(arma::mean(X, 0) * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite);
}

// [[Rcpp::export]]
arma::vec smqrUnifIni(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int p, const double tau = 0.5, 
                      const double tol = 0.0000001, const int iteMax = 5000) {
  int n = X.n_rows;
  double h = std::pow((std::log(n) + p) / n, 0.4);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  arma::vec beta = betaHat;
  arma::vec res = Y - Z * beta;
  double lossOld = sqLossUnif(res, n, tau, h);
  arma::vec gradOld = Z.t() * sqDerUnif(-res / h, n, tau) / n;
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res = Y - Z * beta;
  double lossNew = sqLossUnif(res, n, tau, h);
  arma::vec gradNew = Z.t() * sqDerUnif(-res / h, n, tau) / n;
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaDiff, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    lossOld = lossNew;
    beta -= alpha * gradNew;
    betaDiff = -alpha * gradNew;
    res += alpha * Z * gradNew;
    gradNew = Z.t() * sqDerUnif(-res / h, n, tau) / n;
    lossNew = sqLossUnif(res, n, tau, h);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) /= arma::stddev(X, 0, 0).t();
  beta(0) -= arma::as_scalar(arma::mean(X, 0) * beta.rows(1, p));
  return beta;
}
  
// [[Rcpp::export]]
Rcpp::List smqrPara(const arma::mat& X, const arma::vec& Y, const double tau = 0.5, const double constTau = 1.345, const double tol = 0.0000001, 
                    const int iteMax = 5000) {
  int n = X.n_rows;
  int p = X.n_cols;
  double h = std::pow((std::log(n) + p) / n, 0.4);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  arma::vec beta = huberReg(Z, Y, n, p, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  double lossOld = sqLossPara(res, n, tau, h);
  arma::vec gradOld = Z.t() * sqDerPara(-res / h, n, tau) / n;
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res = Y - Z * beta;
  double lossNew = sqLossPara(res, n, tau, h);
  arma::vec gradNew = Z.t() * sqDerPara(-res / h, n, tau) / n;
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaDiff, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    lossOld = lossNew;
    beta -= alpha * gradNew;
    betaDiff = -alpha * gradNew;
    res += alpha * Z * gradNew;
    gradNew = Z.t() * sqDerPara(-res / h, n, tau) / n;
    lossNew = sqLossPara(res, n, tau, h);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) /= arma::stddev(X, 0, 0).t();
  beta(0) -= arma::as_scalar(arma::mean(X, 0) * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite);
}

// [[Rcpp::export]]
arma::vec smqrParaIni(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int p, const double tau = 0.5, 
                      const double tol = 0.0000001, const int iteMax = 5000) {
  int n = X.n_rows;
  double h = std::pow((std::log(n) + p) / n, 0.4);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  arma::vec beta = betaHat;
  arma::vec res = Y - Z * beta;
  double lossOld = sqLossPara(res, n, tau, h);
  arma::vec gradOld = Z.t() * sqDerPara(-res / h, n, tau) / n;
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res = Y - Z * beta;
  double lossNew = sqLossPara(res, n, tau, h);
  arma::vec gradNew = Z.t() * sqDerPara(-res / h, n, tau) / n;
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaDiff, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    lossOld = lossNew;
    beta -= alpha * gradNew;
    betaDiff = -alpha * gradNew;
    res += alpha * Z * gradNew;
    gradNew = Z.t() * sqDerPara(-res / h, n, tau) / n;
    lossNew = sqLossPara(res, n, tau, h);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) /= arma::stddev(X, 0, 0).t();
  beta(0) -= arma::as_scalar(arma::mean(X, 0) * beta.rows(1, p));
  return beta;
}

// [[Rcpp::export]]
Rcpp::List smqrTrian(const arma::mat& X, const arma::vec& Y, const double tau = 0.5, const double constTau = 1.345, const double tol = 0.0000001, 
                     const int iteMax = 5000) {
  int n = X.n_rows;
  int p = X.n_cols;
  double h = std::pow((std::log(n) + p) / n, 0.4);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  arma::vec beta = huberReg(Z, Y, n, p, tol, constTau, iteMax);
  arma::vec quant = {tau};
  beta(0) = arma::as_scalar(arma::quantile(Y - Z.cols(1, p) * beta.rows(1, p), quant));
  arma::vec res = Y - Z * beta;
  double lossOld = sqLossTrian(res, n, tau, h);
  arma::vec gradOld = Z.t() * sqDerTrian(-res / h, n, tau) / n;
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res = Y - Z * beta;
  double lossNew = sqLossTrian(res, n, tau, h);
  arma::vec gradNew = Z.t() * sqDerTrian(-res / h, n, tau) / n;
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaDiff, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    lossOld = lossNew;
    beta -= alpha * gradNew;
    betaDiff = -alpha * gradNew;
    res += alpha * Z * gradNew;
    gradNew = Z.t() * sqDerTrian(-res / h, n, tau) / n;
    lossNew = sqLossTrian(res, n, tau, h);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) /= arma::stddev(X, 0, 0).t();
  beta(0) -= arma::as_scalar(arma::mean(X, 0) * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("ite") = ite);
}

// [[Rcpp::export]]
arma::vec smqrTrianIni(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int p, const double tau = 0.5, 
                       const double tol = 0.0000001, const int iteMax = 5000) {
  int n = X.n_rows;
  double h = std::pow((std::log(n) + p) / n, 0.4);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  arma::vec beta = betaHat;
  arma::vec res = Y - Z * beta;
  double lossOld = sqLossTrian(res, n, tau, h);
  arma::vec gradOld = Z.t() * sqDerTrian(-res / h, n, tau) / n;
  beta -= gradOld;
  arma::vec betaDiff = -gradOld;
  res = Y - Z * beta;
  double lossNew = sqLossTrian(res, n, tau, h);
  arma::vec gradNew = Z.t() * sqDerTrian(-res / h, n, tau) / n;
  arma::vec gradDiff = gradNew - gradOld;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaDiff, "inf") > tol && ite <= iteMax) {
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 100.0);
    }
    gradOld = gradNew;
    lossOld = lossNew;
    beta -= alpha * gradNew;
    betaDiff = -alpha * gradNew;
    res += alpha * Z * gradNew;
    gradNew = Z.t() * sqDerTrian(-res / h, n, tau) / n;
    lossNew = sqLossTrian(res, n, tau, h);
    gradDiff = gradNew - gradOld;
    ite++;
  }
  beta.rows(1, p) /= arma::stddev(X, 0, 0).t();
  beta(0) -= arma::as_scalar(arma::mean(X, 0) * beta.rows(1, p));
  return beta;
}

// [[Rcpp::export]]
arma::mat smqrGaussInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, const double tau = 0.5, 
                       const int B = 1000, const double tol = 0.0000001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrGaussIni(mbX, mbY, betaHat, p, tau, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrUnifInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, const double tau = 0.5, 
                      const int B = 1000, const double tol = 0.0000001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrUnifIni(mbX, mbY, betaHat, p, tau, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrParaInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, const double tau = 0.5, 
                      const int B = 1000, const double tol = 0.0000001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrParaIni(mbX, mbY, betaHat, p, tau, tol, iteMax);
  }
  return rst;
}

// [[Rcpp::export]]
arma::mat smqrTrianInf(const arma::mat& X, const arma::vec& Y, const arma::vec& betaHat, const int n, const int p, const double tau = 0.5, 
                       const int B = 1000, const double tol = 0.0000001, const int iteMax = 5000) {
  arma::mat rst(p + 1, B);
  for (int b = 0; b < B; b++) {
    arma::uvec idx = arma::find(arma::randi(n, arma::distr_param(0, 1)) == 1);
    arma::mat mbX = X.rows(idx);
    arma::mat mbY = Y.rows(idx);
    rst.col(b) = smqrTrianIni(mbX, mbY, betaHat, p, tau, tol, iteMax);
  }
  return rst;
}
