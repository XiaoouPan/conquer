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

// [[Rcpp::export]]
arma::vec cmptLambdaLasso(const double lambda, const int p) {
  arma::vec rst = lambda * arma::ones(p + 1);
  rst(0) = 0;
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

// conquer loss with different kernels
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

#endif