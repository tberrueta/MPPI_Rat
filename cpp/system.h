#ifndef SYSTEM_FUNS
#define SYSTEM_FUNS
#include <armadillo>
#include <cmath>
#include "utils.h"

// Differential drive dynamics
arma::vec diff_drive(const arma::vec x, const arma::vec u) {
	arma::vec xdot(3, arma::fill::zeros);
	xdot(0) = u(0) * std::cos(x(2));
	xdot(1) = u(0) * std::sin(x(2));
	xdot(2) = u(1);
	return xdot;
}

// Standard quadratic objective
double quadratic_objective(arma::mat x_traj, arma::mat u_traj, arma::mat xdes, arma::mat Q, arma::mat R) {
	double c = 0.0;
	for (int i = 0; i < x_traj.n_cols; i++) {
		c += arma::as_scalar((x_traj.col(i)-xdes.col(i)).t()*Q*(x_traj.col(i) - xdes.col(i))+u_traj.col(i).t()*R* u_traj.col(i));
	}
	return c;
}

// Objective blending quadratic cost and rattling
double rattling_objective(arma::mat x_traj, arma::mat u_traj, arma::mat xdes, double dt, int w_sz, double ov, double w1, double w2, arma::mat Q, arma::mat R) {
	double c = w1 * quadratic_objective(x_traj, u_traj, xdes, Q, R);
	c += w2*arma::mean(rattling_windows(x_traj,dt,w_sz,ov));
	return c;
}
#endif