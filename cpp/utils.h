#ifndef RAT_UTILS
#define RAT_UTILS
#include <armadillo>
#include <cassert>
#include <tuple>


// RK4 integrator
arma::vec integrate(arma::vec(*f)(arma::vec, arma::vec), const arma::vec x, const arma::vec u, double dt) {
	arma::vec k1, k2, k3, k4;
	k1 = f(x, u) * dt;
	k2 = f(x + k1 / 2.0, u) * dt;
	k3 = f(x + k2 / 2.0, u) * dt;
	k4 = f(x + k3, u) * dt;
	return x + (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
}

// Forward simulation helper
arma::mat simulate(arma::vec(*f)(arma::vec, arma::vec), const arma::vec x0, const arma::mat u_traj, double dt, int N) {
	arma::mat x_traj(x0.size(), N, arma::fill::zeros);
	arma::vec x = x0;
	for (int i = 0; i < N; i++) {
		x_traj.col(i) = integrate(*f, x, u_traj.col(i), dt);
		x = x_traj.col(i);
	}
	return x_traj;
}

// Calculates diffusion velocities for rattling
arma::mat diffusion_vel(arma::mat x_traj, double dt) {
	arma::vec times = arma::sqrt(dt*arma::linspace(1., x_traj.n_cols + 1., x_traj.n_cols));
	arma::mat out(x_traj.n_rows, x_traj.n_cols,arma::fill::zeros);
	for (int i = 0; i < x_traj.n_cols; i++) {
		out.col(i) = (x_traj.col(i)-x_traj.col(0))/times(i);
	}
	return out;
}

// Calculating rattling
double rattling(arma::mat x_traj, double dt) {
	arma::mat vels = diffusion_vel(x_traj, dt);
	arma::mat C = arma::cov(vels.t());
	return 0.5 * std::log(arma::det(C));
}

// Ractangular windowing of data
std::vector<std::tuple<int, int>> window_inds(arma::mat data, int w_sz, double ov) {
	assert(w_sz < data.n_cols);
	int ind1 = 0;
	int ind2 = w_sz - 1;
	std::vector<std::tuple<int, int>> ind_vec;
	int ov_ind_diff = (int)std::ceil(std::abs(ov * w_sz));
	if (ov_ind_diff == w_sz) {
		ov_ind_diff--;
	}
	while (ind2 < data.n_cols) {
		ind_vec.push_back(std::make_tuple(ind1, ind2));
		ind1 += w_sz - ov_ind_diff;
		ind2 += w_sz - ov_ind_diff;
	}
	return ind_vec;
}

// Calculating rattling over successive windows
arma::vec rattling_windows(arma::mat x_traj, double dt, int w_sz, double ov) {
	std::vector<std::tuple<int, int>> ind_list = window_inds(x_traj, w_sz, ov);
	arma::vec rats(ind_list.size(), arma::fill::zeros);
	for (int i = 0; i < ind_list.size(); i++) {
		rats(i) = rattling(x_traj.cols(std::get<0>(ind_list[i]), std::get<1>(ind_list[i])), dt);
	}
	return rats;
}

// Applies moving average to a single vector
arma::vec moving_average_vec(arma::vec x, int N) {
	arma::vec ones(N, arma::fill::ones);
	arma::vec out = arma::conv(x, ones / ((float)N), "same");
	return out;
}

// Applies moving average to full matrix
arma::mat moving_average(arma::mat xmat, int N) {
	arma::mat out(xmat.n_rows, xmat.n_cols, arma::fill::zeros);
	for (int i = 0; i < xmat.n_rows; i++) {
		out.row(i) = moving_average_vec(xmat.row(i), N);
	}
	return out;
}

// Generates random samples from multivariate gaussian
arma::mat gaussian_samples(arma::vec mu, arma::mat cov, int N) {
	arma::mat mu_mat(mu.size(), N, arma::fill::zeros);
	for (int i = 0; i < N; i++) mu_mat.col(i) = mu;
	arma::mat res(mu.size(),N,arma::fill::randn);
	arma::mat L = arma::chol(cov, "lower");
	return mu_mat + L * res;
}

#endif