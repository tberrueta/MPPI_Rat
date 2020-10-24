#ifndef MPPIcontroller
#define MPPIcontroller
#include <armadillo>
#include <cmath>
#include "utils.h"

class MPPI {
	public:
		////////////////
		// Parameters //
		////////////////
		int N; // number of timesteps to take
		int K; // number of random samples to take
		double dt; // timestep resolution
		double lambda; // temperature parameter
		arma::vec x0; // initial condition (gets updated each step)
		arma::vec cost_total; // vector containing costs from all samples
		arma::mat u_traj; // current control solution
		std::vector<arma::mat> noise; // list of noise matrices to add to control for each sample
		arma::vec noise_mu; // noise vector mean
		arma::mat noise_sigma; // noise vector covariance
		arma::vec(*dynamics)(arma::vec, arma::vec); // system dynamics passed via pointer
		double(*cost_fun)(arma::mat, arma::mat); // objective function passed via pointer

		/////////////////////////
		// Function prototypes //
		/////////////////////////
		MPPI(arma::vec(*)(arma::vec, arma::vec), double(*)(arma::mat, arma::mat), arma::vec, int, int, double, double, arma::vec, arma::mat);
		void resample();
		void compute_sample_costs();
		void compute_control();
		arma::mat step(arma::vec);
};

// Constructor
MPPI::MPPI(arma::vec(*_dynamics)(arma::vec, arma::vec), double(*_cost_fun)(arma::mat, arma::mat), arma::vec _x0, int _K, int _N, double _dt, double _lambda, arma::vec _noise_mu, arma::mat _noise_sigma) {
	N = _N; K = _K; dt = _dt; lambda = _lambda; x0 = _x0; u_traj = arma::zeros(_noise_mu.size(), N); 
	noise_mu = _noise_mu; noise_sigma = _noise_sigma; dynamics = _dynamics; cost_fun = _cost_fun;
	cost_total = arma::zeros<arma::vec>(K);
}

// Generates noise samples for our control synthesis from multivariate gaussian
void MPPI::resample() {
	noise.clear();
	for (int i = 0; i < K; i++) noise.push_back(gaussian_samples(noise_mu, noise_sigma, N));	
}

// Computes the cost associated with each sampled control trajectory
void MPPI::compute_sample_costs() {
	arma::mat perturbed_u(noise_mu.size(), N, arma::fill::zeros);
	arma::mat x_traj(x0.size(), N, arma::fill::zeros);
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < N; j++) {
			perturbed_u.col(j) = u_traj.col(j) + noise[i].col(j); // the mean of our samples is shifted 
		}														  // to our previous control solution
		x_traj = simulate(*dynamics, x0, perturbed_u, dt, N);
		cost_total(i) = cost_fun(x_traj, perturbed_u);
	}
}

// Controls are computed as an exponentially weigthed expected value over control distributions
void MPPI::compute_control() {
	double beta = cost_total.min();
	arma::vec exp_form_cost = arma::exp(-(1.0 / lambda) * (cost_total - beta));
	double eta = arma::sum(exp_form_cost);
	arma::vec omega = exp_form_cost / eta;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < K; j++) {
			u_traj.col(i) += omega(j) * noise[j].col(i);
		}
	}
}

// Generates a control schedule from the initial condition passed in
arma::mat MPPI::step(arma::vec x_new) {
	cost_total.fill(0);
	x0 = x_new;
	MPPI::resample();
	MPPI::compute_sample_costs();
	MPPI::compute_control();
	arma::mat out_mat = u_traj;
	u_traj.col(N - 1) = arma::zeros<arma::vec>(u_traj.n_rows);
	for (int i = 0; i < N - 1; i++) u_traj.col(i) = out_mat.col(i + 1);
	return out_mat;
}

#endif