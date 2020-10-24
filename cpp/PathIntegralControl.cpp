// PathIntegralControl.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <ctime>
#include <iostream>
#include <armadillo>
#include "system.h"
#include "utils.h"
#include "MPPI.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
typedef std::vector<double> stdvec;

// Global variable declarations
int N = 40; // planning horizon
int K = 100; // number of samples to consider when synthesizing control
int w_sz = 15; // window over which to estimate rattling
int sim_len = 500; // length of simulation
double lambda = 1.0; // temperature hyperparameter of MPPI
double track_weight = 1.0;
double rattling_weight = 1.0; // positive for least-rattling, negative for most-rattling
double ov = 1.0; // overlap of windows for rattling
double dt = 0.05; // timestep
double bound = 5; // environment size
arma::vec x0(3, arma::fill::zeros);
arma::mat xdes(3, N, arma::fill::zeros);
arma::mat Q = 1.0*arma::eye(3, 3);
arma::mat R = 0.01*arma::eye(2, 2);
arma::vec noise_mu(2, arma::fill::zeros); // mean of initial samples
arma::mat noise_sigma = .2*arma::eye(2, 2); // covariance of samples

// Simple helpers and functions to pass
inline stdvec to_vec(arma::vec x) { return arma::conv_to<stdvec>::from(x); }
inline arma::vec dyn_fun(arma::vec x, arma::vec u) { return diff_drive(x, u); }
//inline double obj_fun(arma::mat x_vec, arma::mat u_vec) { return quadratic_objective(x_vec, u_vec, xdes, Q, R); }
inline double obj_fun(arma::mat x_vec, arma::mat u_vec) { return rattling_objective(x_vec, u_vec, xdes, dt, w_sz, ov, track_weight, rattling_weight, Q, R); }


int main()
{   
    /////////////////////
    // Initializations //
    /////////////////////
    arma::arma_rng::set_seed_random();
    Q(2, 2) = 0.001; // we care less about orientation

    // Random inital conditions and goals
    /*arma::vec rand_des = bound * (2. * (arma::randu(2) - 0.5));
    xdes.row(0).fill(rand_des(0));
    xdes.row(1).fill(rand_des(1));
    x0 = bound * (2. * (arma::randu(3) - 0.5));*/
    
    // Not random
    xdes.row(0).fill(-1);
    xdes.row(1).fill(4);
    x0(0) = 4;
    x0(1) = -2;

    arma::vec cost_traj(sim_len, arma::fill::zeros);
    arma::mat x_traj(x0.size(), sim_len, arma::fill::zeros);
    arma::mat u_traj(noise_mu.size(), sim_len, arma::fill::zeros);
    arma::mat u_star(noise_mu.size(), sim_len, arma::fill::zeros);
    MPPI controller(*dyn_fun, *obj_fun, x0, K, N, dt, lambda, noise_mu, noise_sigma);

    ///////////////////////
    // Main control loop //
    ///////////////////////
    double sTime;
    sTime = clock();
    for (int i = 0; i < sim_len; i++) {
        u_star = controller.step(x0);
        x_traj.col(i) = x0;
        u_traj.col(i) = u_star.col(0);
        cost_traj(i) = arma::mean(controller.cost_total);
        x0 = integrate(*dyn_fun, x0, u_star.col(0), dt);
    }
    std::cout << "Time elapsed: " << (clock() - sTime) / CLOCKS_PER_SEC << "s" << "\n";

    ///////////
    // Plots //
    ///////////
    x_traj = x_traj.t();
    u_traj = u_traj.t();
    arma::vec tvec = arma::linspace(0, sim_len * dt, sim_len);
    arma::vec tvec_rats = arma::linspace(0, sim_len * dt, sim_len-w_sz+1);
    arma::vec rats = rattling_windows(x_traj.t(), dt, w_sz, ov);

    // Control signals
    plt::figure();
    plt::plot(to_vec(tvec), to_vec(u_traj.col(0)));
    plt::plot(to_vec(tvec), to_vec(u_traj.col(1)));
    plt::ylabel("Controls");
    plt::xlabel("Time (s)");
    plt::xlim(0.0, sim_len * dt);

    // Cost over time 
    plt::figure();
    plt::plot(to_vec(tvec), to_vec(cost_traj));
    plt::ylabel("Average Cost");
    plt::xlabel("Time (s)");
    plt::xlim(0.0, sim_len * dt);

    // Rattling over time
    plt::figure();
    plt::plot(to_vec(tvec_rats), to_vec(rats));
    plt::ylabel("Rattling");
    plt::xlabel("Time (s)");
    plt::xlim(0.0, sim_len * dt);

    // Parametric plot
    plt::figure();
    plt::plot(to_vec(x_traj.col(0)), to_vec(x_traj.col(1)));
    plt::title("Parametric Trajectory");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::xlim(-bound, bound);
    plt::ylim(-bound, bound);
    plt::grid(true);

    // State tracking 
    plt::figure();
    std::vector<double> tdum = { tvec(0),tvec(sim_len - 1) };
    std::vector<double> xdesdum1 = { xdes(0,0),xdes(0,0) };
    std::vector<double> xdesdum2 = { xdes(1,0),xdes(1,0) };
    plt::plot(to_vec(tvec), to_vec(x_traj.col(0)));
    plt::plot(to_vec(tvec), to_vec(x_traj.col(1)));
    plt::plot(tdum, xdesdum1, { {"color", "black"}, {"linestyle", "--"} });
    plt::plot(tdum, xdesdum2, { {"color", "black"}, {"linestyle", "--"} });
    plt::ylabel("States");
    plt::xlabel("Time (s)");
    plt::xlim(0.0, 10.0);
    plt::show();
}
