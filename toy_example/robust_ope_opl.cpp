#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
List estimate_abstract_model_kmeans(
    const arma::mat& s,
    const arma::mat& s_next,
    const arma::ivec& action,
    const arma::vec& reward,
    const arma::vec& pi_prob,
    const arma::mat& centroids,
    const arma::ivec& clusters) {

  int n = s.n_rows;
  int K = centroids.n_rows;

  // containers for action-specific counts and rewards
  arma::cube count_sa(K, 2, K, arma::fill::zeros);
  arma::mat reward_sum_sa(K, 2, arma::fill::zeros);
  arma::mat count_sa_total(K, 2, arma::fill::zeros);

  // accumulate samples per abstract state for computing pi abstraction
  arma::vec sum_pi(K, arma::fill::zeros);
  arma::vec count_i(K, arma::fill::zeros);

  // first pass: count transitions per action, sum rewards, accumulate pi_prob per abstract state
  for (int t = 0; t < n; ++t) {
    int i = clusters[t] - 1;
    double pi_i = pi_prob[t];
    sum_pi[i] += pi_i;
    count_i[i] += 1.0;

    // find abstract of next state
    double best_dist = arma::datum::inf;
    int j_best = 0;
    for (int j = 0; j < K; ++j) {
      double dist = arma::accu(arma::square(s_next.row(t) - centroids.row(j)));
      if (dist < best_dist) {
        best_dist = dist;
        j_best = j;
      }
    }

    int a = action[t];
    // count and reward per (i,a)->j
    count_sa(i, a, j_best) += 1.0;
    reward_sum_sa(i, a) += reward[t];
    count_sa_total(i, a) += 1.0;
  }

  // compute action-abstracted policy pi_abs per abstract state
  arma::vec pi_abs(K, arma::fill::zeros);
  for (int i = 0; i < K; ++i) {
    if (count_i[i] > 0)
      pi_abs[i] = sum_pi[i] / count_i[i];
    else
      pi_abs[i] = 0.5; // default if no data
  }

  // estimate P_sa and R_sa
  arma::cube P_sa(K, 2, K, arma::fill::zeros);
  arma::mat R_sa(K, 2, arma::fill::zeros);
  for (int i = 0; i < K; ++i) {
    for (int a = 0; a < 2; ++a) {
      if (count_sa_total(i, a) > 0) {
        R_sa(i, a) = reward_sum_sa(i, a) / count_sa_total(i, a);
        for (int j = 0; j < K; ++j) {
          P_sa(i, a, j) = count_sa(i, a, j) / count_sa_total(i, a);
        }
      } else {
        R_sa(i, a) = arma::mean(reward);
        for (int j = 0; j < K; ++j)
          P_sa(i, a, j) = 1.0 / K;
      }
    }
  }

  // combine action-specific models with target policy
  /* arma::mat P(K, K, arma::fill::zeros);
  arma::vec R_vec(K, arma::fill::zeros);
  for (int i = 0; i < K; ++i) {
    double p1 = pi_abs[i];
    for (int j = 0; j < K; ++j) {
      P(i, j) = (1.0 - p1) * P_sa(i, 0, j) + p1 * P_sa(i, 1, j);
    }
    R_vec[i] = (1.0 - p1) * R_sa(i, 0) + p1 * R_sa(i, 1);
  } */

  return List::create(
    Named("P") = P_sa,
    Named("R") = R_sa,
    Named("policy") = pi_abs
  );
}






// Compute candidate lambdas by brute-force pairwise intersections
// [[Rcpp::export]]
vec computeLambdaCandidates(const vec& V, const mat& D, double p) {
    int K = V.n_elem;
    std::set<double> lambdas;
    lambdas.insert(0.0);
    // Precompute D^p
    mat Dp = pow(D, p);

    // Helper lambda to compute intersection
    auto intersect = [&](const std::pair<double,double>& L1,
                        const std::pair<double,double>& L2) {
        // L1: (m1, b1), L2: (m2, b2)
        return (L2.second - L1.second) / (L1.first - L2.first);
    };

    // For each source state j, build its convex hull of lines
    for (int j = 0; j < K; ++j) {
        // Collect lines: slope m = -Dp(j,k), intercept b = -V[k]
        std::map<double,double> slope_map;
        slope_map.clear();
        for (int k = 0; k < K; ++k) {
            double m = -Dp(j, k);
            double b = -V[k];
            auto it = slope_map.find(m);
            if (it == slope_map.end() || b > it->second) {
                slope_map[m] = b;
            }
        }
        // Static hull storage: vector of (m,b)
        std::vector<std::pair<double,double>> hull;
        hull.reserve(slope_map.size());
        // Insert in ascending slope order
        for (auto & kv : slope_map) {
            double m = kv.first;
            double b = kv.second;
            // Maintain upper hull
            while (hull.size() >= 2) {
                auto &L1 = hull[hull.size()-2];
                auto &L2 = hull[hull.size()-1];
                double x12 = intersect(L1, L2);
                double x2c = intersect(L2, {m,b});
                if (x12 >= x2c) {
                    hull.pop_back();
                } else {
                    break;
                }
            }
            hull.emplace_back(m, b);
        }
        // Extract intersection points
        for (size_t t = 0; t+1 < hull.size(); ++t) {
            double lam = intersect(hull[t], hull[t+1]);
            if (lam > 0) lambdas.insert(lam);
        }
    }
    // Output sorted lambdas
    vec out(lambdas.size());
    int idx = 0;
    for (double v : lambdas) {
        out[idx++] = v;
    }
    return out;
}






// [[Rcpp::export]]
List robust_ope_wass(const arma::cube& P_hat,
                               const arma::mat& R_hat,
                               const vec& pi_hat,
                               const vec& delta,
                               const mat& D,
                               double gamma,
                               double p,
                               double tol,
                               int max_iter) {
    int K = R_hat.n_rows;
    // precompute D^p
    mat V_his(K, max_iter);
    mat Dp = pow(D, p);

    mat Q_robust(K, 2);

    double max_norm = 0;

    // solve the non-robust baseline
    arma::mat P_agg(K, K, arma::fill::zeros);
    arma::vec R_agg(K, arma::fill::zeros);
    for (int i = 0; i < K; ++i) {
      double p1 = pi_hat[i];
      for (int j = 0; j < K; ++j) {
        P_agg(i, j) = (1.0 - p1) * P_hat(i, 0, j) + p1 * P_hat(i, 1, j);
      }
      R_agg[i] = (1.0 - p1) * R_hat(i, 0) + p1 * R_hat(i, 1);
    }    
    vec V_original = solve(eye<mat>(K, K) - gamma * P_agg, R_agg);
    vec V = V_original;

    for (int it = 1; it <= max_iter; ++it) {
        Rcpp::checkUserInterrupt();  // allow ESC interruption

        vec V_old = V;
        vec lambdas = computeLambdaCandidates(V_old, D, p);
        int M = lambdas.n_elem;


        // preallocate the matrix of size M x K
        mat Max_Matrix(M, K);
        // negated V_old as row vector, to broadcast
        rowvec Vneg = -V_old.t();

        // parallel fill each row of Max_Matrix
        #pragma omp parallel for schedule(static)
        for (size_t m = 0; m < M; ++m)
        {
            double lam = lambdas[m];

            for (int j = 0; j < K; ++j) {
                double mval = -datum::inf;
                for (int k = 0; k < K; ++k) {
                    double v = -V_old[k] - lam * Dp(j,k);
                    if (v > mval) mval = v;
                }
                Max_Matrix(m,j) = mval;
            }
        
        }

        // now compute the new V
        vec V_new(K);

        arma::mat P0(K,K);
        arma::mat P1(K,K);
        arma::vec R0 = R_hat.col(0);
        arma::vec R1 = R_hat.col(1);
        for (int s = 0; s < K; s++) {
        
          P0.col(s) = P_hat.slice(s).col(0);
          P1.col(s) = P_hat.slice(s).col(1); 
          
        }



        #pragma omp parallel for schedule(static)
        for (int i = 0; i < K; ++i) {
            double di_p = std::pow(delta[i], p);
            // compute y[m] = sum_j P_hat(i,j) * Max_Matrix(m,j)
            vec y0 = Max_Matrix * P0.row(i).t();  // M×K * K×1 → M×1
            vec y1 = Max_Matrix * P1.row(i).t();
            // candidate values over m: λ_m * d_i^p + y[m]
            vec cand0 = lambdas * di_p + y0;
            vec cand1 = lambdas * di_p + y1;
            double best0 = cand0.min();
            double best1 = cand1.min();
            V_new[i] =(1-pi_hat[i])*(R0[i] - gamma * best0) +  pi_hat[i]*(R1[i] - gamma * best1);
            Q_robust(i, 0) = R0[i] - gamma * best0;
            Q_robust(i, 1) = R1[i] - gamma * best1;
        }

        

        V = std::move(V_new);
        V_his.col(it-1) = V;

        max_norm = max(abs(V - V_old));
        if (max(abs(V - V_old)) < tol) {
            return List::create(
                _["V_robust"]   = V,
                _["Q_robust"]   = Q_robust,
                _["V_his"]   = V_his.cols(0,it-1),
                _["V_original"] = V_original,
                _["iters"]      = it
            );
        }

    }

    return List::create(
        _["V_robust"]   = V,
        _["V_his"]   = V_his,
        _["Q_robust"]   = Q_robust,
        _["V_original"] = V_original,
        _["iters"]      = max_iter
    );
}











// [[Rcpp::export]]
arma::uvec assign_to_clusters(const arma::mat& S0,
                                 const arma::mat& centroids) {
  int n = S0.n_rows;
  int K = centroids.n_rows;
  arma::uvec clusters(n);

  for (int i = 0; i < n; ++i) {
 
    arma::rowvec diff = S0.row(i) - centroids.row(0);
    double minDist = arma::norm(diff, 2);
    int minIdx = 0;


    for (int k = 1; k < K; ++k) {
      diff = S0.row(i) - centroids.row(k);
      double d = arma::norm(diff, 2);
      if (d < minDist) {
        minDist = d;
        minIdx = k;
      }
    }


    clusters[i] = minIdx + 1;
  }

  return clusters;
}











// TV distance OPE




// [[Rcpp::export]]
List robust_ope_tv(const arma::cube& P_hat,
                               const arma::mat& R_hat,
                               const vec& pi_hat,
                               const vec& delta,
                               double gamma,
                               double tol,
                               int max_iter) {
    int K = R_hat.n_rows;
    // precompute D^p
    mat V_his(K, max_iter);
    mat Q_robust(K,2);

    double max_norm = 0;

    // solve the non-robust baseline
    arma::mat P_agg(K, K, arma::fill::zeros);
    arma::vec R_agg(K, arma::fill::zeros);
    for (int i = 0; i < K; ++i) {
      double p1 = pi_hat[i];
      for (int j = 0; j < K; ++j) {
        P_agg(i, j) = (1.0 - p1) * P_hat(i, 0, j) + p1 * P_hat(i, 1, j);
      }
      R_agg[i] = (1.0 - p1) * R_hat(i, 0) + p1 * R_hat(i, 1);
    }    

    arma::mat P0(K,K);
    arma::mat P1(K,K);
    arma::vec R0 = R_hat.col(0);
    arma::vec R1 = R_hat.col(1);
    for (int s = 0; s < K; s++) {
    
      P0.col(s) = P_hat.slice(s).col(0);
      P1.col(s) = P_hat.slice(s).col(1); 
      
    }


    vec V_original = solve(eye<mat>(K, K) - gamma * P_agg, R_agg);
    vec V = V_original;

    for (int it = 1; it <= max_iter; ++it) {
        Rcpp::checkUserInterrupt();  // allow ESC interruption
        // now compute the new V
        vec V_old = V;
        vec V_new(K);


        double min_value = arma::min(V_old);
        arma::uvec sorted_index = arma::sort_index(V_old);
        arma::vec sorted_value = V_old(sorted_index);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < K; ++i) {
            
          arma::vec row_i_0 = P0.row(i).t();  
          arma::vec sorted_row_0 = row_i_0(sorted_index);  
          //Rcpp::Rcout << "row prob" << std::endl;
          arma::vec cum_prob_0 = arma::cumsum(sorted_row_0);
          //Rcpp::Rcout << "cumu sum" << std::endl;
          //arma::uvec idx_0 = arma::find(cum_prob_0 >= 1 - delta[i], 1); 
          //double eta_0 = sorted_value(idx_0(0));
          const double eps = 1e-6;      
          double thr = std::max(0.0, 1.0 - delta[i] - eps);
          arma::uvec idx_0 = arma::find(cum_prob_0 >= thr, 1);
          arma::uword id0 = idx_0.n_elem ? idx_0(0) : (K - 1);
          double eta_0 = sorted_value(id0);


          arma::vec row_i_1 = P1.row(i).t();  
          arma::vec sorted_row_1 = row_i_1(sorted_index);  
          //Rcpp::Rcout << "row prob" << std::endl;
          arma::vec cum_prob_1 = arma::cumsum(sorted_row_1);
          //Rcpp::Rcout << "cumu sum" << std::endl;
          //arma::uvec idx_1 = arma::find(cum_prob_1 >= 1 - delta[i], 1); 
          //double eta_1 = sorted_value(idx_1(0));
          arma::uvec idx_1 = arma::find(cum_prob_1 >= thr, 1);
          arma::uword id1 = idx_1.n_elem ? idx_1(0) : (K - 1);
          double eta_1 = sorted_value(id1);


          V_new[i] =(1-pi_hat[i])*(R0[i] + gamma * (arma::dot(P0.row(i), arma::clamp(V_old, -arma::datum::inf, eta_0)) - delta[i]*(eta_0 - min_value)))
           +  pi_hat[i]*(R1[i] + gamma * (arma::dot(P1.row(i), arma::clamp(V_old, -arma::datum::inf, eta_1)) - delta[i]*(eta_1 - min_value)));

            Q_robust(i, 0) = R0[i] + gamma * (arma::dot(P0.row(i), arma::clamp(V_old, -arma::datum::inf, eta_0)) - delta[i]*(eta_0 - min_value));
            Q_robust(i, 1) = R1[i] + gamma * (arma::dot(P1.row(i), arma::clamp(V_old, -arma::datum::inf, eta_1)) - delta[i]*(eta_1 - min_value));          
        }

        

        V = std::move(V_new);
        V_his.col(it-1) = V;

        max_norm = max(abs(V - V_old));
        if (max(abs(V - V_old)) < tol) {
            return List::create(
                _["V_robust"]   = V,
                _["Q_robust"]   = Q_robust,
                _["V_his"]   = V_his.cols(0,it-1),
                _["V_original"] = V_original,
                _["iters"]      = it
            );
        }

    }

    return List::create(
        _["V_robust"]   = V,
        _["Q_robust"]   = Q_robust,
        _["V_his"]   = V_his,
        _["V_original"] = V_original,
        _["iters"]      = max_iter
    );
}


