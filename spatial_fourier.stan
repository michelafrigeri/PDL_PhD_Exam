
data {
    int<lower=0> N; 
    int<lower=0> K;
    int<lower=0> p;
    int<lower=0> G;
    
    vector[N] Y;
    matrix[N, p] X;
    vector[N] t;
    vector[N] r;
    array[N] int stazione;
    real omega;
    array[G] vector[2] coord;
}

parameters {

    real<lower=0> sigma_sq;
    
    vector[p] beta;
    
    vector[K] a;
    vector[K] b;
    real c;
    
    vector[K] a_r;
    vector[K] b_r;
    real c_r;
    
    vector[G] w;
    real<lower=0> alpha;
}


transformed parameters {
    
    vector[N] ft = to_vector(rep_array(c, N));
    vector[N] ft_r = to_vector(rep_array(c_r, N));
    
    for (j in 1:K){
     ft += a[j]*sin(j*omega*t) + b[j]*cos(j*omega*t);
     ft_r += a_r[j]*sin(j*omega*t) + b_r[j]*cos(j*omega*t);
     }
     
    cov_matrix[G] H = gp_exp_quad_cov(coord, alpha, 0.0045);
    
    vector[N] mu;
    mu = rows_dot_product(r, ft_r) + rows_dot_product(1-r, ft) + X*beta;
    mu[1:N] += w[stazione[1:N]];
}

model {  

    sigma_sq ~ inv_gamma(3, 2);
    
    beta ~ normal(0, 1);
    
    a ~ normal(0, 1);
    b ~ normal(0, 1);
    c ~ normal(0, 1);
    
    a_r ~ normal(0, 1);
    b_r ~ normal(0, 1);
    c_r ~ normal(0, 1);

    w ~ multi_normal(rep_vector(0,G), H);
    alpha ~ inv_gamma(3, 2);
    
    Y ~ normal(mu, sqrt(sigma_sq));   
}

generated quantities  { 
    vector[N] log_lik;
    for (j in 1:N) {
        log_lik[j] = normal_lpdf(Y[j] | mu[j], sqrt(sigma_sq));
    }
}

