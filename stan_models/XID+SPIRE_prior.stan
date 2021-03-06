//Full Bayesian inference fit  XID
data {
  int<lower=0> nsrc;//number of sources
  //----PSW----
  int<lower=0> npix_psw;//number of pixels
  int<lower=0> nnz_psw; //number of non neg entries in A
  vector[npix_psw] db_psw;//flattened map
  vector[npix_psw] sigma_psw;//flattened uncertianty map (assuming no covariance between pixels)
  real bkg_prior_psw;//prior estimate of background
  real bkg_prior_sig_psw;//sigma of prior estimate of background
  vector[nnz_psw] Val_psw;//non neg values in image matrix
  int Row_psw[nnz_psw];//Rows of non neg valies in image matrix
  int Col_psw[nnz_psw];//Cols of non neg values in image matrix
  vector[nsrc] psw_prior;//prior on flux
  //----PMW----
  int<lower=0> npix_pmw;//number of pixels
  int<lower=0> nnz_pmw; //number of non neg entries in A
  vector[npix_pmw] db_pmw;//flattened map
  vector[npix_pmw] sigma_pmw;//flattened uncertianty map (assuming no covariance between pixels)
  real bkg_prior_pmw;//prior estimate of background
  real bkg_prior_sig_pmw;//sigma of prior estimate of background
  vector[nnz_pmw] Val_pmw;//non neg values in image matrix
  int Row_pmw[nnz_pmw];//Rows of non neg valies in image matrix
  int Col_pmw[nnz_pmw];//Cols of non neg values in image matrix
  vector[nsrc] pmw_prior;//prior on flux

  //----PLW----
  int<lower=0> npix_plw;//number of pixels
  int<lower=0> nnz_plw; //number of non neg entries in A
  vector[npix_plw] db_plw;//flattened map
  vector[npix_plw] sigma_plw;//flattened uncertianty map (assuming no covariance between pixels)
  real bkg_prior_plw;//prior estimate of background
  real bkg_prior_sig_plw;//sigma of prior estimate of background
  vector[nnz_plw] Val_plw;//non neg values in image matrix
  int Row_plw[nnz_plw];//Rows of non neg valies in image matrix
  int Col_plw[nnz_plw];//Cols of non neg values in image matrix
  vector[nsrc] plw_prior;//prior on flux

  

}
parameters {
  vector[nsrc] src_f_psw;//source vector
  real bkg_psw;//background
  vector[nsrc] src_f_pmw;//source vector
  real bkg_pmw;//background
  vector[nsrc] src_f_plw;//source vector
  real bkg_plw;//background

}


model {
  vector[npix_psw] db_hat_psw;//model of map
  vector[npix_pmw] db_hat_pmw;//model of map
  vector[npix_plw] db_hat_plw;//model of map


  vector[nsrc+1] f_vec_psw;//vector of source fluxes and background
  vector[nsrc+1] f_vec_pmw;//vector of source fluxes and background
  vector[nsrc+1] f_vec_plw;//vector of source fluxes and background


  //Prior on background 
  bkg_psw ~normal(bkg_prior_psw,bkg_prior_sig_psw);
  bkg_pmw ~normal(bkg_prior_pmw,bkg_prior_sig_pmw);
  bkg_plw ~normal(bkg_prior_plw,bkg_prior_sig_plw);
 
  //Prior on flux of sources
  src_f_psw ~normal(psw_prior,0.5);
  src_f_pmw ~normal(pmw_prior,0.5);
  src_f_plw ~normal(plw_prior,0.5);
  

  //background is now contribution from confusion only!!
  f_vec_psw[nsrc+1] <-bkg_psw;
  f_vec_pmw[nsrc+1] <-bkg_pmw;
  f_vec_plw[nsrc+1] <-bkg_plw;

  // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
  for (n in 1:nsrc) {
    f_vec_psw[n] <- pow(10.0,src_f_psw[n]);
    f_vec_pmw[n] <- pow(10.0,src_f_pmw[n]);
    f_vec_plw[n] <- pow(10.0,src_f_plw[n]);


  }
   
 
  // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix_psw) {
    db_hat_psw[k] <- 0;
  }
  for (k in 1:nnz_psw) {
    db_hat_psw[Row_psw[k]+1] <- db_hat_psw[Row_psw[k]+1] + Val_psw[k]*f_vec_psw[Col_psw[k]+1];
      }

  for (k in 1:npix_pmw) {
    db_hat_pmw[k] <- 0;
  }
  for (k in 1:nnz_pmw) {
    db_hat_pmw[Row_pmw[k]+1] <- db_hat_pmw[Row_pmw[k]+1] + Val_pmw[k]*f_vec_pmw[Col_pmw[k]+1];
      }

  for (k in 1:npix_plw) {
    db_hat_plw[k] <- 0;
  }
  for (k in 1:nnz_plw) {
    db_hat_plw[Row_plw[k]+1] <- db_hat_plw[Row_plw[k]+1] + Val_plw[k]*f_vec_plw[Col_plw[k]+1];
      }



  // likelihood of observed map|model map
  db_psw ~ normal(db_hat_psw,sigma_psw);
  db_pmw ~ normal(db_hat_pmw,sigma_pmw);
  db_plw ~ normal(db_hat_plw,sigma_plw);


  // As actual maps are mean subtracted, requires a Jacobian adjustment
  //db_psw <- db_obs_psw - mean(db_obs_psw)
  //increment_log_prob(log((size(db_obs_psw)-1)/size(db_obs_psw)))
  //db_pmw <- db_obs_pmw - mean(db_obs_pmw)
  //increment_log_prob(log((size(db_obs_pmw)-1)/size(db_obs_pmw)))
  //db_plw <- db_obs_plw - mean(db_obs_plw)
  //increment_log_prob(log((size(db_obs_plw)-1)/size(db_obs_plw)))
    }
