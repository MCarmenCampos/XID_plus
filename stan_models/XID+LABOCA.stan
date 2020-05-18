//Full Bayesian inference fit  XID
data {
  int<lower=0> nsrc;//number of sources
  vector[nsrc] f_low_lim[1];//upper limit of flux
  vector[nsrc] f_up_lim[1];//upper limit of flux
  real bkg_prior[1];//prior estimate of background
  real bkg_prior_sig[1];//sigma of prior estimate of background
  
  //----LABOCA------
  int<lower=0> npix_lb850;//number of pixels
  int<lower=0> nnz_lb850; //number of non neg entries in A
  vector[npix_lb850] db_lb850;//flattened map
  vector[npix_lb850] sigma_lb850;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_lb850] Val_lb850;//non neg values in image matrix
  int Row_lb850[nnz_lb850];//Rows of non neg valies in image matrix
  int Col_lb850[nnz_lb850];//Cols of non neg values in image matrix
}
parameters {
  vector<lower=0.0,upper=1.0>[nsrc] src_f[1];//source vector
  real bkg[1];//background
  real<lower=0.0,upper=0.00001> sigma_conf[1];
}


model {
  vector[npix_lb850] db_hat_lb850;//model of map
  vector[npix_lb850] sigma_tot_lb850;

  vector[nsrc] f_vec[1];//vector of source fluxes

  for (i in 1:1){
  // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
  for (n in 1:nsrc) {
    f_vec[i,n] <- f_low_lim[i,n]+(f_up_lim[i,n]-f_low_lim[i,n])*src_f[i,n];

  }

 //Prior on background 
  bkg[i] ~normal(bkg_prior[i],bkg_prior_sig[i]);

 //Prior on conf
  sigma_conf[i] ~normal(0,5);
  }
   
  // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix_lb850) {
    db_hat_lb850[k] <- bkg[1];
    sigma_tot_lb850[k]<-sqrt(square(sigma_lb850[k])+square(sigma_conf[1]));
  }
  for (k in 1:nnz_lb850) {
    db_hat_lb850[Row_lb850[k]+1] <- db_hat_lb850[Row_lb850[k]+1] + Val_lb850[k]*f_vec[1][Col_lb850[k]+1];
      }


  // likelihood of observed map|model map
  db_lb850 ~ normal(db_hat_lb850,sigma_tot_lb850);


    }
