functions {
int intFloor(int leftStart, int rightStart, real iReal)
{
  // This is absurd. Use bisection algorithm to find int floor.
  int left;
  int right;

  left <- leftStart;
  right <- rightStart;

  while((left + 1) < right) {
    int mid;
    // print("left, right, mid, i, ", left, ", ", right, ", ", mid, ", ", iReal);
    mid <- left + (right - left) / 2;
    if(iReal < mid) {
      right <- mid;
    }
    else {
      left <- mid;
    }
  }
  return left;
}

// Interpolate arr using a non-integral index i
// Note: 1 <= i <= length(arr)
real interpolateLinear(real[] arr, real i)
{
  int iLeft;
  real valLeft;
  int iRight;
  real valRight;

  // print("interpolating ", i);

  // Get i, value at left. If exact time match, then return value.
  iLeft <- intFloor(1, size(arr), i);
  valLeft <- arr[iLeft];
  if(iLeft == i) {
    return valLeft;
  }

  // Get i, value at right.
  iRight <- iLeft + 1;
  valRight <- arr[iRight];

  // Linearly interpolate between values at left and right.
  return valLeft + (valRight - valLeft) * (i - iLeft);
}


}
data
{
  int<lower=0> nsrc;//number of sources
  real bkg_prior[5];//prior estimate of background
  real bkg_prior_sig[5];//sigma of prior estimate of background
  vector[nsrc] f_low_lim[7];//upper limit of flux
  vector[nsrc] f_up_lim[7];//upper limit of flux
  vector[nsrc] f_mu[2];         //mu of flux distribution 
  vector[nsrc] f_sigma[2];      //sigma of flux distribution 
  real conf_prior_sig[5];
  real<lower=0.0> z_median[nsrc];
  real<lower=0.0> z_sig[nsrc];
  //----PSW----
  int<lower=0> npix_psw;//number of pixels
  int<lower=0> nnz_psw; //number of non neg entries in A
  vector[npix_psw] db_psw;//flattened map
  vector[npix_psw] sigma_psw;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_psw] Val_psw;//non neg values in image matrix
  int Row_psw[nnz_psw];//Rows of non neg values in image matrix
  int Col_psw[nnz_psw];//Cols of non neg values in image matrix
  //----PMW----
  int<lower=0> npix_pmw;//number of pixels
  int<lower=0> nnz_pmw; //number of non neg entries in A
  vector[npix_pmw] db_pmw;//flattened map
  vector[npix_pmw] sigma_pmw;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_pmw] Val_pmw;//non neg values in image matrix
  int Row_pmw[nnz_pmw];//Rows of non neg valies in image matrix
  int Col_pmw[nnz_pmw];//Cols of non neg values in image matrix
  //----PLW----
  int<lower=0> npix_plw;//number of pixels
  int<lower=0> nnz_plw; //number of non neg entries in A
  vector[npix_plw] db_plw;//flattened map
  vector[npix_plw] sigma_plw;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_plw] Val_plw;//non neg values in image matrix
  int Row_plw[nnz_plw];//Rows of non neg valies in image matrix
  int Col_plw[nnz_plw];//Cols of non neg values in image matrix
    //----PACS green----
  int<lower=0> npix_pacs100;//number of pixels
  int<lower=0> nnz_pacs100; //number of non neg entries in A
  vector[npix_pacs100] db_pacs100;//flattened map
  vector[npix_pacs100] sigma_pacs100;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_pacs100] Val_pacs100;//non neg values in image matrix
  int Row_pacs100[nnz_pacs100];//Rows of non neg values in image matrix
  int Col_pacs100[nnz_pacs100];//Cols of non neg values in image matrix
    //----PACS red----
  int<lower=0> npix_pacs160;//number of pixels
  int<lower=0> nnz_pacs160; //number of non neg entries in A
  vector[npix_pacs160] db_pacs160;//flattened map
  vector[npix_pacs160] sigma_pacs160;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_pacs160] Val_pacs160;//non neg values in image matrix
  int Row_pacs160[nnz_pacs160];//Rows of non neg values in image matrix
  int Col_pacs160[nnz_pacs160];//Cols of non neg values in image matrix

  // ----SED templates----
  int nTemp;
  int nz;
  int nband;
  real SEDs[nTemp,nband,nz];
  //real SEDs_sig[nTemp,nz];
  //-----------------------

}

parameters {
  real<lower=4.0,upper=16> Nbb[nsrc];
  real<lower=0.001,upper=7> z[nsrc]; 
  vector<lower=0,upper=1>[nband] src_f[nsrc];//vector of source src_fes
  real bkg[5];//background
  //real<lower=0.0> sigma_conf[5];

}
transformed parameters{
  real<lower=0.0> sigma_conf[5];
  for (i in 1:5){
    sigma_conf[i]=0.0;
}
}

model{
  vector[npix_psw] db_hat_psw;//model of map
  vector[npix_pmw] db_hat_pmw;//model of map
  vector[npix_plw] db_hat_plw;//model of map
  vector[npix_pacs100] db_hat_pacs100;//model of map
  vector[npix_pacs160] db_hat_pacs160;//model of map



  vector[npix_psw] sigma_tot_psw;
  vector[npix_pmw] sigma_tot_pmw;
  vector[npix_plw] sigma_tot_plw;
  vector[npix_pacs100] sigma_tot_pacs100;
  vector[npix_pacs160] sigma_tot_pacs160;

  vector[nband] f_vec[nsrc];//vector of source fluxes


  for (n in 1:nsrc){
    src_f[n,1] ~ normal(f_mu[1,n],f_sigma[1,n]) T[0,1];
    src_f[n,7] ~ normal(f_mu[2,n],f_sigma[2,n]) T[0,1];
  // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
  for (i in 1:nband) {
   f_vec[n,i] = f_low_lim[i,n]+(f_up_lim[i,n]-f_low_lim[i,n])*src_f[n,i];
  }
   }
      

   
  for (i in 1:5){
  //Prior on background
  bkg[i] ~normal(bkg_prior[i],bkg_prior_sig[i]);

 //Prior on conf
//  sigma_conf[i] ~normal(0,conf_prior_sig[i]);
  }



  for (i in 1:nsrc){
    vector[nTemp] ps;//log prob
    z[i]~normal(z_median[i],z_sig[i]);

    for (t in 1:nTemp){
        vector[nband] f_tmp;
	for (b in 1:nband){
        f_tmp[b]=pow(10.0,Nbb[i])*interpolateLinear(SEDs[t,b], z[i]*100.0);
        //f_tmp[b]=pow(10.0,Nbb[i])*interpolateLinear(SEDs[t,b], z_median[i]*100.0);
	}
	//print(f_tmp)
        ps[t]<-normal_lpdf(f_vec[i]|f_tmp,0.3*f_tmp);    //pow(10.0,Nbb[i])*interpolateLinear(SEDs_sig[t],z[i]*100.0));
    }
    target+=log_sum_exp(ps);

  }

   
  // Create model maps (i.e. db_hat = A*f) using sparse multiplication

  for (k in 1:npix_pacs100) {
    db_hat_pacs100[k] <- bkg[1];
    sigma_tot_pacs100[k]<-sqrt(square(sigma_pacs100[k])+square(sigma_conf[1]));
  }
  for (k in 1:nnz_pacs100) {
    db_hat_pacs100[Row_pacs100[k]+1] <- db_hat_pacs100[Row_pacs100[k]+1] + Val_pacs100[k]*f_vec[Col_pacs100[k]+1][2];
      }

  for (k in 1:npix_pacs160) {
    db_hat_pacs160[k] <- bkg[2];
    sigma_tot_pacs160[k]<-sqrt(square(sigma_pacs160[k])+square(sigma_conf[2]));
  }
  for (k in 1:nnz_pacs160) {
    db_hat_pacs160[Row_pacs160[k]+1] <- db_hat_pacs160[Row_pacs160[k]+1] + Val_pacs160[k]*f_vec[Col_pacs160[k]+1][3];
      }


  for (k in 1:npix_psw) {
    db_hat_psw[k] <- bkg[3];
    sigma_tot_psw[k]<-sqrt(square(sigma_psw[k])+square(sigma_conf[3]));
  }
  for (k in 1:nnz_psw) {
    db_hat_psw[Row_psw[k]+1] <- db_hat_psw[Row_psw[k]+1] + Val_psw[k]*f_vec[Col_psw[k]+1][4];
      }

  for (k in 1:npix_pmw) {
    db_hat_pmw[k] <-  bkg[4];
    sigma_tot_pmw[k]<-sqrt(square(sigma_pmw[k])+square(sigma_conf[4]));
  }
  for (k in 1:nnz_pmw) {
    db_hat_pmw[Row_pmw[k]+1] <- db_hat_pmw[Row_pmw[k]+1] + Val_pmw[k]*f_vec[Col_pmw[k]+1][5];
      }

  for (k in 1:npix_plw) {
    db_hat_plw[k] <- bkg[5];
    sigma_tot_plw[k]<-sqrt(square(sigma_plw[k])+square(sigma_conf[5]));
  }
  for (k in 1:nnz_plw) {
    db_hat_plw[Row_plw[k]+1] <- db_hat_plw[Row_plw[k]+1] + Val_plw[k]*f_vec[Col_plw[k]+1][6];
      }

      


  // likelihood of observed map|model map
  db_psw ~ normal(db_hat_psw,sigma_tot_psw);
  db_pmw ~ normal(db_hat_pmw,sigma_tot_pmw);
  db_plw ~ normal(db_hat_plw,sigma_tot_plw);
  db_pacs100 ~ normal(db_hat_pacs100,sigma_tot_pacs100);
  db_pacs160 ~ normal(db_hat_pacs160,sigma_tot_pacs160);



}
generated quantities {

matrix[nsrc,nTemp] p;
for (i in 1:nsrc){
    vector[nTemp] p_raw;
     for (t in 1:nTemp){
        vector[nband] f_tmp;
	for (b in 1:nband) {
        f_tmp[b]=pow(10.0,Nbb[i])*interpolateLinear(SEDs[t,b], z[i]*100.0);
        //f_tmp[b]=pow(10.0,Nbb[i])*interpolateLinear(SEDs[t,b], z_median[i]*100.0);
	}
        p_raw[t] = (1.0/nTemp)*exp(normal_lpdf(src_f[i]|f_tmp,0.3*f_tmp));//pow(10.0,Nbb[i])*interpolateLinear(SEDs_sig[t],z[i]*100.0)));
     }
     for (t in 1:nTemp){
     p[i,t]=p_raw[t]/sum(p_raw);
     }
 }
}