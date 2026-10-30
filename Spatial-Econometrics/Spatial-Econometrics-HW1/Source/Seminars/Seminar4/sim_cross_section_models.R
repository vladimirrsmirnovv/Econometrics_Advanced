library(spdep)
library(spatialreg)

set.seed(2026)

################ lattice, weights and covariates ################

side <- 30
n <- side^2
I <- diag(n)

nb <- cell2nb(side, side, type = "queen")
W_list <- nb2listw(nb, style = "W")
W <- listw2mat(W_list)

# spatially smooth covariates, so that WX carries information
x1 <- as.numeric(solve(I - 0.5 * W, rnorm(n)))
x2 <- as.numeric(solve(I - 0.5 * W, rnorm(n)))
eps <- rnorm(n)

################ (I - rho W)^-1 as a series ################

rho <- 0.6
A <- solve(I - rho * W)
rW <- rho * W

# partial sums I + rW + (rW)^2 + ... approach the inverse
# the largest error is 0.035 with 4 terms and 0.006 with 7 terms
S3 <- I + rW + rW %*% rW + rW %*% rW %*% rW
S6 <- S3 + rW %*% rW %*% rW %*% rW + rW %*% rW %*% rW %*% rW %*% rW + rW %*% rW %*% rW %*% rW %*% rW %*% rW
max(abs(S3 - A))
max(abs(S6 - A))

# W has zero diagonal, W^2 does not (path i -> j -> i): this is where feedback comes from
max(abs(diag(W)))
mean(diag(W %*% W))

# eigenvalues of W bound the admissible rho: (1/min, 1/max) = (-1.9, 1) here,
# the series converges for |rho| < 1, which is the usual working range
range(Re(eigen(W, only.values = TRUE)$values))

################ simulating the six models ################

# reduced forms: y = (I - rho W)^-1 [ a + X b + WX theta + (I - lambda W)^-1 eps ]
mu <- 1 + x1 - 0.5 * x2
mu_slx <- mu + 0.6 * as.numeric(W %*% x1) + 0.3 * as.numeric(W %*% x2)

y_sar   <- as.numeric(solve(I - 0.5 * W, mu + eps))                      # rho = 0.5
y_sem   <- as.numeric(mu + solve(I - 0.6 * W, eps))                      # lambda = 0.6
y_slx   <- as.numeric(mu_slx + eps)                                      # theta = (0.6, 0.3)
y_sdm   <- as.numeric(solve(I - 0.5 * W, mu_slx + eps))                  # rho = 0.5, theta = (0.6, 0.3)
y_sdem  <- as.numeric(mu_slx + solve(I - 0.6 * W, eps))                  # lambda = 0.6, theta = (0.6, 0.3)
y_sarar <- as.numeric(solve(I - 0.4 * W, mu + solve(I - 0.4 * W, eps)))  # rho = lambda = 0.4

df <- data.frame(y_sar, y_sem, y_slx, y_sdm, y_sdem, y_sarar, x1, x2)

################ estimating each model on its own data ################

# true beta = (1, -0.5) everywhere; every model recovers its own parameters
# (rho = 0.5 in SAR and SDM, lambda = 0.6 in SEM and SDEM, theta = (0.6, 0.3), SARAR 0.4 / 0.4 -> 0.39 / 0.41)
# but the raw x1 coefficient in SAR / SDM / SARAR is not the marginal effect of x1, see below
sar <- lagsarlm(y_sar ~ x1 + x2, df, W_list)
summary(sar)

sem <- errorsarlm(y_sem ~ x1 + x2, df, W_list)
summary(sem)

slx <- lmSLX(y_slx ~ x1 + x2, df, W_list)
summary(slx)

sdm <- lagsarlm(y_sdm ~ x1 + x2, df, W_list, Durbin = TRUE)
summary(sdm)

sdem <- errorsarlm(y_sdem ~ x1 + x2, df, W_list, Durbin = TRUE)
summary(sdem)

sarar <- sacsarlm(y_sarar ~ x1 + x2, df, W_list)
summary(sarar)

################ marginal effects ################

# SAR: beta is not the effect; direct, indirect and total effects come from (I - rho W)^-1 beta
# true effects of x1 (beta = 1, rho = 0.5): 1.05 / 0.95 / 2.0; impacts() gives about 1.00 / 0.90 / 1.90
S_sar <- solve(I - 0.5 * W) * 1
c(direct = mean(diag(S_sar)), total = sum(S_sar) / n, indirect = sum(S_sar) / n - mean(diag(S_sar)))

summary(impacts(sar, listw = W_list, R = 300), zstats = TRUE, short = TRUE)

# SDM: (I - rho W)^-1 (beta I + theta W); the indirect / direct ratio is now variable-specific
# true effects of x1 (beta = 1, theta = 0.6) and x2 (beta = -0.5, theta = 0.3)
# for x2 beta and theta offset: the true total effect is -0.4, the estimate is about -0.2 and not significant
S_sdm1 <- solve(I - 0.5 * W) %*% (1 * I + 0.6 * W)
S_sdm2 <- solve(I - 0.5 * W) %*% (-0.5 * I + 0.3 * W)
c(direct = mean(diag(S_sdm1)), total = sum(S_sdm1) / n, indirect = sum(S_sdm1) / n - mean(diag(S_sdm1)))
c(direct = mean(diag(S_sdm2)), total = sum(S_sdm2) / n, indirect = sum(S_sdm2) / n - mean(diag(S_sdm2)))

summary(impacts(sdm, listw = W_list, R = 300), zstats = TRUE, short = TRUE)

# SLX: direct = beta, indirect = theta
summary(impacts(slx, R = 300), zstats = TRUE, short = TRUE)

################ OLS on SAR data: Wy is endogenous ################

# (1/n) (Wy)'eps converges to sigma^2 tr[W (I - rho W)^-1] / n = 0.09, not to zero
sum(diag(W %*% solve(I - 0.5 * W))) / n

# OLS overstates rho (0.59 vs true 0.5) and pulls beta1 down (0.92); ML and 2SLS are close to the truth
df$Wy_sar <- as.numeric(W %*% y_sar)
ols_lag <- lm(y_sar ~ Wy_sar + x1 + x2, df)     # Wy treated as exogenous
coef(ols_lag)

coef(sar)                                       # ML
coef(stsls(y_sar ~ x1 + x2, df, W_list))        # spatial 2SLS, instruments X, WX, W^2X

################ OLS on SEM data: unbiased, but wrong standard errors ################

# slope estimates are similar, but OLS understates the intercept SE by more than half (0.036 vs 0.081)
# and the slope SEs only slightly (x1: 0.035 vs 0.032); this is one draw, the gap grows with lambda
# and with the spatial smoothness of X
ols_sem <- lm(y_sem ~ x1 + x2, df)
summary(ols_sem)$coefficients
summary(sem)$Coef

################ omitted variable z (gamma = 1) ################

# true model: y = 1 + 1 * x1 - 0.5 * x2 + z + noise, z is a spatially dependent variable we do not observe
# (nu is the spatially dependent part of z); nothing in the true model is a spillover of x
nu <- as.numeric(solve(I - 0.7 * W, rnorm(n)))

# (b) z = 0.5 * x1 + nu: z is correlated with x1 and spatially dependent
z_b <- 0.5 * x1 + nu
y_b <- 1 + x1 - 0.5 * x2 + z_b + rnorm(n, 0, 0.5)
df$y_b <- y_b
df$z_b <- z_b

# benchmark: if z were observed, x1 = 1 (true value)
coef(lm(y_b ~ x1 + x2 + z_b, df))

# z omitted: x1 absorbs the 0.5 * z part, so the target is 1 + 0.5 = 1.5 (we get about 1.57)
# OLS, SEM and SDEM are equally wrong: an error model handles the spatial dependence in nu (lambda = 0.64
# vs 0.7 true), but it cannot remove the bias, which comes from the correlation between x1 and z
coef(lm(y_b ~ x1 + x2, df))
coef(errorsarlm(y_b ~ x1 + x2, df, W_list))
coef(errorsarlm(y_b ~ x1 + x2, df, W_list, Durbin = TRUE))

# (c) z = 0.5 * x1 + 0.4 * Wx1 + nu: z also depends on neighbours' x1
z_c <- 0.5 * x1 + 0.4 * as.numeric(W %*% x1) + nu
df$y_c <- 1 + x1 - 0.5 * x2 + z_c + rnorm(n, 0, 0.5)

# SDEM: lag.x1 = 0.59 (expected 0.4, SE about 0.1) although x1 has no spillover on y in the true model;
# the coefficient on Wx1 is picking up how the omitted z depends on neighbours' x1
coef(errorsarlm(y_c ~ x1 + x2, df, W_list))
coef(errorsarlm(y_c ~ x1 + x2, df, W_list, Durbin = TRUE))

# SDM: rho = 0.62 and lag.x1 = -0.60; the error dependence is now written as a lag of y, which turns
# theta1 into 0.4 - rho * 1.5 = about -0.5 (the confounder term minus the common-factor term), so a negative
# theta here is neither a negative spillover nor the effect of the omitted variable alone
coef(lagsarlm(y_c ~ x1 + x2, df, W_list, Durbin = TRUE))

# spatial Hausman test, OLS vs SEM: it rejects here (statistic 169), but I would not rely on it for this:
# both estimators carry the same bias, and in a Monte Carlo (n = 225) it rejected about half the time
# even when z was independent of x1
Hausman.test(errorsarlm(y_b ~ x1 + x2, df, W_list))

################ choosing between models ################

# data generated by SEM: both LM tests reject, the robust versions point to the error model (125 vs 0.7)
lm.RStests(ols_sem, W_list, test = c("RSerr", "RSlag", "adjRSerr", "adjRSlag"))

sar_1   <- lagsarlm(y_sem ~ x1 + x2, df, W_list)
sdm_1   <- lagsarlm(y_sem ~ x1 + x2, df, W_list, Durbin = TRUE)
sdem_1  <- errorsarlm(y_sem ~ x1 + x2, df, W_list, Durbin = TRUE)
sarar_1 <- sacsarlm(y_sem ~ x1 + x2, df, W_list)

# information criteria compare non-nested models (SAR vs SEM)
# SEM has the lowest AIC (2546.0) but SDM, SDEM and SARAR are within 2 points; SAR is 68 points worse
AIC(sar_1, sem, sdm_1, sdem_1, sarar_1)
BIC(sar_1, sem, sdm_1, sdem_1, sarar_1)

# LR tests only for nested pairs: SDM vs SAR (theta = 0), SDM vs SEM (common factor, theta = -rho beta)
# SAR is rejected (p < 0.001), the SEM restriction is not (p = 0.21)
LR.Sarlm(sdm_1, sar_1)
LR.Sarlm(sdm_1, sem)

# data generated by SAR: robust LM-lag is much larger than robust LM-error (111 vs 6.5),
# SAR and SDM have the same AIC (2527.7 / 2527.8), the SAR restriction theta = 0 is not rejected (p = 0.14),
# the SEM restriction is (p < 0.001)
ols_sar <- lm(y_sar ~ x1 + x2, df)
lm.RStests(ols_sar, W_list, test = c("RSerr", "RSlag", "adjRSerr", "adjRSlag"))

sem_2   <- errorsarlm(y_sar ~ x1 + x2, df, W_list)
sdm_2   <- lagsarlm(y_sar ~ x1 + x2, df, W_list, Durbin = TRUE)
sdem_2  <- errorsarlm(y_sar ~ x1 + x2, df, W_list, Durbin = TRUE)
sarar_2 <- sacsarlm(y_sar ~ x1 + x2, df, W_list)

AIC(sar, sem_2, sdm_2, sdem_2, sarar_2)
BIC(sar, sem_2, sdm_2, sdem_2, sarar_2)
LR.Sarlm(sdm_2, sar)
LR.Sarlm(sdm_2, sem_2)

################ SARAR with the same W in both parts ################

# SAR data (lambda = 0) fitted by SARAR: lambda = -0.03 (p = 0.73), rho is less precise than in the plain SAR
# (z = 13.4 vs 16.6) and AIC is worse
summary(sarar_2)
summary(sar)

# different matrix for the error part (everything within 2.1 cells)
coords <- as.matrix(expand.grid(x = 1:side, y = 1:side))
W2_list <- nb2listw(dnearneigh(coords, 0, 2.1), style = "W")
W2 <- listw2mat(W2_list)

y_sarar2 <- as.numeric(solve(I - 0.4 * W, mu + solve(I - 0.4 * W2, eps)))
df$y_sarar2 <- y_sarar2

# with W in both parts rho = 0.39, lambda = 0.30 (SE 0.08) and AIC 2550.8;
# with the right matrices rho = 0.36, lambda = 0.49 (true 0.4 / 0.4) and AIC 2528.7, 22 points better
# (sacsarlm may print NaN warnings while optimising, they are harmless here)
summary(sacsarlm(y_sarar2 ~ x1 + x2, df, W_list))                       # both parts with W
summary(sacsarlm(y_sarar2 ~ x1 + x2, df, W_list, listw2 = W2_list))     # W for rho, W2 for lambda

################ W^2 X is not second-order neighbours ################

W_2 <- W %*% W

# structure of the non-zero entries of W^2: every unit reaches itself (i -> j -> i),
# 33% of the entries are first-order neighbours, only 63% are exactly second-order
mean(diag(W_2) > 0)
sum(W_2 > 0 & W > 0) / sum(W_2 > 0)
sum(W_2 > 0 & W == 0 & I == 0) / sum(W_2 > 0)

# exact second-order neighbours
W_pure <- listw2mat(nb2listw(nblag(nb, 2)[[2]], style = "W"))

# true first ring effect 0.6, second ring effect 0.5
# with W^2 X the first ring coefficient drops to 0.27 and the second rises to 0.92;
# with exact second-order neighbours we get 0.63 and 0.50
df$y_ring <- as.numeric(1 + x1 - 0.5 * x2 + 0.6 * W %*% x1 + 0.5 * W_pure %*% x1 + rnorm(n))
df$Wx1 <- as.numeric(W %*% x1)
df$W2x1 <- as.numeric(W_2 %*% x1)
df$Wpx1 <- as.numeric(W_pure %*% x1)

coef(lm(y_ring ~ x1 + x2 + Wx1 + W2x1, df))
coef(lm(y_ring ~ x1 + x2 + Wx1 + Wpx1, df))
