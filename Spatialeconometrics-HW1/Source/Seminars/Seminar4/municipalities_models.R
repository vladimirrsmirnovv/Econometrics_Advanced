library(sf)
library(dplyr)
library(ggplot2)
library(patchwork)
library(spdep)
library(spatialreg)
library(sphet)

set.seed(1)

################ data and estimation sample ################

mo <- st_read("russia_municipalities_2024_extended.gpkg", layer = "municipalities_extended")

crs_albers_siberia <- "+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=105 +x_0=18500000 +y_0=0 +ellps=krass +units=m +towgs84=28,-130,-95,0,0,0,0 +no_defs"
mo <- st_transform(mo, crs_albers_siberia)
mo <- st_make_valid(mo)

# outcome: log average monthly wage 2023; covariates: size, urbanisation, climate and remoteness

mo$log_dist_million <- log(mo$dist_million_city_km)
mo <- mo %>% filter(!is.na(log_wage), !is.na(log_pop), !is.na(urban_share), !is.na(jan_temp_mean), !is.na(log_dist_million))

f <- log_wage ~ log_pop + urban_share + jan_temp_mean + log_dist_million

################ maps ################

map_wage <- ggplot(mo) +
  geom_sf(aes(fill = log_wage), lwd = 0, color = NA) +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "Log average monthly wage, 2023")

hist_wage <- ggplot(mo, aes(x = log_wage)) +
  geom_histogram(aes(fill = after_stat(x)), bins = 30, color = "white") +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  labs(x = NULL, y = NULL)

map_wage + hist_wage

################ spatial weights ################

coords <- cbind(mo$x_km, mo$y_km)
nb <- make.sym.nb(knn2nb(knearneigh(coords, k = 6)))
lw <- nb2listw(nb, style = "W")

################ global and local indices ################

# log wage is strongly clustered: I = 0.72
moran.test(mo$log_wage, lw)
moran.mc(mo$log_wage, lw, nsim = 999)
moran.plot(mo$log_wage, lw, xlab = "log wage", ylab = "W log wage", pch = 16, cex = 0.3, labels = FALSE)

# the covariates are spatially smooth too: temperature (I = 0.98) and distance to a million city (0.94)
# are almost pure geography, population (0.35) and urban share (0.19) much less so
moran.test(mo$jan_temp_mean, lw)
moran.test(mo$log_dist_million, lw)
moran.test(mo$urban_share, lw)
moran.test(mo$log_pop, lw)

# local Moran's I, closed-form p-values, Bonferroni: one test per municipality, so 0.05 / n
alpha_bonf <- 0.05 / nrow(mo)
# 156 High-High and 8 Low-Low municipalities survive the threshold 0.05 / 2036 = 2.5e-05,
# there are no High-Low / Low-High outliers at this level
lm_wage <- localmoran(mo$log_wage, lw)

cluster_wage <- as.character(attr(lm_wage, "quadr")$mean)
p_wage <- lm_wage[, "Pr(z != E(Ii))"]
cluster_wage[is.na(p_wage) | p_wage >= alpha_bonf] <- "Not significant"
mo$cluster_wage <- cluster_wage
table(mo$cluster_wage)

lisa_colors <- c("High-High" = "#d7301f", "Low-Low" = "#2166ac", "High-Low" = "#fdae61",
                 "Low-High" = "#92c5de", "Not significant" = "grey85")

ggplot(mo) +
  geom_sf(aes(fill = cluster_wage), lwd = 0, color = NA) +
  scale_fill_manual(values = lisa_colors, name = NULL) +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "Local Moran's I - log wage (6 nearest neighbours)")

################ OLS and diagnostics ################

dat <- st_drop_geometry(mo)

ols <- lm(f, dat)
summary(ols)

# residual Moran's I is 0.58; the robust LM-lag statistic (205) is larger than the robust LM-error one (72),
# both are far beyond any critical value, so the tests point to the lag but do not separate the two
lm.morantest(ols, lw)
lm.RStests(ols, lw, test = c("RSerr", "RSlag", "adjRSerr", "adjRSlag"))


################ ML estimation ################

# rho / lambda: SAR 0.77, SEM 0.83, SDM 0.78, SDEM 0.80, SARAR rho = -0.73 and lambda = 0.96;
# raw coefficients are not comparable across models (see impacts below)
# SLX: temperature is insignificant on its own (0.003) but its neighbours' value is not (-0.026)
sar <- lagsarlm(f, dat, lw)
summary(sar)

sem <- errorsarlm(f, dat, lw)
summary(sem)

slx <- lmSLX(f, dat, lw)
summary(slx)

sdm <- lagsarlm(f, dat, lw, Durbin = TRUE)
summary(sdm)

sdem <- errorsarlm(f, dat, lw, Durbin = TRUE)
summary(sdem)

# SARAR is weird
sarar <- sacsarlm(f, dat, lw)
summary(sarar)

################ GMM / IV estimation ################

# SAR: spatial 2SLS with instruments X, WX, W^2 X (spatialreg); rho = 0.79 vs 0.77 by ML
sar_iv <- stsls(f, dat, lw)
summary(sar_iv)

# sphet::spreg adds heteroskedasticity-robust versions (White standard errors for the IV estimators, GMM for
# the error coefficient robust to heteroskedasticity) and a Durbin = TRUE option, so no lags are built by hand;
# NB: sphet calls the coefficient on Wy "lambda" and the error coefficient "rho", the opposite of spatialreg
# SAR: the same estimates as stsls, robust SE 0.031 vs 0.028 classical
sar_het <- spreg(f, dat, lw, model = "lag", het = TRUE)
summary(sar_het)

# SEM: Kelejian-Prucha GMM for lambda, then FGLS for beta (spatialreg): lambda = 0.74 vs 0.83 by ML
sem_gmm <- GMerrorsar(f, dat, lw)
summary(sem_gmm)

# SEM: heteroskedasticity-robust GMM (sphet): error coefficient 0.80, close to ML; distance to a million city
# is +0.041 (p = 0.04), against +0.016 (n.s.) for the classical GMM and -0.004 (n.s.) by ML
sem_het <- spreg(f, dat, lw, model = "error", het = TRUE)
summary(sem_het)

# SARAR: generalised spatial 2SLS (spatialreg), rho = 0.81 and lambda = -0.19: close to the SAR fit
# and the opposite of the ML SARAR
sarar_gmm <- gstsls(f, dat, lw)
summary(sarar_gmm)

# sphet's SARAR fails with the same W in both parts (singular instrument matrix: W W2 X and W2 W X coincide),
# it needs two different matrices, here 12 nearest neighbours for the error part;
# Wy coefficient 0.88 (sphet's "lambda") and error coefficient -0.37 (sphet's "rho"), both significant,
# but not comparable with the same-W fits above since the second matrix is different
nb12 <- make.sym.nb(knn2nb(knearneigh(coords, k = 12)))
lw12 <- nb2listw(nb12, style = "W")
sarar_het <- spreg(f, dat, lw, lw12, model = "sarar", het = TRUE)
summary(sarar_het)

# SDM by 2SLS (sphet, Durbin = TRUE builds WX and uses W^2 X as extra instruments): does not work well here,
# the coefficient on Wy is 1.03, outside the admissible range and far from the ML rho = 0.78, and the lag
# coefficients change sign relative to ML. Temperature, its lag and the W^2 X instruments are all extremely
# smooth (Moran's I about 0.98), so X, WX and W^2 X are nearly collinear and the instruments carry little
# independent information. A hand-written 2SLS gives the same numbers, so it is not a sphet artefact
sdm_het <- spreg(f, dat, lw, model = "lag", het = TRUE, Durbin = TRUE)
summary(sdm_het)

# SDEM by GMM: error coefficient 0.78 (ML 0.80), same signs as ML for all eight coefficients, but the
# lag of urban share is 0.26 vs 0.07 by ML
sdem_het <- spreg(f, dat, lw, model = "error", het = TRUE, Durbin = TRUE)
summary(sdem_het)

# SLX has no endogenous term: OLS is the estimator (both ML and GMM coincide with it)

################ model selection ################

# LM tests from the OLS residuals are above; information criteria for the ML models:
# SARAR is lowest (AIC -1482), then SDM (-1435), SDEM (-1403), SAR (-1394), SEM (-1357); SLX is far behind (-122),
# BIC gives the same order except that SAR moves ahead of SDEM; among the models with interpretable
# parameters SDM wins by 40 AIC points over SAR and 32 over SDEM
AIC(slx, sar, sem, sdm, sdem, sarar)
BIC(slx, sar, sem, sdm, sdem, sarar)

# LR tests for nested pairs: SAR (theta = 0) and SEM (common factor) are both rejected against SDM,
# theta = 0 is rejected in SDEM, lambda = 0 in SLX -> SDEM, and both restrictions in SARAR;
# so the tests prefer SDM to SAR and SEM, and SDEM to SEM and SLX (the SARAR result is not used, see above)
LR.Sarlm(sdm, sar)      # theta = 0
LR.Sarlm(sdm, sem)      # common factor restriction, theta = -rho beta
LR.Sarlm(sdem, sem)     # theta = 0
2 * (logLik(sdem) - logLik(slx))                                       # lambda = 0, SLX is an lm so by hand
pchisq(as.numeric(2 * (logLik(sdem) - logLik(slx))), 1, lower.tail = FALSE)
LR.Sarlm(sarar, sar)    # lambda = 0
LR.Sarlm(sarar, sem)    # rho = 0

# is spatial dependence left in the residuals? no positive dependence in any of the five (I between -0.03 and -0.07)
moran.test(residuals(sar), lw)$estimate[1]
moran.test(residuals(sem), lw)$estimate[1]
moran.test(residuals(sdm), lw)$estimate[1]
moran.test(residuals(sdem), lw)$estimate[1]
moran.test(residuals(sarar), lw)$estimate[1]

# heteroskedasticity: ML assumes constant variance, and it is rejected (BP = 78, p < 1e-12), so the ML
# standard errors are not fully reliable; the robust 2SLS standard errors above are only slightly larger
# (the spatial Hausman test, Hausman.test(sem), returns a negative statistic here and is not used)
bptest.Sarlm(sdm)

################ impacts ################

# beta is not the marginal effect in SAR, SDM and SARAR, theta is part of it in SLX, SDM and SDEM
#
# the direct effect is beta times the average diagonal of (I - rho W)^-1 and the total effect is beta times its
# average row sum. Inverting a 2036 x 2036 matrix for each of the 500 simulated draws of rho is slow, but
# (I - rho W)^-1 = I + rho W + rho^2 W^2 + ..., so the average diagonal is 1 + sum_k rho^k tr(W^k) / n:
# only the traces tr(W^k), k = 1..30, are needed, and they are computed once by trW(); the total effect is
# beta / (1 - rho) for row-standardised W. Truncating at k = 30 costs nothing (0.77^30 is below 0.0005)
W_sp <- as(lw, "CsparseMatrix")
tr <- trW(W_sp, type = "mult")


# SAR: with rho = 0.77 the multiplier is about 4, so the indirect effects are about 2.7 times the direct ones
# for every variable (the ratio is fixed by rho and W): +0.1 in urban share -> +1.2% wage directly, +4.3% in total
summary(impacts(sar, tr = tr, R = 500), zstats = TRUE, short = TRUE)

# the same by 2SLS with heteroskedasticity-robust variance (sphet): identical point estimates to the
# stsls fit, indirect and total effects slightly larger than ML with about twice the simulated standard errors
summary(impacts(sar_het, tr = tr, R = 500), zstats = TRUE, short = TRUE)

# SDM (the preferred specification): the temperature effect is almost entirely indirect (direct -0.003, n.s.;
# indirect -0.021, total -0.023: one degree colder everywhere -> about +2.3% wage, almost all of it through the
# neighbours), and distance to a million
# city changes sign in the direct effect (-0.058 vs +0.019 in SAR) with a positive indirect effect (+0.137);
# urban share total 0.51, population total 0.14
summary(impacts(sdm, tr = tr, R = 500), zstats = TRUE, short = TRUE)

# SARAR: negative indirect effects follow from rho = -0.73 and contradict every other model
summary(impacts(sarar, tr = tr, R = 500), zstats = TRUE, short = TRUE)

# SLX and SDEM: direct = beta, indirect = theta, no multiplier; the temperature totals (-0.022 and -0.025)
# and the SLX total for urban share (0.51) are close to the SDM ones, but SDEM gives 0.15 for urban share,
# a third of the SDM value
summary(impacts(slx, R = 500), zstats = TRUE, short = TRUE)
impacts(sdem)

# SEM has no impacts to report: the effect of a covariate is its own coefficient
