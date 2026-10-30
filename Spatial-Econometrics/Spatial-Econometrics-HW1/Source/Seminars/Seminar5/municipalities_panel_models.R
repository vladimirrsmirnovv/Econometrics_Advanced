install.packages("fixest")
install.packages("SDPDmod")
install.packages("spgwr")
library(sf)
library(dplyr)
library(ggplot2)
library(spdep)
library(plm)
library(splm)
library(lmtest)
library(fixest)
library(SDPDmod)
library(spgwr)

set.seed(1)

################ panel and weights ################

# built by sem4_municipal_data/15a-16: Rosstat municipal database 2017-2023, one row per municipality x year,
# with the polygon (Albers-Siberia) repeated in every year
panel <- st_read("russia_municipalities_panel_2017_2023.gpkg", layer = "municipalities_panel")

# outcome: log average monthly wage (nominal, year effects absorb inflation); covariates: log population,
# jobs per resident, log industrial output per resident, fiscal autonomy (own revenue share)
f <- log_wage ~ log_avg_pop + jobs_per_pop + log_ind_pc + own_revenue_share
vars <- c("log_wage", "log_avg_pop", "jobs_per_pop", "log_ind_pc", "own_revenue_share")

# balanced panel: municipalities observed in all 7 years
balanced <- st_drop_geometry(panel) %>% filter(if_all(all_of(vars), ~ !is.na(.x))) %>% count(GID_2) %>% filter(n == 7)
d_sf <- panel %>% filter(GID_2 %in% balanced$GID_2) %>% arrange(year, GID_2)
# 1502 municipalities x 7 years = 10514 observations (1999 municipalities have a wage in all years, the other
# covariates reduce the balanced sample; the gaps are regional, so the sample is not random)
nrow(balanced)

# numeric unit index in the order of the weights matrix: plm, splm and SDPDmod sort units by the index, and
# sorting the GID_2 strings in R's locale gives a different order from the one W is built in (944 of 1502
# positions differ), which silently mismatches W and the units; with a numeric index there is nothing to sort
d_sf$unit <- match(d_sf$GID_2, d_sf$GID_2[d_sf$year == 2017])
d_sf <- d_sf %>% arrange(year, unit)
d <- st_drop_geometry(d_sf)
table(d$year)

# 6 nearest neighbours by centroid, symmetric, row-standardised; units do not move, so one matrix for all years
coords <- as.matrix(d[d$year == 2017, c("x_km", "y_km")])
nb <- make.sym.nb(knn2nb(knearneigh(coords, k = 6)))
lw <- nb2listw(nb, style = "W")
W <- listw2mat(lw)

pd <- pdata.frame(d, index = c("unit", "year"))

################ brief look: wage growth and its spatial pattern ################

w17 <- d$log_wage[d$year == 2017]
w23 <- d$log_wage[d$year == 2023]
growth <- w23 - w17

# Moran's I is 0.71 in 2017 and 0.70 in 2023: the dependence is in the levels; the 2017-2023 growth is
# weaker but still clearly clustered (I = 0.28)
moran.test(w17, lw)
moran.test(w23, lw)
moran.test(growth, lw)

mo <- d_sf[d_sf$year == 2017, ]
mo$growth <- growth

map_growth <- ggplot(mo) +
  geom_sf(aes(fill = growth), lwd = 0, color = NA) +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "Log wage growth, 2017-2023")

hist_growth <- ggplot(mo, aes(x = growth)) +
  geom_histogram(aes(fill = after_stat(x)), bins = 30, color = "white") +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  labs(x = NULL, y = NULL)

map_growth
hist_growth

################ conventional panel models ################

pool <- plm(f, pd, model = "pooling")
fe <- plm(f, pd, effect = "twoways", model = "within")
re <- plm(f, pd, effect = "twoways", model = "random")

summary(pool)
summary(fe)
summary(re)

# poolability (F = 253), random effects needed, and FE preferred to RE (Hausman 4722)
pFtest(fe, pool)
plmtest(pool, effect = "twoways", type = "bp")
phtest(fe, re)

pbgtest(fe)
pcdtest(fe, test = "cd")
pcdtest(fe, test = "rho", w = W)

coeftest(fe, vcov = vcovHC(fe, method = "arellano", cluster = "group"))
coeftest(fe, vcov = vcovSCC(fe, type = "HC1"))

################ LM tests for spatial dependence in the FE model ################

slmtest(fe, listw = lw, test = "lml")
slmtest(fe, listw = lw, test = "lme")
slmtest(fe, listw = lw, test = "rlml")
slmtest(fe, listw = lw, test = "rlme")

################ spatial models with fixed effects (ML) ################

# splm calls the coefficient on Wy "lambda" and the error coefficient "rho"
# SEM rho = 0.46, SAR lambda = 0.45, SARAR lambda = -0.66 with rho = 0.77
sem_fe <- spml(f, pd, listw = lw, model = "within", effect = "twoways", lag = FALSE, spatial.error = "b")
summary(sem_fe)

sar_fe <- spml(f, pd, listw = lw, model = "within", effect = "twoways", lag = TRUE, spatial.error = "none")
summary(sar_fe)

summary(impacts(sar_fe, listw = lw, time = 7, R = 200), zstats = TRUE, short = TRUE)

sarar_fe <- spml(f, pd, listw = lw, model = "within", effect = "twoways", lag = TRUE, spatial.error = "b")
summary(sarar_fe)

summary(impacts(sarar_fe, listw = lw, time = 7, R = 200), zstats = TRUE, short = TRUE)

pd$W_log_avg_pop <- slag(pd$log_avg_pop, lw)
pd$W_jobs_per_pop <- slag(pd$jobs_per_pop, lw)
pd$W_log_ind_pc <- slag(pd$log_ind_pc, lw)
pd$W_own_revenue_share <- slag(pd$own_revenue_share, lw)
f_d <- update(f, . ~ . + W_log_avg_pop + W_jobs_per_pop + W_log_ind_pc + W_own_revenue_share)

all.equal(as.numeric(W %*% d$log_avg_pop[d$year == 2017]), as.numeric(pd$W_log_avg_pop[pd$year == 2017]))
all.equal(as.numeric(W %*% d$own_revenue_share[d$year == 2023]), as.numeric(pd$W_own_revenue_share[pd$year == 2023]))

slx_fe <- plm(f_d, pd, effect = "twoways", model = "within")
summary(slx_fe)

sdm_fe <- spml(f_d, pd, listw = lw, model = "within", effect = "twoways", lag = TRUE, spatial.error = "none")
summary(sdm_fe)

sdem_fe <- spml(f_d, pd, listw = lw, model = "within", effect = "twoways", lag = FALSE, spatial.error = "b")
summary(sdem_fe)

sar_dyn_fe <- spml(log_wage ~ lag(log_wage, 1) + log_avg_pop + jobs_per_pop + log_ind_pc + own_revenue_share, pd,
                   listw = lw, model = "within", effect = "twoways", lag = TRUE, spatial.error = "none")
summary(sar_dyn_fe)

# short-run effects (the row for the lagged wage is not an effect of a covariate); jobs per resident 0.092 direct,
# 0.040 indirect, 0.132 total; industrial output 0.025 / 0.011 / 0.036; population is significant here
# (0.029 direct, 0.042 total); long-run effects are larger by 1 / (1 - tau - lambda), about 2.8 times
summary(impacts(sar_dyn_fe, listw = lw, time = 6, R = 200), zstats = TRUE, short = TRUE)

################ spatial models with random effects (ML) ################

# splm's random effects are individual only, so year dummies are added by hand
f_y <- update(f, . ~ . + factor(year))

# random-effects ML is slow here (SAR 40 s, SEM with spatially correlated effects a few minutes, SEM with
# independent effects about 20 min, so the last one is left out)
sar_re <- spml(f_y, pd, listw = lw, model = "random", effect = "individual", lag = TRUE, spatial.error = "none")
summary(sar_re)

# larger effects than with FE (jobs per resident total 0.52, industrial output 0.10), and population is
# significant (total 0.11), as in the RE coefficients; the year dummies are not meaningful as effects
summary(impacts(sar_re, listw = lw, time = 7, R = 200), zstats = TRUE, short = TRUE)

sem_re <- spml(f_y, pd, listw = lw, model = "random", effect = "individual", lag = FALSE, spatial.error = "kkp")
summary(sem_re)

# SAR lambda 0.61 and SEM rho 0.64 with random effects (0.45 and 0.46 with fixed effects)
# Hausman test for the spatial lag model: FE vs RE, RE rejected (472)
sphtest(sar_fe, sar_re)

################ GMM / IV with fixed effects ################

# spgm has no time effects either, so the same year dummies f_y are used
# the GMM spatial lag coefficient is larger and noisier than the ML one (SAR 0.60 with SE 0.043 vs 0.45;
# SDM 0.74 with SE 0.083 vs 0.44), the SEM error coefficient is close (0.42 vs 0.46); SARAR again gives
# lambda 0.63 with rho -0.32; impacts() is not used on spgm objects: its simulated standard errors do not line up
# with the coefficients (jobs per resident has t = 10.7 but z = 0.8)
sar_gmm <- spgm(f_y, pd, listw = lw, model = "within", lag = TRUE, spatial.error = FALSE)
summary(sar_gmm)

sem_gmm <- spgm(f_y, pd, listw = lw, model = "within", lag = FALSE, spatial.error = TRUE)
summary(sem_gmm)

sarar_gmm <- spgm(f_y, pd, listw = lw, model = "within", lag = TRUE, spatial.error = TRUE)
summary(sarar_gmm)

sdm_gmm <- spgm(f_y, pd, listw = lw, model = "within", lag = TRUE, spatial.error = FALSE,
                Durbin = ~ log_avg_pop + jobs_per_pop + log_ind_pc + own_revenue_share)
summary(sdm_gmm)

################ model selection ################

# LM tests are above; LR tests for nested ML models with two-way fixed effects: SAR in SDM (theta = 0) and
# SEM in SDEM (theta = 0). The log-likelihoods that splm stores are comparable only within the lag family
# or within the error family, and SARAR stores none, so lag vs error models and SARAR cannot be tested this way;
# theta = 0 is rejected in both pairs (p = 5e-19 and 1e-17)
pchisq(2 * (sdm_fe$logLik - sar_fe$logLik), df = 4, lower.tail = FALSE)
pchisq(2 * (sdem_fe$logLik - sem_fe$logLik), df = 4, lower.tail = FALSE)

# Bayesian posterior model probabilities (SDPDmod), static and with the lagged wage: the static comparison
# puts 99.99% on SDM; with the lagged wage in the model SDM keeps 79%, SAR gets 17% and SDEM 3%
blmpSDPD(f, pd, W = W, index = c("unit", "year"), model = list("sar", "sdm", "sem", "sdem", "slx", "ols"),
         effect = "twoways", ldet = "mc", dynamic = FALSE, prior = "uniform")$probs

blmpSDPD(f, pd, W = W, index = c("unit", "year"), model = list("sar", "sdm", "sem", "sdem", "slx", "ols"),
         effect = "twoways", ldet = "mc", dynamic = TRUE, prior = "uniform")$probs

# spatial dependence left in the residuals (Moran's I of one year at a time; plm and splm return the residuals
# ordered by unit and then year, so they are put back in the order of d first); the two-way FE residuals have
# I = 0.27 in both years, the spatial models bring it to about zero (SAR 0.040 in 2017 and 0.000 in 2023,
# SEM 0.051 / 0.013, SDM 0.042 / 0.001): a small positive remainder in 2017, none in 2023
key <- paste(d$unit, d$year, sep = "-")
res_fe <- as.numeric(residuals(fe))[match(key, names(residuals(fe)))]
res_sar <- as.numeric(residuals(sar_fe))[match(key, names(residuals(sar_fe)))]
res_sem <- as.numeric(residuals(sem_fe))[match(key, names(residuals(sem_fe)))]
res_sdm <- as.numeric(residuals(sdm_fe))[match(key, names(residuals(sdm_fe)))]

moran.test(res_fe[d$year == 2017], lw)$estimate[1]
moran.test(res_fe[d$year == 2023], lw)$estimate[1]
moran.test(res_sar[d$year == 2017], lw)$estimate[1]
moran.test(res_sar[d$year == 2023], lw)$estimate[1]
moran.test(res_sem[d$year == 2017], lw)$estimate[1]
moran.test(res_sem[d$year == 2023], lw)$estimate[1]
moran.test(res_sdm[d$year == 2017], lw)$estimate[1]
moran.test(res_sdm[d$year == 2023], lw)$estimate[1]

################ dynamic spatial models (SDPDmod) ################

# lagged wage, its spatial lag and the spatial lags of the covariates, two-way fixed effects
# (the Lee-Yu transformation option, LYtrans = TRUE, gives implausible estimates and takes minutes here)
# last year's wage matters (0.44, SE 0.017), the contemporaneous spatial lag falls to rho = 0.32 (SE 0.09), the
# lagged neighbours' wage and the lags of X are not significant, jobs per resident (0.11) and industrial
# output (0.025) stay; with T = 7 the lagged dependent variable is biased downwards in the fixed-effects
# transformation, so 0.44 is a lower bound for the persistence
# (impactsSDPDm(sdm_dyn) is not run: it inverts N x N matrices for every simulation draw, 80 s for 375 units
# and hours for 1502)
sdm_dyn <- SDPDm(f, pd, W = W, index = c("unit", "year"), model = "sdm", effect = "twoways", ldet = "mc", dynamic = TRUE)
summary(sdm_dyn)

# dynamic SAR without time effects: lagged wage 0.45, rho 0.63 (the common trend again pushes rho up)
sar_dyn <- SDPDm(f, pd, W = W, index = c("unit", "year"), model = "sar", effect = "individual", ldet = "mc", dynamic = TRUE)
summary(sar_dyn)

################ geographically weighted regression (2023 cross-section) ################

# GWR fits one weighted regression per municipality, with weights that decay with distance, so every coefficient
# becomes a surface; it is a cross-sectional method, so only the last year is used
d23 <- d[d$year == 2023, ]
mo23 <- d_sf[d_sf$year == 2023, ]
coords23 <- cbind(d23$x_km, d23$y_km)

# global OLS on the same cross-section, for comparison: population 0.035, jobs per resident 0.67, industrial
# output 0.081, own revenue share 0 (n.s.), R2 0.55; the residual Moran's I is 0.54
ols23 <- lm(f, d23)
summary(ols23)
moran.test(residuals(ols23), lw)

# bandwidth by cross-validation, adaptive (a share of the nearest neighbours, so it adapts to the very uneven
# density of municipalities); it picks 0.0038, i.e. 6 neighbours: the fit then uses 677 effective parameters
# out of 1502 observations, the local coefficients are wild (jobs per resident from -1.8 to 3.2) and the
# model nearly interpolates the data (quasi-global R2 0.93); the AICc criterion behaves the same way (0.0061,
# 9 neighbours, and about 7 minutes to compute), because wage is so strongly clustered in space
bw_cv <- gwr.sel(f, data = d23, coords = coords23, adapt = TRUE, gweight = gwr.Gauss, method = "cv")
gwr_cv <- gwr(f, data = d23, coords = coords23, adapt = bw_cv, gweight = gwr.Gauss, hatmatrix = TRUE, se.fit = TRUE)
gwr_cv

# so a fixed share of 10% of the municipalities (about 150 nearest neighbours) is used instead: 40 effective
# parameters, local coefficients between 0.37 and 1.26 for jobs per resident (global 0.67), 0.04 and 0.11 for
# industrial output (global 0.081), 0.017 and 0.099 for population (global 0.035); the sign of these three never
# changes, so the heterogeneity is in size, not direction (own revenue share does change sign, but it is not
# significant)
gwr_10 <- gwr(f, data = d23, coords = coords23, adapt = 0.1, gweight = gwr.Gauss, hatmatrix = TRUE, se.fit = TRUE)
gwr_10
names(gwr_10$results)

# maps of the local coefficients; one t-test per municipality, so the Bonferroni level 0.05 / n again,
# and locations that are not significant at that level are left grey; industrial output is significant
# everywhere (1502 of 1502), jobs per resident in 1104 (not in most of the Central district, 248 of 321, nor in
# much of Volga, 137 of 375); the effect of jobs per resident is largest in the Urals and North Caucasus (district
# means 0.95 and 0.98), then Northwestern, Volga and Southern (0.80-0.81), and smallest in Siberia, the Central
# district and the Far East (0.60, 0.60, 0.48); industrial output is strongest in the Far East (0.107)
# and Siberia (0.084)
alpha_bonf <- 0.05 / nrow(d23)
t_jobs <- gwr_10$SDF$jobs_per_pop / gwr_10$SDF$jobs_per_pop_se
t_ind <- gwr_10$SDF$log_ind_pc / gwr_10$SDF$log_ind_pc_se
mo23$b_jobs <- ifelse(2 * pt(-abs(t_jobs), df = gwr_10$results$edf) < alpha_bonf, gwr_10$SDF$jobs_per_pop, NA)
mo23$b_ind <- ifelse(2 * pt(-abs(t_ind), df = gwr_10$results$edf) < alpha_bonf, gwr_10$SDF$log_ind_pc, NA)
sum(!is.na(mo23$b_jobs))
sum(!is.na(mo23$b_ind))

ggplot(mo23) +
  geom_sf(aes(fill = b_jobs), lwd = 0, color = NA) +
  scale_fill_viridis_c(name = NULL, na.value = "grey85") +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "GWR: local effect of jobs per resident on log wage, 2023")

ggplot(mo23) +
  geom_sf(aes(fill = b_ind), lwd = 0, color = NA) +
  scale_fill_viridis_c(name = NULL, na.value = "grey85") +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "GWR: local effect of industrial output per resident on log wage, 2023")
