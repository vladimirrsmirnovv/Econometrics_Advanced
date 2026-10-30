############################################################
# Эконометрика. Домашнее задание
# Эластичность экспорта подсолнечного масла из РФ (HS 1512), 2017-2024
# Гравитационная модель: OLS / FE / RE, Tobit, Heckman, Bayes, Bootstrap
############################################################
options(scipen = 999)

# install.packages('tidyverse')
# install.packages('plm')
# install.packages('lmtest')
# install.packages('sandwich')
# install.packages('stargazer')
# install.packages('AER')
# install.packages('MCMCpack')
# install.packages('coda')

library(tidyverse)
library(plm)
library(lmtest)
library(sandwich)
library(stargazer)
library(AER)
library(MCMCpack)
library(coda)

df <- read_csv('panel_hs1512.csv', show_col_types = FALSE)

# В CEPII справочнике несколько исторических кодов с одинаковым ISO3
# (Sudan / Sudan(...2011), Germany / Fed.Rep.Germany(...1990) и т.п.)
# Из-за этого в панели появились 90 дубликатов. Уберём.
df <- df %>% distinct(iso3, year, .keep_all = TRUE)

clse <- function(m) coeftest(m, vcov = vcovHC(m, type = 'HC1', cluster = 'group'))


############################################################
# БАЗОВЫЕ РЕГРЕССИИ: OLS, FE, RE
############################################################

# Берём только торгующих, без выбросов unit value, без NA в регрессорах
reg <- df %>%
  filter(value_thsd_usd > 0,
         uv_outlier == 0,
         !is.na(log_price_clean),
         !is.na(log_gdp_d),
         !is.na(log_dist)) %>%
  mutate(
    log_value    = log(value_thsd_usd * 1000),
    lp_unfr_post = log_price_clean * unfriendly * post_2022,
    unfr_post    = unfriendly * post_2022
  )

cat('Базовая выборка:', nrow(reg), 'наблюдений,', n_distinct(reg$iso3), 'стран\n')

pdata <- pdata.frame(reg, index = c('iso3', 'year'))

# Pooled OLS
m_ols <- plm(log_value ~ log_price_clean + log_gdp_d + log_dist +
               contig + comlang_off + unfriendly + post_2022 +
               unfr_post + lp_unfr_post,
             data = pdata, model = 'pooling')
clse(m_ols)

# Fixed Effects (страны + годы);
m_fe <- plm(log_value ~ log_price_clean + log_gdp_d + unfr_post + lp_unfr_post,
            data = pdata, model = 'within', effect = 'twoways')
clse(m_fe)

# Random Effects
m_re <- plm(log_value ~ log_price_clean + log_gdp_d + log_dist +
              contig + comlang_off + unfriendly + post_2022 +
              unfr_post + lp_unfr_post,
            data = pdata, model = 'random')
clse(m_re)

# --- Тесты выбора модели ---
form <- log_value ~ log_price_clean + log_gdp_d + unfr_post + lp_unfr_post
m_pool_t <- plm(form, data = pdata, model = 'pooling')
m_fe_t   <- plm(form, data = pdata, model = 'within', effect = 'twoways')

# (1) F-тест: pooled OLS против two-way FE. H0: эффектов нет
pFtest(m_fe_t, m_pool_t)

# (2) LM Бройша-Пагана от pooled, two-way. H0: дисперсия панельных эффектов = 0
plmtest(m_pool_t, type = 'bp', effect = 'twoways')

# (3) Хаусман на согласованной спецификации.
form_h <- log_value ~ log_price_clean + log_gdp_d +
  unfr_post + lp_unfr_post + factor(year)
m_fe_h <- plm(form_h, data = pdata, model = 'within', effect = 'individual')
m_re_h <- plm(form_h, data = pdata, model = 'random',  effect = 'individual')
phtest(m_fe_h, m_re_h)             # H0: RE состоятельна; малое p -> FE

# (доп.) нужны ли годовые эффекты поверх страновых. H0: year-эффектов нет
m_fe_ind <- plm(form, data = pdata, model = 'within', effect = 'individual')
pFtest(m_fe_t, m_fe_ind)

pwartest(form, data = pdata)       # автокорреляция: H0 - её нет

m_lm <- lm(log_value ~ log_price_clean + log_gdp_d + log_dist + contig +
             comlang_off + unfriendly + post_2022 +
             unfr_post + lp_unfr_post, data = reg)
bptest(m_lm)                       # гетероскедастичность: H0 - гомоскед.

# Сводная таблица базовых моделей с КЛАСТЕРНЫМИ SE
se_ols <- sqrt(diag(vcovHC(m_ols, type = 'HC1', cluster = 'group')))
se_fe  <- sqrt(diag(vcovHC(m_fe,  type = 'HC1', cluster = 'group')))
se_re  <- sqrt(diag(vcovHC(m_re,  type = 'HC1', cluster = 'group')))

stargazer(m_ols, m_fe, m_re,
          type = 'text',
          column.labels = c('Pooled OLS', 'FE (2-way)', 'RE'),
          dep.var.labels = 'log(value)',
          se = list(se_ols, se_fe, se_re),
          star.cutoffs = c(0.1, 0.05, 0.01),
          digits = 3)





############################################################
# TOBIT
############################################################

# Все переменные, входящие в Tobit. comlang_off содержит 28 NA,
# из-за которых tobit() раньше silently отбрасывал строки (1463 -> 1435).
# Делаем отбор явным, чтобы nrow(reg_t) совпадал с выборкой модели.
tobit_vars <- c("log_price_imp", "log_gdp_d", "log_dist", "contig",
                "comlang_off", "unfriendly", "post_2022",
                "unfr_post", "lp_unfr_post")

reg_t <- df %>%
  filter(uv_outlier == 0 | value_thsd_usd == 0) %>%
  mutate(
    value_usd     = value_thsd_usd * 1000,
    log_value_p1  = log(value_usd + 1),
    log_price_imp = ifelse(is.na(log_price_clean),
                            (log_palm_oil + log_soybean_oil) / 2,
                            log_price_clean),
    lp_unfr_post  = log_price_imp * unfriendly * post_2022,
    unfr_post     = unfriendly * post_2022
  ) %>%
  filter(if_all(all_of(tobit_vars), ~ !is.na(.)))   # complete-case

cat('Tobit выборка:', nrow(reg_t), 'наблюдений,',
    sum(reg_t$value_usd > 0), 'торгующих\n')

m_tob <- tobit(log_value_p1 ~ log_price_imp + log_gdp_d + log_dist +
                 contig + comlang_off + unfriendly + post_2022 +
                 unfr_post + lp_unfr_post,
               data = reg_t, left = 0, right = Inf)

# Проверка: число наблюдений модели должно совпасть с nrow(reg_t)
cat('Наблюдений в модели Tobit:', m_tob$nobs %||% length(m_tob$y), '\n')
summary(m_tob)

# --- Latent coefficient vs marginal effect ---
b        <- coef(m_tob)
xb       <- predict(m_tob, type = "linear")
p_uncens <- pnorm(xb / m_tob$scale)
grp      <- reg_t$unfriendly * reg_t$post_2022          # 1 = unfriendly & post-2022
d_logP   <- b["log_price_imp"] + b["lp_unfr_post"] * grp # латентный эффект по набл.

cat('--- Tobit: латентные эффекты lnP ---\n')
cat('базовая группа           :', round(b["log_price_imp"], 3), '\n')
cat('unfr x post = 1          :', round(b["log_price_imp"] + b["lp_unfr_post"], 3),
    '(= beta_lnP + beta_int)\n')

cat('\n--- Tobit: наблюдаемые-эффекты (x Phi по наблюдениям) ---\n')
cat('базовая группа (grp==0)  :', round(mean(p_uncens[grp == 0] * b["log_price_imp"]), 3), '\n')
cat('unfr x post (grp==1)     :', round(mean(p_uncens[grp == 1] *
                                    (b["log_price_imp"] + b["lp_unfr_post"])), 3), '\n')

ape_logP <- mean(p_uncens * d_logP)


############################################################
# HECKMAN (exclusion restriction = только comrelig)
############################################################

reg_h <- df %>%
  filter(uv_outlier == 0 | value_thsd_usd == 0) %>%
  filter(!is.na(log_gdp_d), !is.na(log_dist),
         !is.na(contig), !is.na(comlang_off), !is.na(comrelig)) %>%
  mutate(
    value_usd     = value_thsd_usd * 1000,
    log_value     = ifelse(value_usd > 0, log(value_usd), NA),
    trade         = as.integer(value_usd > 0),
    log_price_imp = ifelse(is.na(log_price_clean),
                            (log_palm_oil + log_soybean_oil) / 2,
                            log_price_clean),
    lp_unfr_post  = log_price_imp * unfriendly * post_2022,
    unfr_post     = unfriendly * post_2022
  )

# Probit отбора. Exclusion restriction -
# только comrelig (входит в selection, отсутствует в outcome).
m_probit <- glm(trade ~ log_gdp_d + log_dist + contig + comrelig +
                  unfriendly + post_2022 + unfr_post,
                data = reg_h, family = binomial(link = 'probit'))

# --- Диагностика separation / convergence ---
cat('Probit converged:', m_probit$converged, '\n')
prob_se   <- summary(m_probit)$coefficients[, 'Std. Error']
prob_coef <- coef(m_probit)
sep_flag  <- which(abs(prob_coef) > 4 & prob_se > 10)
if (length(sep_flag) > 0) {
  cat('ВНИМАНИЕ: возможна quasi-separation в переменных:',
      paste(names(prob_coef)[sep_flag], collapse = ', '),
      '(большой коэффициент при огромной SE)\n')
} else {
  cat('Признаков separation не обнаружено.\n')
}
cat('Pseudo-R2:', round(1 - m_probit$deviance / m_probit$null.deviance, 3), '\n')

# Inverse Mills ratio
reg_h$lambda <- dnorm(predict(m_probit, type = 'link')) /
                pnorm(predict(m_probit, type = 'link'))
reg_htr <- reg_h %>% filter(trade == 1)

# Outcome equation: contig и comlang_off ВКЛЮЧЕНЫ (они не excluded),
# единственная исключённая переменная - comrelig
m_heck <- lm(log_value ~ log_price_imp + log_gdp_d + log_dist +
               contig + comlang_off + unfriendly + post_2022 +
               unfr_post + lp_unfr_post + lambda,
             data = reg_htr)
ct_heck <- coeftest(m_heck, vcov = vcovCL(m_heck, cluster = ~iso3))
print(ct_heck)

lam_p <- ct_heck['lambda', 'Pr(>|t|)']
cat(sprintf('\nlambda = %+.3f, p = %.4f', ct_heck['lambda','Estimate'], lam_p),
    ifelse(lam_p < 0.05, ' -> самоотбор значим',
                          ' -> самоотбор НЕ значим при чистом exclusion restriction'), '\n')















############################################################
# BAYESIAN (MCMCregress)
############################################################

# Используем ту же выборку, что и базовые модели (торгующие).
form_b <- log_value ~ log_price_clean + log_gdp_d + log_dist +
                      contig + comlang_off + unfriendly + post_2022 +
                      unfr_post + lp_unfr_post

# Диффузный prior: b0=0, B0=0 -> практически OLS
set.seed(2026)
m_bayes_d <- MCMCregress(form_b, data = reg,
                         burnin = 1000, mcmc = 10000,
                         b0 = 0, B0 = 0, c0 = 0.001, d0 = 0.001)

# Информативный prior: на 2-й коэф (log_price) ставим N(-0.5, 0.5)
b0_inf <- rep(0, 10); b0_inf[2] <- -0.5
B0_inf <- rep(0, 10); B0_inf[2] <- 1 / (0.5^2)   # precision = 1/var
set.seed(2026)
m_bayes_i <- MCMCregress(form_b, data = reg,
                         burnin = 1000, mcmc = 10000,
                         b0 = b0_inf, B0 = B0_inf, c0 = 0.001, d0 = 0.001)

summary(m_bayes_d)
summary(m_bayes_i)

# Диагностика сходимости: Geweke z должен быть |<1.96|, ESS большим
cat('Geweke (diffuse) max|z|:', round(max(abs(geweke.diag(m_bayes_d)$z)), 2), '\n')
cat('ESS (diffuse) min:', round(min(effectiveSize(m_bayes_d))), '\n')

post_d <- as.data.frame(m_bayes_d)
post_i <- as.data.frame(m_bayes_i)

# Posterior эластичности + вероятность отрицательности
cat('\nlog(price) diffuse:    mean', round(mean(post_d$log_price_clean), 3),
    '95% CI [', round(quantile(post_d$log_price_clean, .025), 3), ',',
    round(quantile(post_d$log_price_clean, .975), 3), ']\n')
cat('log(price) informative: mean', round(mean(post_i$log_price_clean), 3),
    '95% CI [', round(quantile(post_i$log_price_clean, .025), 3), ',',
    round(quantile(post_i$log_price_clean, .975), 3), ']\n')
cat('P(эластичность < 0): diffuse', round(mean(post_d$log_price_clean < 0), 3),
    ', informative', round(mean(post_i$log_price_clean < 0), 3), '\n')
cat('P(DiD-эффект > 0):', round(mean(post_d$lp_unfr_post > 0), 3), '\n')





############################################################
# BOOTSTRAP стандартных ошибок Хекмана (5 схем)
############################################################

B <- 1000
set.seed(2026)

# Полная 2-step Хекмана как функция (для панельного бутстрапа)
heckman_beta <- function(d) {
  pr <- tryCatch(
    glm(trade ~ log_gdp_d + log_dist + contig + comrelig +
          unfriendly + post_2022 + unfr_post,
        data = d, family = binomial('probit')),
    error = function(e) NULL, warning = function(w) NULL)
  if (is.null(pr) || !isTRUE(pr$converged)) return(NA_real_)

  d$lam <- dnorm(predict(pr, type = 'link')) / pnorm(predict(pr, type = 'link'))
  dt <- d[d$trade == 1, ]
  m <- tryCatch(
    lm(log_value ~ log_price_imp + log_gdp_d + log_dist +
         contig + comlang_off + unfriendly + post_2022 +
         unfr_post + lp_unfr_post + lam, data = dt),
    error = function(e) NULL)
  if (is.null(m)) return(NA_real_)
  b <- coef(m)['log_price_imp']
  if (is.na(b)) NA_real_ else unname(b)
}

# Вспомогательное: одношаговая переоценка outcome при фиксированной lambda
outcome_beta <- function(d) {
  tryCatch(
    coef(lm(log_value ~ log_price_imp + log_gdp_d + log_dist +
              contig + comlang_off + unfriendly + post_2022 +
              unfr_post + lp_unfr_post + lambda, data = d))['log_price_imp'],
    error = function(e) NA_real_)
}

clusters <- unique(reg_h$iso3)
by_iso   <- split(reg_h, reg_h$iso3)   # страны заранее сгруппированы
y_hat    <- predict(m_heck)
res      <- residuals(m_heck)
n_tr     <- nrow(reg_htr)
sigma_h  <- summary(m_heck)$sigma

boot_panel <- boot_par <- boot_res <- boot_wild <- boot_cwild <- numeric(B)

for (b in 1:B) {
  # 1. ПАНЕЛЬНЫЙ ДВУХШАГОВЫЙ: ресэмпл стран целиком + полная переоценка 2-step
  cl <- sample(clusters, replace = TRUE)
  d1 <- bind_rows(by_iso[cl])
  boot_panel[b] <- heckman_beta(d1)

  # 2. Параметрический: y из подогнанной outcome + N(0, sigma)
  d2 <- reg_htr
  d2$log_value <- y_hat + rnorm(n_tr, 0, sigma_h)
  boot_par[b] <- outcome_beta(d2)

  # 3. Bootstrap остатков
  d3 <- reg_htr
  d3$log_value <- y_hat + sample(res, replace = TRUE)
  boot_res[b] <- outcome_beta(d3)

  # 4. Дикий (Радемахер)
  d4 <- reg_htr
  d4$log_value <- y_hat + res * sample(c(-1, 1), n_tr, replace = TRUE)
  boot_wild[b] <- outcome_beta(d4)

  # 5. Кластерный дикий: один знак на страну
  #    as.character() - чтобы индексирование не сломалось при factor
  vc <- sample(c(-1, 1), length(clusters), replace = TRUE)
  names(vc) <- clusters
  d5 <- reg_htr
  d5$log_value <- y_hat + res * vc[as.character(reg_htr$iso3)]
  boot_cwild[b] <- outcome_beta(d5)
}

boot_list <- list('Панельный 2-step (ресэмпл стран)' = boot_panel,
                  'Параметрический'                  = boot_par,
                  'Bootstrap ошибок'                 = boot_res,
                  'Wild (Rademacher)'                = boot_wild,
                  'Кластерный wild'                  = boot_cwild)

boot_summary <- map_dfr(names(boot_list), function(nm) {
  v <- boot_list[[nm]]
  data.frame(
    method  = nm,
    success = sum(!is.na(v)),
    mean    = mean(v, na.rm = TRUE),
    SE      = sd(v, na.rm = TRUE),
    CI_lo   = quantile(v, .025, na.rm = TRUE),
    CI_hi   = quantile(v, .975, na.rm = TRUE)
  )
})
boot_summary[, 3:6] <- round(boot_summary[, 3:6], 4)
print(boot_summary)

cat('\nАналитическая cluster SE:',
    round(coeftest(m_heck, vcov = vcovCL(m_heck, cluster = ~iso3))['log_price_imp','Std. Error'], 4), '\n')





############################################################
# МОДЕЛЬ SWAMY (random coefficients)
# Проверяем, одинакова ли эластичность у всех стран или у каждой своя
############################################################

reg_s <- reg %>%
  group_by(iso3) %>% filter(n() >= 5) %>% ungroup()

cat('Выборка Swamy:', nrow(reg_s), 'наблюдений,', n_distinct(reg_s$iso3), 'стран\n')

pdat_s <- pdata.frame(reg_s, index = c('iso3', 'year'))

# Компактная спецификация: наклон по каждой стране оценивается отдельно,
# поэтому много регрессоров не вместить (не хватит степеней свободы)
form_s <- log_value ~ log_price_clean + log_gdp_d

# Модель Swamy: pvcm с model='random' оценивает среднее и дисперсию наклонов
m_swamy <- pvcm(form_s, data = pdat_s, model = 'random')
summary(m_swamy)

# pvcm 'within' даёт отдельную регрессию для каждой страны - из них видно
# разброс индивидуальных наклонов
m_swamy_w <- pvcm(form_s, data = pdat_s, model = 'within')
lp <- m_swamy_w$coefficients[['log_price_clean']]

cat('\nИндивидуальные наклоны log_price по странам:\n')
print(round(summary(lp), 3))
cat('SD наклонов:', round(sd(lp), 3), '\n')
cat('Доля стран с отрицательной эластичностью:', round(mean(lp < 0), 3), '\n')

# Тест на однородность наклонов (Swamy / Chow-type)
# H0: все страны имеют одинаковые коэффициенты (можно пулить)
# H1: коэффициенты гетерогенны (нужна модель Swamy)
m_pool_s <- plm(form_s, data = pdat_s, model = 'pooling')
cat('\n=== Тест на однородность коэффициентов ===\n')
print(pooltest(m_pool_s, m_swamy_w))

# Сравнение средней эластичности
cat('\nСредняя эластичность log_price:\n')
cat('  Pooled OLS (та же выборка):', round(coef(m_pool_s)['log_price_clean'], 3), '\n')
cat('  Swamy (mean of coefficients):', round(m_swamy$coefficients['log_price_clean'], 3), '\n')

############################################################
# СВОДКА ЭЛАСТИЧНОСТИ ПО ВСЕМ МЕТОДАМ
############################################################

elast <- data.frame(
  method = c('OLS', 'FE', 'RE', 'Tobit', 'Heckman',
             'Bayes diffuse', 'Bayes informative', 'Swamy (mean)'),
  beta = c(coef(m_ols)['log_price_clean'],
           coef(m_fe)['log_price_clean'],
           coef(m_re)['log_price_clean'],
           coef(m_tob)['log_price_imp'],
           coef(m_heck)['log_price_imp'],
           mean(post_d$log_price_clean),
           mean(post_i$log_price_clean),
           m_swamy$coefficients['log_price_clean'])
)
elast$beta <- round(elast$beta, 3)
print(elast)

