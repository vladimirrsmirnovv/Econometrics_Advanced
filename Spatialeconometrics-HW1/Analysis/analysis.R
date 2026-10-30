# Гончаров Алексей, Дугарев Денис, Русских Никита, Смирнов Владимир,
# Хмельницкий Богдан
#
# Пространственные эффекты в выручке продуктового ритейла:
# моделирование розничного товарооборота магазинов сети "Пятерочка"
# по локальным рынкам России.

options(scipen = 0, digits = 5, width = 150, warn = 1)
set.seed(2026)

# Подключение пакетов
library("arrow")
library("sf")
library("spdep")
library("spatialreg")
library("sphet")
library("splm")
library("SDPDmod")
library("spgwr")
library("plm")
library("lmtest")
library("sandwich")
library("dplyr")
library("ggplot2")

assets_directory <- "Analysis/assets"
dir.create(assets_directory, showWarnings = FALSE)

save_figure <- function(plot, file_name, width = 6, height = 4.5) {
  ggsave(file.path(assets_directory, file_name), plot,
    width = width,
    height = height, dpi = 200, bg = "white"
  )
}

section <- function(title) cat("\n\n==========", title, "==========\n")

# ==================================================
# Часть 1. Подготовка данных
# ==================================================

# 1.1. Панель "населенный пункт - месяц"
settlement_panel <- read_parquet("Data/settlement_panel.parquet")

# 1.2. Срез: одна строка на населенный пункт. Характеристики пунктов
# во времени не меняются, а выручка усредняется в логарифмах за весь
# период, что сглаживает месячный шум и большую часть сезонности
cross_section <- settlement_panel |>
  group_by(settlement_id) |>
  summarise(
    settlement_name = first(settlement_name),
    shop_region = first(shop_region),
    latitude = first(latitude),
    longitude = first(longitude),
    store_count = first(store_count),
    population_source = first(population_source),
    stores_with_revenue_jump = first(stores_with_revenue_jump),
    log_revenue = mean(log_revenue_per_store_real),
    log_revenue_24_months = mean(
      log_revenue_per_store_real[time_index >= 3]
    ),
    revenue_million = mean(revenue_per_store_real_rub) / 1e6,
    log_revenue_median_store = mean(
      log(revenue_median_store_rub / food_price_index)
    ),
    log_population = log(first(population)),
    log_households = log(first(households_median)),
    log_foot_traffic = log(first(foot_traffic_per_hour_median)),
    log_car_traffic = log(first(car_traffic_per_hour_median)),
    grocery_competitors = first(grocery_competitors_500m_mean),
    pyaterochka_density = first(pyaterochka_stores_500m_mean),
    pickup_points = first(pickup_points_100m_mean),
    medical_and_pharmacies = first(medical_and_pharmacies_300m_mean),
    checkouts = first(checkouts_mean),
    share_large_area = first(share_large_area),
    share_new_stores = first(share_new_stores),
    log_real_wage = log(mean(real_wage_rub)),
    unemployment_rate = mean(unemployment_rate_percent),
    .groups = "drop"
  ) |>
  arrange(settlement_id)
cross_section$region <- factor(cross_section$shop_region)
cross_section$log_store_count <- log(cross_section$store_count)

# 1.3. Спецификация: размер рынка, поток покупателей, конкуренция,
# формат магазина, окружение и доходы в субъекте. Кассы и доля больших
# магазинов оставлены, чтобы сравнивать магазины одного формата
format_regressors <- c("checkouts", "share_large_area")
regional_regressors <- c("log_real_wage", "unemployment_rate")
regressors <- c(
  "log_population", "log_households", "log_foot_traffic",
  "log_car_traffic", "grocery_competitors", "pyaterochka_density",
  "pickup_points", "medical_and_pharmacies", "checkouts",
  "share_large_area", "share_new_stores", "log_real_wage",
  "unemployment_rate"
)
model_formula <- reformulate(regressors, response = "log_revenue")

# Лаги WX - только для спроса и конкуренции: зарплата и безработица у
# соседей из того же субъекта почти совпадают со своими
durbin_regressors <- c(
  "log_population", "log_households", "log_foot_traffic",
  "log_car_traffic", "grocery_competitors", "pyaterochka_density"
)
durbin_formula <- reformulate(durbin_regressors)

# 1.4. Координаты, расстояния считаются по большому кругу
coordinates <- cbind(cross_section$longitude, cross_section$latitude)

# Проекция для карт
map_projection <- paste(
  "+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=60",
  "+ellps=krass +units=m +no_defs"
)
settlement_points <- st_transform(
  st_as_sf(cross_section,
    coords = c("longitude", "latitude"),
    crs = 4326
  ),
  map_projection
)
russia_regions <- st_transform(
  st_read("Data/Maps/russia_regions_complete_simplified.geojson", quiet = TRUE),
  map_projection
)
map_frame <- st_bbox(st_buffer(settlement_points, 150000))
map_theme <- theme_minimal() +
  theme(
    axis.text = element_blank(), panel.grid = element_blank(),
    legend.position = "bottom", legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  )

map_layer <- function() {
  list(
    geom_sf(
      data = russia_regions, fill = "grey96", color = "grey70",
      linewidth = 0.15
    ),
    map_theme
  )
}

map_frame_coordinates <- function() {
  coord_sf(
    xlim = map_frame[c("xmin", "xmax")],
    ylim = map_frame[c("ymin", "ymax")]
  )
}

# ==================================================
# Часть 2. Предварительный анализ данных
# ==================================================

section("2.1. Описательные статистики")
described <- cross_section[c("log_revenue", regressors)]
description <- data.frame(
  mean = sapply(described, mean),
  sd = sapply(described, sd),
  min = sapply(described, min),
  max = sapply(described, max)
)
print(round(description, 3))
cat(
  "Населенных пунктов:", nrow(cross_section),
  " магазинов:", sum(cross_section$store_count), "\n"
)
cat(
  "Доля дисперсии log_revenue, объясняемая субъектами:",
  round(summary(lm(log_revenue ~ region, cross_section))$r.squared, 3),
  "\n"
)

section("2.2. Корреляции и VIF")
correlation <- cor(described)
print(round(correlation[, "log_revenue"], 3))
variance_inflation <- sapply(regressors, function(regressor) {
  others <- reformulate(setdiff(regressors, regressor), regressor)
  1 / (1 - summary(lm(others, cross_section))$r.squared)
})
print(round(sort(variance_inflation, decreasing = TRUE), 2))

# 2.3. Распределение зависимой переменной
figure_distribution <- ggplot(cross_section, aes(log_revenue)) +
  geom_histogram(bins = 40, fill = "#3b6ea5", color = "white") +
  labs(x = "Логарифм реальной выручки магазина", y = "Число пунктов") +
  theme_minimal()
save_figure(figure_distribution, "revenue_distribution.png")

# 2.4. Динамика средней выручки магазина по месяцам
monthly_revenue <- settlement_panel |>
  group_by(period) |>
  summarise(
    nominal = mean(revenue_per_store_rub) / 1e6,
    real = mean(revenue_per_store_real_rub) / 1e6
  )
section("2.4. Средняя выручка магазина по месяцам, млн руб.")
print(as.data.frame(monthly_revenue), digits = 4)
figure_dynamics <- ggplot(monthly_revenue, aes(as.Date(period))) +
  geom_line(aes(y = nominal, color = "Номинальная")) +
  geom_line(aes(y = real, color = "В ценах января 2023 года")) +
  scale_color_manual(values = c("#3b6ea5", "#d7301f"), name = NULL) +
  labs(x = NULL, y = "Млн рублей в месяц") +
  theme_minimal() +
  theme(legend.position = "bottom")
save_figure(figure_dynamics, "revenue_dynamics.png")

# 2.5. Расположение населенных пунктов на карте России
russia_projection <- paste(
  "+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=100",
  "+ellps=krass +units=m +no_defs"
)
figure_location <- ggplot() +
  geom_sf(
    data = st_transform(russia_regions, russia_projection),
    fill = "grey96", color = "grey70", linewidth = 0.15
  ) +
  geom_sf(
    data = st_transform(settlement_points, russia_projection),
    aes(size = store_count), color = "#d7301f", alpha = 0.6
  ) +
  scale_size_area(
    max_size = 5, breaks = c(3, 10, 50, 500),
    name = "Магазинов сети"
  ) +
  map_theme
save_figure(figure_location, "settlements_russia.png",
  width = 10,
  height = 5.5
)

# 2.6. Карта выручки
figure_revenue_map <- ggplot() +
  map_layer() +
  geom_sf(
    data = settlement_points, aes(color = log_revenue),
    size = 0.9
  ) +
  scale_color_viridis_c(name = "Логарифм выручки") +
  map_frame_coordinates()
save_figure(figure_revenue_map, "revenue_map.png")

section("2.7. Базовая модель МНК")
ols_model <- lm(model_formula, cross_section)
print(summary(ols_model))
cat("Стандартные ошибки, робастные к гетероскедастичности (HC1):\n")
ols_robust <- coeftest(ols_model, vcov = vcovHC(ols_model, "HC1"))
print(ols_robust)
# Зарплата и безработица общие для всего субъекта, поэтому ошибки
# кластеризуются по субъектам
cat("Стандартные ошибки, кластеризованные по субъектам:\n")
print(coeftest(ols_model, vcov = vcovCL(ols_model,
  cluster = ~region,
  type = "HC1"
)))

# ==================================================
# Часть 3. Матрицы пространственных весов
# ==================================================

# Все матрицы нормированы по строкам. Пункт без соседей в радиусе
# отсечения получает ближайшего
nearest_neighbour <- knn2nb(knearneigh(coordinates,
  k = 1,
  longlat = TRUE
))

distance_weights <- function(cutoff_km, weight_function) {
  neighbours <- union.nb(
    dnearneigh(coordinates, 0, cutoff_km, longlat = TRUE),
    nearest_neighbour
  )
  distances <- nbdists(neighbours, coordinates, longlat = TRUE)
  nb2listw(neighbours,
    glist = lapply(distances, weight_function),
    style = "W"
  )
}

# 3.1. Экспоненциальное затухание exp(-d / 50) с отсечением 300 км,
# вес соседа падает вдвое каждые 35 км
weights_exponential_50 <- distance_weights(
  300, function(distance) exp(-distance / 50)
)

# 3.2. Симметризованная матрица k ближайших соседей
weights_knn_8 <- nb2listw(
  make.sym.nb(knn2nb(knearneigh(coordinates, k = 8, longlat = TRUE))),
  style = "W"
)

# 3.3. Обратные расстояния 1/d и 1/d^2
weights_inverse_200 <- distance_weights(
  200, function(distance) 1 / distance
)
weights_inverse_squared_400 <- distance_weights(
  400, function(distance) 1 / distance^2
)

# 3.4. Пороговая матрица соседства в радиусе 100 км
band_100 <- dnearneigh(coordinates, 0, 100, longlat = TRUE)
weights_band_100 <- nb2listw(band_100, style = "W", zero.policy = TRUE)
cat(
  "Пунктов без соседей в радиусе 100 км:", sum(card(band_100) == 0),
  "\n"
)

# 3.5. Экономическая матрица: 10 ближайших пунктов по численности
# населения без учета географии
population_position <- cbind(
  cross_section$log_population + rnorm(nrow(cross_section), 0, 1e-9), 0
)
weights_economic <- nb2listw(
  knn2nb(knearneigh(population_position, k = 10)),
  style = "W"
)

section("3.6. Сравнение матриц весов")
candidate_weights <- list(
  "exp(-d/50), 300 км" = weights_exponential_50,
  "kNN, k = 8" = weights_knn_8,
  "1/d, 200 км" = weights_inverse_200,
  "1/d^2, 400 км" = weights_inverse_squared_400,
  "Порог 100 км" = weights_band_100,
  "Экономическая" = weights_economic
)
compare_weights <- function(weights_name) {
  weight_list <- candidate_weights[[weights_name]]
  moran <- moran.test(cross_section$log_revenue, weight_list,
    zero.policy = TRUE
  )
  sar_candidate <- lagsarlm(model_formula, cross_section, weight_list,
    zero.policy = TRUE
  )
  sdm_candidate <- lagsarlm(model_formula, cross_section, weight_list,
    Durbin = durbin_formula, zero.policy = TRUE
  )
  data.frame(
    weights = weights_name,
    neighbours = mean(card(weight_list$neighbours)),
    moran = unname(moran$estimate[1]),
    rho_sar = sar_candidate$rho,
    loglik_sar = as.numeric(logLik(sar_candidate)),
    rho_sdm = sdm_candidate$rho,
    loglik_sdm = as.numeric(logLik(sdm_candidate))
  )
}
weights_comparison <- do.call(rbind, lapply(
  names(candidate_weights),
  compare_weights
))
print(weights_comparison, digits = 4, row.names = FALSE)

section("3.7. Профиль правдоподобия по масштабу затухания")
decay_grid <- c(15, 20, 30, 40, 50, 60, 75, 100, 150)
decay_profile <- data.frame(
  decay_km = decay_grid,
  loglik_sdm = sapply(decay_grid, function(decay) {
    weight_list <- distance_weights(
      6 * decay, function(distance) exp(-distance / decay)
    )
    as.numeric(logLik(lagsarlm(model_formula, cross_section, weight_list,
      Durbin = durbin_formula
    )))
  })
)
print(decay_profile, digits = 5)
figure_profile <- ggplot(decay_profile, aes(decay_km, loglik_sdm)) +
  geom_line(color = "#3b6ea5") +
  geom_point(color = "#3b6ea5") +
  labs(
    x = "Масштаб затухания, км",
    y = "Логарифм правдоподобия SDM"
  ) +
  theme_minimal()
save_figure(figure_profile, "weights_profile.png")

# Основная матрица
weights <- weights_exponential_50

section("3.8. Эффективное расстояние до соседей в основной матрице")
# Средневзвешенное расстояние до соседей sum_j w_ij d_ij
effective_distance <- mapply(
  function(distances, row_weights) sum(distances * row_weights),
  nbdists(weights$neighbours, coordinates, longlat = TRUE),
  weights$weights
)
print(round(quantile(
  effective_distance,
  c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
), 1))
weights_traces <- trW(as(weights, "CsparseMatrix"),
  m = 30,
  type = "mult"
)

# ==================================================
# Часть 4. Глобальная пространственная автокорреляция
# ==================================================

section("4.1. Глобальный индекс Морана")
# H0: I = E(I), случайное размещение; H1: I > E(I), кластеризация
moran_revenue <- moran.test(cross_section$log_revenue, weights)
print(moran_revenue)
moran_permutation <- moran.mc(cross_section$log_revenue, weights,
  nsim = 999
)
print(moran_permutation)

section("4.2. Индекс Гири")
# H0: C = 1; H1: C < 1, положительная автокорреляция
geary_revenue <- geary.test(cross_section$log_revenue, weights)
print(geary_revenue)

section("4.3. Глобальная статистика Гетиса-Орда")
# G считается по выручке в рублях и бинарной матрице соседства в
# радиусе 100 км, для которых она определена
# H0: G = E(G); H1: G > E(G)
weights_band_binary <- nb2listw(band_100, style = "B", zero.policy = TRUE)
getis_revenue <- globalG.test(cross_section$revenue_million,
  weights_band_binary,
  zero.policy = TRUE
)
print(getis_revenue)

section("4.4. Двумерный индекс Морана")
# Выручка пункта и численность или автотрафик соседей
# H0: I_xy = 0; H1: I_xy != 0, двусторонний перестановочный тест
permutations <- 999
two_sided_pseudo_p <- function(bivariate) {
  smaller_tail <- min(
    sum(bivariate$t <= bivariate$t0),
    sum(bivariate$t >= bivariate$t0)
  )
  min(1, 2 * (smaller_tail + 1) / (permutations + 1))
}
bivariate_population <- moran_bv(cross_section$log_revenue,
  cross_section$log_population, weights,
  nsim = permutations
)
bivariate_population_p <- two_sided_pseudo_p(bivariate_population)
bivariate_traffic <- moran_bv(cross_section$log_revenue,
  cross_section$log_car_traffic, weights,
  nsim = permutations
)
bivariate_traffic_p <- two_sided_pseudo_p(bivariate_traffic)
cat(
  "I_xy с численностью =", round(bivariate_population$t0, 4),
  " pseudo p =", bivariate_population_p, "\n"
)
cat(
  "I_xy с автотрафиком =", round(bivariate_traffic$t0, 4),
  " pseudo p =", bivariate_traffic_p, "\n"
)


section("4.5. Индекс Морана регрессоров")
regressor_moran <- sapply(regressors, function(variable) {
  unname(moran.test(cross_section[[variable]], weights)$estimate[1])
})
print(round(regressor_moran, 3))

# Диаграмма Морана и доли квадрантов
moran_scatter <- data.frame(
  standardized = as.numeric(scale(cross_section$log_revenue))
)
moran_scatter$spatial_lag <- lag.listw(
  weights,
  moran_scatter$standardized
)
quadrant <- with(moran_scatter, ifelse(
  standardized >= 0,
  ifelse(spatial_lag >= 0, "High-High", "High-Low"),
  ifelse(spatial_lag >= 0, "Low-High", "Low-Low")
))
cat("\nДоли квадрантов диаграммы Морана:\n")
print(round(prop.table(table(quadrant)), 3))
# Цвет точки - квадрант диаграммы, в углах доли пунктов
moran_scatter$quadrant <- factor(
  quadrant,
  levels = c("High-High", "Low-High", "Low-Low", "High-Low"),
  labels = c("HH", "LH", "LL", "HL")
)
quadrant_colors <- c(
  HH = "#d7301f", LH = "#2166ac", LL = "#1a9850", HL = "#f0b400"
)
quadrant_labels <- data.frame(
  quadrant = c("HH", "LH", "LL", "HL"),
  standardized = c(1, -1, -1, 1) * max(abs(moran_scatter$standardized)),
  spatial_lag = c(1, 1, -1, -1) * max(abs(moran_scatter$spatial_lag)),
  share = as.numeric(prop.table(table(moran_scatter$quadrant))[
    c("HH", "LH", "LL", "HL")
  ])
)
quadrant_labels$text <- paste0(
  quadrant_labels$quadrant, ", ",
  round(100 * quadrant_labels$share, 1), "%"
)
figure_moran <- ggplot(moran_scatter, aes(standardized, spatial_lag)) +
  geom_hline(yintercept = 0, linetype = 2, color = "grey50") +
  geom_vline(xintercept = 0, linetype = 2, color = "grey50") +
  geom_point(aes(color = quadrant), alpha = 0.6, size = 0.9) +
  geom_smooth(
    method = "lm", formula = y ~ x, color = "grey20",
    se = FALSE
  ) +
  geom_label(
    data = quadrant_labels,
    aes(label = text, color = quadrant),
    hjust = "inward", vjust = "inward", size = 4, fontface = "bold"
  ) +
  scale_color_manual(values = quadrant_colors, guide = "none") +
  labs(
    x = "Логарифм выручки, стандартизованный",
    y = "Пространственный лаг"
  ) +
  theme_minimal()
save_figure(figure_moran, "moran_scatter.png")

# ==================================================
# Часть 5. Локальная пространственная автокорреляция
# ==================================================

# Значимость локальных статистик - двусторонний перестановочный тест
# с 9999 перестановками и поправкой Бенджамини-Хохберга. Для индекса
# Морана добавлены поправка Бенджамини-Иекутиели, верная при
# зависимых тестах, и Бонферрони на аналитических p-значениях
significant <- function(p_values) p.adjust(p_values, "fdr") < 0.05

cluster_names <- c(
  "High-High" = "Высокий - высокий", "Low-Low" = "Низкий - низкий",
  "High-Low" = "Высокий - низкий", "Low-High" = "Низкий - высокий",
  "Other Positive" = "Прочее сходство", "Negative" = "Различие",
  "Positive" = "Сходство", "High" = "Горячая зона",
  "Low" = "Холодная зона"
)
translate_clusters <- function(clusters, p_values) {
  translated <- unname(cluster_names[as.character(clusters)])
  translated[!significant(p_values)] <- "Не значим"
  translated
}

section("5.1. Локальный индекс Морана")
local_moran <- localmoran_perm(cross_section$log_revenue, weights,
  nsim = 9999
)
settlement_points$local_moran <- translate_clusters(
  attr(local_moran, "quadr")$mean, local_moran[, "Pr(z != E(Ii)) Sim"]
)
print(table(settlement_points$local_moran))
region_clusters <- table(
  cross_section$shop_region,
  settlement_points$local_moran
)
print(region_clusters[rowSums(region_clusters[, colnames(region_clusters)
!= "Не значим",
drop = FALSE
]) >= 5, ])
local_moran_analytic <- localmoran(cross_section$log_revenue, weights)
bonferroni_cluster <- as.character(attr(
  local_moran_analytic,
  "quadr"
)$mean)
bonferroni_cluster[local_moran_analytic[, "Pr(z != E(Ii))"] >=
  0.05 / nrow(cross_section)] <- "Не значим"
cat("Аналитические p-значения, поправка Бонферрони:\n")
print(table(bonferroni_cluster))
cat(
  "Значимых пунктов по аналитическим p-значениям: FDR",
  sum(significant(local_moran_analytic[, "Pr(z != E(Ii))"])),
  " Бонферрони", sum(bonferroni_cluster != "Не значим"), "\n"
)
cat(
  "Значимых пунктов по перестановочным p-значениям, поправка",
  "Бенджамини-Иекутиели:",
  sum(p.adjust(local_moran[, "Pr(z != E(Ii)) Sim"], "BY") < 0.05), "\n"
)

section("5.2. Локальная статистика Гетиса-Орда Gi*")
# Основная матрица, в которую добавлен сам пункт с весом 1
neighbours_with_self <- include.self(weights$neighbours)
distances_with_self <- nbdists(neighbours_with_self, coordinates,
  longlat = TRUE
)
weights_with_self <- nb2listw(
  neighbours_with_self,
  glist = lapply(
    distances_with_self,
    function(distance) exp(-distance / 50)
  ),
  style = "W"
)
local_getis <- localG_perm(cross_section$log_revenue, weights_with_self,
  nsim = 9999
)
settlement_points$local_getis <- translate_clusters(
  attr(local_getis, "cluster"),
  attr(local_getis, "internals")[, "Pr(z != E(Gi)) Sim"]
)
print(table(settlement_points$local_getis))

section("5.3. Локальный индекс Гири")
local_geary <- localC_perm(cross_section$log_revenue, weights,
  nsim = 9999
)
settlement_points$local_geary <- translate_clusters(
  attr(local_geary, "cluster"),
  attr(local_geary, "pseudo-p")[, "Pr(z != E(Ci)) Sim"]
)
print(table(settlement_points$local_geary))

section("5.4. Многомерный локальный индекс Гири")
# Сходство с соседями сразу по трем стандартизованным показателям
multivariate_variables <- data.frame(
  log_revenue = as.numeric(scale(cross_section$log_revenue)),
  log_car_traffic = as.numeric(scale(cross_section$log_car_traffic)),
  pyaterochka_density = as.numeric(
    scale(cross_section$pyaterochka_density)
  )
)
multivariate_geary <- localC_perm(multivariate_variables, weights,
  nsim = 9999
)
settlement_points$multivariate_geary <- translate_clusters(
  attr(multivariate_geary, "cluster"),
  attr(multivariate_geary, "pseudo-p")[, "Pr(z != E(Ci)) Sim"]
)
print(table(settlement_points$multivariate_geary))

section("5.5. Двумерный локальный индекс Морана")
local_bivariate <- localmoran_bv(cross_section$log_revenue,
  cross_section$log_population, weights,
  nsim = 9999
)
print(table(translate_clusters(
  attr(local_bivariate, "quadr")$mean,
  local_bivariate[, "Pr(z != E(Ibvi)) Sim"]
)))

cluster_colors <- c(
  "Высокий - высокий" = "#d7301f", "Низкий - низкий" = "#2166ac",
  "Высокий - низкий" = "#fdae61", "Низкий - высокий" = "#92c5de",
  "Прочее сходство" = "#fdae61", "Различие" = "#92c5de",
  "Сходство" = "#d7301f", "Горячая зона" = "#d7301f",
  "Холодная зона" = "#2166ac", "Не значим" = "grey80"
)

cluster_map <- function(cluster_column) {
  ordered_points <- settlement_points[
    order(settlement_points[[cluster_column]] != "Не значим"),
  ]
  ggplot() +
    map_layer() +
    geom_sf(
      data = ordered_points,
      aes(color = .data[[cluster_column]]), size = 0.9
    ) +
    scale_color_manual(values = cluster_colors, name = NULL) +
    map_frame_coordinates() +
    guides(color = guide_legend(
      nrow = 2,
      override.aes = list(size = 3)
    ))
}
save_figure(cluster_map("local_moran"), "map_local_moran.png")
save_figure(cluster_map("local_getis"), "map_local_getis.png")
save_figure(cluster_map("local_geary"), "map_local_geary.png")
save_figure(
  cluster_map("multivariate_geary"),
  "map_multivariate_geary.png"
)

# ==================================================
# Часть 6. Диагностика пространственной зависимости в МНК
# ==================================================

section("6.1. Тест Морана для остатков МНК")
# H0: остатки пространственно не автокоррелированы
ols_residual_moran <- lm.morantest(ols_model, weights)
print(ols_residual_moran)

section("6.2. Тесты множителей Лагранжа")
# Подход от простой модели к сложной (Anselin et al., 1996)
lagrange_tests <- lm.RStests(ols_model, weights, test = "all")
print(lagrange_tests)

section("6.3. Тест Бройша-Пагана для МНК")
print(bptest(ols_model))

# ==================================================
# Часть 7. Пространственные модели
# ==================================================

# Оценки методом максимального правдоподобия
slx_model <- lmSLX(model_formula, cross_section, weights,
  Durbin = durbin_formula
)
sar_model <- lagsarlm(model_formula, cross_section, weights)
sem_model <- errorsarlm(model_formula, cross_section, weights)
sdm_model <- lagsarlm(model_formula, cross_section, weights,
  Durbin = durbin_formula
)
sdem_model <- errorsarlm(model_formula, cross_section, weights,
  Durbin = durbin_formula
)
sarar_model <- sacsarlm(model_formula, cross_section, weights)
gns_model <- sacsarlm(model_formula, cross_section, weights,
  Durbin = durbin_formula
)
# SDM с лагами всех регрессоров нужна для проверки ограничения общего
# множителя theta = -rho beta, сводящего ее к SEM
sdm_all_lags_model <- lagsarlm(model_formula, cross_section, weights,
  Durbin = TRUE
)

section("7.1. SLX")
print(summary(slx_model))
cat("SLX, робастные стандартные ошибки HC1:\n")
print(coeftest(slx_model, vcov = vcovHC(slx_model, "HC1")))
section("7.2. SAR")
sar_summary <- summary(sar_model, Nagelkerke = TRUE)
print(sar_summary)
section("7.3. SEM")
print(summary(sem_model, Nagelkerke = TRUE))
section("7.4. SDM")
sdm_summary <- summary(sdm_model, Nagelkerke = TRUE)
print(sdm_summary)
section("7.5. SDEM")
print(summary(sdem_model, Nagelkerke = TRUE))
section("7.6. SARAR")
sarar_summary <- summary(sarar_model, Nagelkerke = TRUE)
print(sarar_summary)
section("7.7. GNS")
gns_summary <- summary(gns_model, Nagelkerke = TRUE)
print(gns_summary)

section("7.8. LR-тесты вложенных моделей")
# Подход от общей модели к упрощенной (Elhorst, 2014)
likelihood_ratio <- function(general, restricted, transition,
                             restriction) {
  statistic <- 2 * (as.numeric(logLik(general)) -
    as.numeric(logLik(restricted)))
  degrees <- attr(logLik(general), "df") -
    attr(logLik(restricted), "df")
  data.frame(
    transition = transition, restriction = restriction,
    statistic = statistic, df = degrees,
    p_value = pchisq(statistic, degrees, lower.tail = FALSE)
  )
}
likelihood_ratio_tests <- rbind(
  likelihood_ratio(gns_model, sdm_model, "GNS -> SDM", "lambda = 0"),
  likelihood_ratio(gns_model, sdem_model, "GNS -> SDEM", "rho = 0"),
  likelihood_ratio(gns_model, sarar_model, "GNS -> SARAR", "theta = 0"),
  likelihood_ratio(sdm_model, sar_model, "SDM -> SAR", "theta = 0"),
  likelihood_ratio(
    sdm_all_lags_model, sem_model, "SDM, все WX -> SEM",
    "theta = -rho beta"
  ),
  likelihood_ratio(
    sdm_all_lags_model, sdm_model, "SDM, все WX -> SDM",
    "7 лагов равны нулю"
  ),
  likelihood_ratio(sdm_model, slx_model, "SDM -> SLX", "rho = 0"),
  likelihood_ratio(sdem_model, sem_model, "SDEM -> SEM", "theta = 0"),
  likelihood_ratio(sarar_model, sar_model, "SARAR -> SAR", "lambda = 0"),
  likelihood_ratio(sar_model, ols_model, "SAR -> МНК", "rho = 0"),
  likelihood_ratio(
    sdm_model, ols_model, "SDM -> МНК",
    "rho = 0, theta = 0"
  )
)
print(likelihood_ratio_tests, digits = 4, row.names = FALSE)

section("7.9. Информационные критерии")
# Остаточная автокорреляция: тест Морана для МНК и SLX, LM-тест для
# моделей с лагом y
models <- list(
  "МНК" = ols_model, SLX = slx_model, SAR = sar_model,
  SEM = sem_model, SDM = sdm_model, SDEM = sdem_model,
  SARAR = sarar_model, GNS = gns_model
)
spatial_parameter <- function(model, parameter) {
  value <- model[[parameter]]
  if (is.null(value)) {
    return(NA_real_)
  }
  value
}
parameter_p_value <- function(model, parameter) {
  if (parameter == "rho" && !is.null(model$rho)) {
    return(2 * pnorm(-abs(model$rho / model$rho.se)))
  }
  if (parameter == "lambda" && !is.null(model$lambda)) {
    return(2 * pnorm(-abs(model$lambda / model$lambda.se)))
  }
  NA_real_
}
ols_residual_test <- lm.morantest(ols_model, weights)
slx_residual_test <- lm.morantest(slx_model, weights)
residual_statistic <- c(
  ols_residual_test$estimate[1],
  slx_residual_test$estimate[1],
  sar_summary$LMtest, NA, sdm_summary$LMtest,
  NA, NA, NA
)
residual_p_value <- c(
  ols_residual_test$p.value,
  slx_residual_test$p.value,
  pchisq(sar_summary$LMtest, 1, lower.tail = FALSE),
  NA,
  pchisq(sdm_summary$LMtest, 1, lower.tail = FALSE),
  NA, NA, NA
)
model_selection <- data.frame(
  model = names(models),
  rho = sapply(models, spatial_parameter, "rho"),
  rho_p = sapply(models, parameter_p_value, "rho"),
  lambda = sapply(models, spatial_parameter, "lambda"),
  lambda_p = sapply(models, parameter_p_value, "lambda"),
  parameters = sapply(models, function(model) {
    attr(logLik(model), "df")
  }),
  log_likelihood = sapply(models, function(model) {
    as.numeric(logLik(model))
  }),
  aic = sapply(models, AIC),
  bic = sapply(models, BIC),
  residual_statistic = residual_statistic,
  residual_p_value = residual_p_value
)
print(model_selection, digits = 4, row.names = FALSE)

section("7.10. Тест Бройша-Пагана для SDM")
print(bptest.Sarlm(sdm_model))

section("7.11. Оценки, робастные к гетероскедастичности")
# SAR: пространственный двухшаговый МНК, инструменты X, WX, W^2 X
print(summary(stsls(model_formula, cross_section, weights,
  robust = TRUE
)))
# SEM: FGLS Келеджиана-Пручи с преобразованием (I - lambda W)
print(summary(GMerrorsar(model_formula, cross_section, weights)))
# SEM: GMM, робастный к гетероскедастичности (sphet)
print(summary(spreg(model_formula, cross_section, weights,
  model = "error",
  het = TRUE
)))
# SDM: двухшаговый МНК со стандартными ошибками Уайта
sdm_robust_model <- spreg(model_formula, cross_section, weights,
  model = "lag", het = TRUE,
  Durbin = durbin_formula
)
print(summary(sdm_robust_model))
# Тест Вальда для theta = 0 с робастной ковариацией, поскольку LR-тест
# предполагает однородную дисперсию
durbin_lags <- grep("^lag_", rownames(sdm_robust_model$var), value = TRUE)
durbin_estimates <- sdm_robust_model$coefficients[durbin_lags, 1]
robust_wald <- as.numeric(t(durbin_estimates) %*%
  solve(sdm_robust_model$var[
    durbin_lags,
    durbin_lags
  ]) %*%
  durbin_estimates)
cat(
  "Робастный тест Вальда theta = 0: W =", robust_wald, " df =",
  length(durbin_lags), " p =",
  pchisq(robust_wald, length(durbin_lags), lower.tail = FALSE), "\n"
)


# ==================================================
# Часть 8. Прямые, косвенные и общие эффекты
# ==================================================

# В SAR и SDM эффект расходится на соседей всех порядков через
# (I - rho W)^-1, в SLX и SDEM - только на прямых соседей
simulations <- 2000

section("8.1. SDM: эффекты")
sdm_impacts <- impacts(sdm_model,
  tr = weights_traces, R = simulations,
  Q = 5
)
sdm_impacts_summary <- summary(sdm_impacts, zstats = TRUE, short = TRUE)
print(sdm_impacts_summary)
# 95%-интервалы прямых эффектов конкурентов и плотности сети
direct_interval <- function(effects) {
  simulated <- as.matrix(effects$sres$direct)
  colnames(simulated) <- sub(" dy/dx$", "", colnames(simulated))
  round(apply(
    simulated[, c("grocery_competitors", "pyaterochka_density")],
    2, quantile, c(0.025, 0.975)
  ), 4)
}
cat("95%-интервалы прямых эффектов SDM:\n")
print(direct_interval(sdm_impacts))

section("8.2. SDM, робастный 2МНК: эффекты")
sdm_robust_impacts <- impacts(sdm_robust_model,
  tr = weights_traces,
  R = simulations
)
sdm_robust_impacts_summary <- summary(sdm_robust_impacts,
  zstats = TRUE,
  short = TRUE
)
print(sdm_robust_impacts_summary)
cat("95%-интервалы прямых эффектов, робастный 2МНК:\n")
print(direct_interval(sdm_robust_impacts))


section("8.3. SAR: эффекты")
sar_impacts <- impacts(sar_model, tr = weights_traces, R = simulations)
sar_impacts_summary <- summary(sar_impacts, zstats = TRUE, short = TRUE)
print(sar_impacts_summary)

section("8.4. SLX и SDEM: локальные эффекты")
slx_impacts <- impacts(slx_model)
sdem_impacts <- impacts(sdem_model)
slx_impacts_summary <- summary(slx_impacts, zstats = TRUE, short = TRUE)
sdem_impacts_summary <- summary(sdem_impacts, zstats = TRUE, short = TRUE)
print(slx_impacts_summary)
print(sdem_impacts_summary)

# Сравнение косвенных эффектов ключевых переменных по моделям
compared_variables <- c(
  "log_population", "log_car_traffic",
  "grocery_competitors", "pyaterochka_density"
)
compared_labels <- c(
  "Численность", "Автотрафик", "Конкуренты",
  "Плотность сети"
)
indirect_comparison <- data.frame(
  model = rep(c("SAR", "SDM", "SLX", "SDEM"),
    each = length(compared_variables)
  ),
  variable = rep(compared_labels, 4),
  indirect = c(
    setNames(
      sar_impacts$res$indirect,
      regressors
    )[compared_variables],
    sdm_impacts$res$indirect[compared_variables],
    slx_impacts$impacts$indirect[compared_variables],
    sdem_impacts$impacts$indirect[compared_variables]
  ),
  standard_error = c(
    sar_impacts_summary$semat[
      paste(compared_variables, "dy/dx"),
      "Indirect"
    ],
    sdm_impacts_summary$semat[
      paste(compared_variables, "dy/dx"),
      "Indirect"
    ],
    slx_impacts_summary$semat[
      paste(compared_variables, "dy/dx"),
      "Indirect"
    ],
    sdem_impacts_summary$semat[
      paste(compared_variables, "dy/dx"),
      "Indirect"
    ]
  )
)
indirect_comparison$model <- factor(indirect_comparison$model,
  levels = c(
    "SAR", "SDM", "SLX",
    "SDEM"
  )
)
figure_indirect <- ggplot(
  indirect_comparison,
  aes(variable, indirect, fill = model)
) +
  geom_col(position = position_dodge(width = 0.9)) +
  geom_errorbar(
    aes(
      ymin = indirect - 1.96 * standard_error,
      ymax = indirect + 1.96 * standard_error
    ),
    position = position_dodge(width = 0.9), width = 0.25,
    color = "grey25"
  ) +
  geom_hline(yintercept = 0, color = "grey40") +
  scale_fill_manual(values = c(
    "#bdbdbd", "#d7301f", "#3b6ea5",
    "#92c5de"
  ), name = NULL) +
  labs(x = NULL, y = "Косвенный эффект") +
  theme_minimal() +
  theme(legend.position = "bottom")
save_figure(figure_indirect, "impacts_comparison.png")

section("8.5. SDM: слагаемые ряда (I - rho W)^-1 (beta I + theta W)")
# Слагаемые ряда (rho W)^q (beta I + theta W)
series_terms <- t(attr(sdm_impacts$res, "Qres")$total)
colnames(series_terms) <- paste0("q = ", 0:4)
print(round(series_terms, 4))
series_data <- data.frame(
  term = rep(0:4, 4),
  variable = rep(c(
    "Численность населения", "Автотрафик",
    "Плотность сети", "Реальная зарплата"
  ), each = 5),
  effect = c(
    series_terms["log_population", ],
    series_terms["log_car_traffic", ],
    series_terms["pyaterochka_density", ],
    series_terms["log_real_wage", ]
  )
)
figure_series <- ggplot(series_data, aes(term, effect,
  color = variable
)) +
  geom_hline(yintercept = 0, color = "grey60") +
  geom_line() +
  geom_point() +
  labs(
    x = "Номер слагаемого q ряда мультипликатора",
    y = "Вклад в общий эффект", color = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(nrow = 2))
save_figure(figure_series, "impacts_by_order.png")

# ==================================================
# Часть 9. Проверка устойчивости
# ==================================================

key_variables <- c(
  "log_population", "log_car_traffic",
  "grocery_competitors", "pyaterochka_density"
)

# Строка таблицы устойчивости: rho, прямой эффект конкурентов и
# косвенные эффекты ключевых переменных
robustness_row <- function(label, model, effects, sample_size) {
  p_values <- summary(effects, zstats = TRUE, short = TRUE)$pzmat
  rownames(p_values) <- sub(" dy/dx$", "", rownames(p_values))
  result <- data.frame(
    check = label, n = sample_size, rho = model$rho,
    rho_p = parameter_p_value(model, "rho"),
    competitors_direct =
      effects$res$direct["grocery_competitors"],
    competitors_direct_p =
      p_values["grocery_competitors", "Direct"]
  )
  result[paste0(key_variables, "_indirect")] <- as.list(
    effects$res$indirect[key_variables]
  )
  result[paste0(key_variables, "_p")] <- as.list(
    p_values[key_variables, "Indirect"]
  )
  result
}

estimate_sdm <- function(data, weight_list, formula = model_formula) {
  model <- lagsarlm(formula, data, weight_list, Durbin = durbin_formula)
  traces <- trW(as(weight_list, "CsparseMatrix"), m = 30, type = "mult")
  list(
    model = model,
    effects = impacts(model, tr = traces, R = simulations)
  )
}

subsample_weights <- function(filter) {
  subsample_coordinates <- coordinates[filter, ]
  neighbours <- union.nb(
    dnearneigh(subsample_coordinates, 0, 300, longlat = TRUE),
    knn2nb(knearneigh(subsample_coordinates, k = 1, longlat = TRUE))
  )
  distances <- nbdists(neighbours, subsample_coordinates,
    longlat = TRUE
  )
  nb2listw(neighbours,
    glist = lapply(distances, function(distance) {
      exp(-distance / 50)
    }),
    style = "W"
  )
}

without_format <- setdiff(regressors, format_regressors)
formula_without_format <- reformulate(without_format, "log_revenue")
formula_region <- reformulate(c(
  setdiff(regressors, regional_regressors),
  "region"
), "log_revenue")
formula_region_without_format <- reformulate(
  c(setdiff(without_format, regional_regressors), "region"), "log_revenue"
)

section("9.1. Формат магазина и фиксированные эффекты субъектов")
# Эффекты субъектов поглощают региональные цены, доходы и управление
# сетью, а с ними зарплату и безработицу
fit_without_format <- estimate_sdm(
  cross_section, weights,
  formula_without_format
)
fit_region <- estimate_sdm(cross_section, weights, formula_region)
fit_region_without_format <- estimate_sdm(
  cross_section, weights,
  formula_region_without_format
)
print(summary(fit_without_format$effects, zstats = TRUE, short = TRUE))
cat(
  "rho =", fit_without_format$model$rho, " se =",
  fit_without_format$model$rho.se, "\n"
)
print(summary(fit_region$effects, zstats = TRUE, short = TRUE))
cat("rho =", fit_region$model$rho, " se =", fit_region$model$rho.se, "\n")
print(summary(fit_region_without_format$effects,
  zstats = TRUE,
  short = TRUE
))
cat(
  "rho =", fit_region_without_format$model$rho, " se =",
  fit_region_without_format$model$rho.se, "\n"
)
cat("Тест Морана остатков МНК с эффектами субъектов:\n")
print(lm.morantest(lm(formula_region, cross_section), weights))

section("9.2. SDM на разных матрицах весов")
fit_knn <- estimate_sdm(cross_section, weights_knn_8)
fit_inverse <- estimate_sdm(cross_section, weights_inverse_200)
fit_inverse_squared <- estimate_sdm(
  cross_section,
  weights_inverse_squared_400
)
fit_economic <- estimate_sdm(cross_section, weights_economic)

section("9.3. SDM на подвыборках")
at_least_five <- cross_section$store_count >= 5
without_capitals <- !cross_section$settlement_name %in%
  c("Москва г", "Санкт-Петербург г")
rosstat_population <- cross_section$population_source == "rosstat_2024"
without_jumps <- cross_section$stores_with_revenue_jump == 0
fit_five <- estimate_sdm(
  cross_section[at_least_five, ],
  subsample_weights(at_least_five)
)
fit_capitals <- estimate_sdm(
  cross_section[without_capitals, ],
  subsample_weights(without_capitals)
)
fit_rosstat <- estimate_sdm(
  cross_section[rosstat_population, ],
  subsample_weights(rosstat_population)
)
fit_jumps <- estimate_sdm(
  cross_section[without_jumps, ],
  subsample_weights(without_jumps)
)
fit_median <- estimate_sdm(
  cross_section, weights,
  update(
    model_formula,
    log_revenue_median_store ~ .
  )
)
# Без пунктов, у которых нет соседей в радиусе 100 км
connected <- card(band_100) > 0
fit_connected <- estimate_sdm(
  cross_section[connected, ],
  subsample_weights(connected)
)
# Окно ровно 24 месяца, март 2023 - февраль 2025
fit_24_months <- estimate_sdm(
  cross_section, weights,
  update(
    model_formula,
    log_revenue_24_months ~ .
  )
)
# Число магазинов сети в пункте как контроль того, на сколько
# магазинов делится спрос
fit_store_count <- estimate_sdm(
  cross_section, weights,
  update(
    model_formula,
    . ~ . + log_store_count
  )
)
cat("Логарифм числа магазинов в SDM:\n")
print(summary(fit_store_count$model)$Coef["log_store_count", ])

# Состав подвыборки из пунктов с пятью и более магазинами
cat(
  "Медиана численности населения: вся выборка",
  median(exp(cross_section$log_population)), " 5 и более магазинов",
  median(exp(cross_section$log_population[at_least_five])), "\n"
)
cat(
  "Субъектов: вся выборка", nlevels(droplevels(cross_section$region)),
  " 5 и более магазинов",
  nlevels(droplevels(cross_section$region[at_least_five])), "\n"
)

robustness_table <- rbind(
  robustness_row(
    "Основная спецификация", sdm_model, sdm_impacts,
    nrow(cross_section)
  ),
  robustness_row(
    "Без касс и доли больших магазинов",
    fit_without_format$model, fit_without_format$effects,
    nrow(cross_section)
  ),
  robustness_row(
    "Эффекты субъектов", fit_region$model,
    fit_region$effects, nrow(cross_section)
  ),
  robustness_row(
    "Эффекты субъектов, без касс и доли больших",
    fit_region_without_format$model,
    fit_region_without_format$effects, nrow(cross_section)
  ),
  robustness_row(
    "kNN, k = 8", fit_knn$model, fit_knn$effects,
    nrow(cross_section)
  ),
  robustness_row(
    "1/d, 200 км", fit_inverse$model,
    fit_inverse$effects, nrow(cross_section)
  ),
  robustness_row(
    "1/d^2, 400 км", fit_inverse_squared$model,
    fit_inverse_squared$effects, nrow(cross_section)
  ),
  robustness_row(
    "Экономическая матрица", fit_economic$model,
    fit_economic$effects, nrow(cross_section)
  ),
  robustness_row(
    "5 и более магазинов", fit_five$model,
    fit_five$effects, sum(at_least_five)
  ),
  robustness_row(
    "Без Москвы и Санкт-Петербурга", fit_capitals$model,
    fit_capitals$effects, sum(without_capitals)
  ),
  robustness_row(
    "Численность по данным Росстата", fit_rosstat$model,
    fit_rosstat$effects, sum(rosstat_population)
  ),
  robustness_row(
    "Без магазинов со скачками выручки", fit_jumps$model,
    fit_jumps$effects, sum(without_jumps)
  ),
  robustness_row(
    "Медианная выручка магазина", fit_median$model,
    fit_median$effects, nrow(cross_section)
  ),
  robustness_row(
    "Без пунктов без соседей в 100 км", fit_connected$model,
    fit_connected$effects, sum(connected)
  ),
  robustness_row(
    "Окно 24 месяца", fit_24_months$model,
    fit_24_months$effects, nrow(cross_section)
  ),
  robustness_row(
    "С числом магазинов в пункте", fit_store_count$model,
    fit_store_count$effects, nrow(cross_section)
  )
)
print(robustness_table, digits = 3, row.names = FALSE)

# ==================================================
# Часть 10. Устойчивость во времени и пространственная панель
# ==================================================

section("10.1. Индекс Морана и rho SDM по месяцам")
# Для каждого месяца оценивается та же SDM на данных этого месяца
monthly_cross_sections <- split(
  settlement_panel[order(settlement_panel$settlement_id), ],
  settlement_panel$period[order(settlement_panel$settlement_id)]
)
monthly_stability <- do.call(rbind, lapply(
  monthly_cross_sections,
  function(month_data) {
    month_data <- month_data |>
      select(
        settlement_id, period, log_revenue_per_store_real,
        real_wage_rub, unemployment_rate_percent
      ) |>
      left_join(
        cross_section[c("settlement_id", regressors)] |>
          select(-log_real_wage, -unemployment_rate),
        by = "settlement_id"
      ) |>
      mutate(
        log_revenue = log_revenue_per_store_real,
        log_real_wage = log(real_wage_rub),
        unemployment_rate = unemployment_rate_percent
      )
    month_model <- lagsarlm(model_formula, month_data, weights,
      Durbin = durbin_formula
    )
    data.frame(
      period = month_data$period[1],
      moran = unname(moran.test(
        month_data$log_revenue,
        weights
      )$estimate[1]),
      rho = month_model$rho
    )
  }
))
print(monthly_stability, digits = 3, row.names = FALSE)
figure_monthly <- ggplot(monthly_stability, aes(as.Date(period))) +
  geom_line(aes(y = moran, color = "Индекс Морана выручки")) +
  geom_point(aes(y = moran, color = "Индекс Морана выручки")) +
  geom_line(aes(y = rho, color = "Коэффициент rho SDM")) +
  geom_point(aes(y = rho, color = "Коэффициент rho SDM")) +
  scale_color_manual(values = c("#3b6ea5", "#d7301f"), name = NULL) +
  scale_y_continuous(limits = c(0, 0.7)) +
  labs(x = NULL, y = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
save_figure(figure_monthly, "moran_monthly.png")

# Месячная панель: 933 пункта x 26 месяцев = 24 258 наблюдений.
monthly_panel <- settlement_panel |>
  transmute(
    unit = match(settlement_id, cross_section$settlement_id),
    month_index = time_index,
    log_revenue = log_revenue_per_store_real,
    log_real_wage = log(real_wage_rub),
    unemployment_rate = unemployment_rate_percent
  ) |>
  arrange(month_index, unit) |>
  as.data.frame()
# Порядок пунктов внутри месяца совпадает с порядком строк матрицы W
stopifnot(all(monthly_panel$unit[monthly_panel$month_index == 1] ==
  seq_len(nrow(cross_section))))
monthly_frame <- pdata.frame(monthly_panel,
  index = c("unit", "month_index")
)
weights_matrix <- listw2mat(weights)

# Во времени меняются только показатели субъекта, остальное
# поглощают индивидуальные эффекты
panel_formula <- log_revenue ~ log_real_wage + unemployment_rate

section("10.2. Двунаправленные фиксированные эффекты и тест Хаусмана")
panel_fixed_model <- plm(panel_formula, monthly_frame,
  model = "within",
  effect = "twoways"
)
panel_random_model <- plm(panel_formula, monthly_frame,
  model = "random", effect = "twoways"
)
print(summary(panel_fixed_model))
# H0: случайные эффекты состоятельны; H1: состоятельны только FE
print(phtest(panel_fixed_model, panel_random_model))
# H0: нет сериальной корреляции ошибок; H1: она есть
print(pbgtest(panel_fixed_model))
# Стандартные ошибки Дрисколла-Краая, устойчивые к сериальной и
# пространственной корреляции
print(coeftest(panel_fixed_model, vcov = vcovSCC(panel_fixed_model,
  type = "HC1"
)))

section("10.3. CD-тесты Песарана по остаткам модели FE")
# H0: нет зависимости между пунктами; H1: зависимость есть
print(pcdtest(panel_fixed_model, test = "cd"))
print(pcdtest(panel_fixed_model, test = "rho", w = weights_matrix))

section("10.4. LM-тесты пространственной зависимости в модели FE")
print(slmtest(panel_fixed_model, listw = weights, test = "lml"))
print(slmtest(panel_fixed_model, listw = weights, test = "lme"))
print(slmtest(panel_fixed_model, listw = weights, test = "rlml"))
print(slmtest(panel_fixed_model, listw = weights, test = "rlme"))
invisible(gc())

section("10.5. Пространственные модели с двунаправленными FE")
# splm называет коэффициент при Wy "lambda", а при Wu - "rho"
panel_sar_model <- spml(panel_formula, monthly_frame,
  listw = weights,
  model = "within", effect = "twoways", lag = TRUE,
  spatial.error = "none"
)
print(summary(panel_sar_model))
panel_sem_model <- spml(panel_formula, monthly_frame,
  listw = weights,
  model = "within", effect = "twoways", lag = FALSE,
  spatial.error = "b"
)
print(summary(panel_sem_model))
panel_sarar_model <- spml(panel_formula, monthly_frame,
  listw = weights,
  model = "within", effect = "twoways", lag = TRUE,
  spatial.error = "b"
)
print(summary(panel_sarar_model))

# Лаги регрессоров для SDM и SDEM добавляются отдельными столбцами
monthly_frame$lag_log_real_wage <- slag(
  monthly_frame$log_real_wage,
  weights
)
monthly_frame$lag_unemployment_rate <- slag(
  monthly_frame$unemployment_rate, weights
)
panel_durbin_formula <- update(
  panel_formula, . ~ . + lag_log_real_wage + lag_unemployment_rate
)
panel_sdm_model <- spml(panel_durbin_formula, monthly_frame,
  listw = weights, model = "within",
  effect = "twoways", lag = TRUE,
  spatial.error = "none"
)
print(summary(panel_sdm_model))
panel_sdem_model <- spml(panel_durbin_formula, monthly_frame,
  listw = weights, model = "within",
  effect = "twoways", lag = FALSE,
  spatial.error = "b"
)
print(summary(panel_sdem_model))
# LR-тесты внутри семейств: theta = 0 (2 ограничения)
cat(
  "LR SDM FE против SAR FE:",
  2 * (panel_sdm_model$logLik - panel_sar_model$logLik), " p =",
  pchisq(2 * (panel_sdm_model$logLik - panel_sar_model$logLik), 2,
    lower.tail = FALSE
  ), "\n"
)
cat(
  "LR SDEM FE против SEM FE:",
  2 * (panel_sdem_model$logLik - panel_sem_model$logLik), " p =",
  pchisq(2 * (panel_sdem_model$logLik - panel_sem_model$logLik), 2,
    lower.tail = FALSE
  ), "\n"
)

section("10.6. Байесовские апостериорные вероятности моделей")
panel_model_list <- list("sar", "sdm", "sem", "sdem", "slx", "ols")
print(blmpSDPD(panel_formula, monthly_frame,
  W = weights_matrix,
  index = c("unit", "month_index"), model = panel_model_list,
  effect = "twoways", ldet = "mc", dynamic = FALSE,
  prior = "uniform"
)$probs)
print(blmpSDPD(panel_formula, monthly_frame,
  W = weights_matrix,
  index = c("unit", "month_index"), model = panel_model_list,
  effect = "twoways", ldet = "mc", dynamic = TRUE,
  prior = "uniform"
)$probs)

section("10.7. Региональные месячные шоки")
# Эффекты "субъект x месяц" поглощают региональные месячные шоки.
# Индекс Морана остатков показывает, сколько зависимости остается
monthly_panel$region_month <- factor(paste(
  cross_section$shop_region[monthly_panel$unit], monthly_panel$month_index
))
region_month_frame <- pdata.frame(monthly_panel,
  index = c("unit", "month_index")
)
region_month_model <- plm(log_revenue ~ region_month,
  region_month_frame,
  model = "within",
  effect = "individual"
)
observation_key <- paste(monthly_panel$unit, monthly_panel$month_index,
  sep = "-"
)
residuals_in_order <- function(model) {
  as.numeric(residuals(model))[match(
    observation_key,
    names(residuals(model))
  )]
}
fixed_residuals <- residuals_in_order(panel_fixed_model)
region_month_residuals <- residuals_in_order(region_month_model)
month_numbers <- sort(unique(monthly_panel$month_index))
monthly_moran_test <- function(residuals_vector, month_number) {
  moran.test(
    residuals_vector[monthly_panel$month_index == month_number],
    weights
  )
}
monthly_residual_moran <- data.frame(
  month_index = month_numbers,
  twoways = sapply(month_numbers, function(month_number) {
    monthly_moran_test(fixed_residuals, month_number)$estimate[1]
  }),
  region_month = sapply(month_numbers, function(month_number) {
    monthly_moran_test(region_month_residuals, month_number)$estimate[1]
  }),
  region_month_p = sapply(month_numbers, function(month_number) {
    monthly_moran_test(region_month_residuals, month_number)$p.value
  })
)
print(monthly_residual_moran, digits = 3, row.names = FALSE)
# p-значения теста Морана для остатков приближенные, поэтому число
# значимых месяцев дано и с поправкой на 26 проверок
cat(
  "Месяцев со значимым индексом Морана остатков (5%):",
  sum(monthly_residual_moran$region_month_p < 0.05), "из",
  length(month_numbers), " после поправки FDR:",
  sum(p.adjust(monthly_residual_moran$region_month_p, "fdr") < 0.05),
  "\n"
)
print(pcdtest(region_month_model, test = "cd"))
print(pcdtest(region_month_model, test = "rho", w = weights_matrix))

# Средний по месяцам индекс Морана остатков
mean_residual_moran <- function(model) {
  model_residuals <- residuals_in_order(model)
  mean(sapply(month_numbers, function(month_number) {
    monthly_moran_test(model_residuals, month_number)$estimate[1]
  }))
}
cat(
  "Средний индекс Морана остатков: FE", mean_residual_moran(
    panel_fixed_model
  ), " SAR FE", mean_residual_moran(panel_sar_model),
  " SEM FE", mean_residual_moran(panel_sem_model), "\n"
)

# Корреляция остатка пункта с его остатком в прошлом месяце
previous_key <- paste(monthly_panel$unit, monthly_panel$month_index - 1,
  sep = "-"
)
residual_autocorrelation <- function(model) {
  model_residuals <- residuals_in_order(model)
  cor(model_residuals, model_residuals[match(
    previous_key,
    observation_key
  )],
  use = "complete.obs"
  )
}
cat(
  "Автокорреляция остатков первого порядка: FE",
  residual_autocorrelation(panel_fixed_model), " SAR FE",
  residual_autocorrelation(panel_sar_model), " SEM FE",
  residual_autocorrelation(panel_sem_model), "\n"
)

section("10.8. Динамическая панель: МНК, FE и GMM Ареллано-Бонда")
# МНК завышает инерцию, FE при коротком T ее занижает (смещение
# Никелла). GMM берет лаги уровня со второго как инструменты для
# первых разностей, инструменты свернуты. Лаг y строится внутри
# пункта, чтобы ряд не сдвигался через границы пунктов
monthly_panel <- monthly_panel |>
  group_by(unit) |>
  arrange(month_index, .by_group = TRUE) |>
  mutate(log_revenue_previous = dplyr::lag(log_revenue)) |>
  ungroup() |>
  arrange(month_index, unit) |>
  as.data.frame()
lagged_frame <- pdata.frame(monthly_panel,
  index = c("unit", "month_index")
)
dynamic_formula <- log_revenue ~ log_revenue_previous + log_real_wage +
  unemployment_rate
# МНК и FE на тех же месяцах и с теми же эффектами времени, что GMM
comparable_frame <- lagged_frame[
  as.integer(as.character(lagged_frame$month_index)) >= 3,
]
dynamic_pooled_model <- plm(
  update(
    dynamic_formula,
    . ~ . + factor(month_index)
  ),
  comparable_frame,
  model = "pooling"
)
print(coeftest(dynamic_pooled_model,
  vcov = vcovHC(dynamic_pooled_model,
    cluster = "group"
  )
)[2:4, ])
dynamic_fixed_model <- plm(dynamic_formula, comparable_frame,
  model = "within", effect = "twoways"
)
print(coeftest(dynamic_fixed_model,
  vcov = vcovHC(dynamic_fixed_model, cluster = "group")
))
cat(
  "Наблюдений: МНК", nobs(dynamic_pooled_model), " FE",
  nobs(dynamic_fixed_model), "\n"
)
difference_gmm_model <- pgmm(
  log_revenue ~ lag(log_revenue, 1) + log_real_wage + unemployment_rate |
    lag(log_revenue, 2:4),
  data = monthly_frame, effect = "twoways", model = "twosteps",
  transformation = "d", collapse = TRUE
)
print(summary(difference_gmm_model, robust = TRUE))
# Число ограничений J-теста - инструменты минус параметры
cat(
  "Инструментов:", ncol(difference_gmm_model$W[[1]]), " параметров:",
  length(coef(difference_gmm_model)), "\n"
)
system_gmm_model <- pgmm(
  log_revenue ~ lag(log_revenue, 1) + log_real_wage + unemployment_rate |
    lag(log_revenue, 2:4),
  data = monthly_frame, effect = "twoways", model = "twosteps",
  transformation = "ld", collapse = TRUE
)
print(summary(system_gmm_model, robust = TRUE))
# Чувствительность к выбору инструментов: лаги с третьего
print(summary(pgmm(
  log_revenue ~ lag(log_revenue, 1) + log_real_wage + unemployment_rate |
    lag(log_revenue, 3:5),
  data = monthly_frame, effect = "twoways", model = "twosteps",
  transformation = "d", collapse = TRUE
), robust = TRUE))

section("10.9. Динамические пространственные модели с FE")
# Оценка с FE при T = 26 смещена на величину порядка 1/T, поэтому
# включена коррекция смещения Ли-Ю (скорректированные оценки в
# coefficients1, rho1 и std1)
dynamic_sar_model <- SDPDm(panel_formula, monthly_frame,
  W = weights_matrix,
  index = c("unit", "month_index"), model = "sar",
  effect = "twoways", ldet = "mc",
  dynamic = TRUE, LYtrans = TRUE
)
print(summary(dynamic_sar_model))
cat("Исходные оценки без коррекции:\n")
print(c(dynamic_sar_model$coefficients, rho = dynamic_sar_model$rho))
dynamic_sdm_model <- SDPDm(panel_formula, monthly_frame,
  W = weights_matrix,
  index = c("unit", "month_index"), model = "sdm",
  effect = "twoways", ldet = "mc",
  dynamic = TRUE, LYtrans = TRUE
)
print(summary(dynamic_sdm_model))

# Тест Вальда H0: eta + gamma rho = 0 дельта-методом. При этом
# ограничении соседи связаны только через одновременные шоки.
# Порядок в varcov - gamma, eta, регрессоры, rho, дисперсия
dynamic_scale <- dynamic_sar_model$std1[1]^2 /
  dynamic_sar_model$varcov[1, 1]
dynamic_covariance <- dynamic_sar_model$varcov[c(1, 2, 5), c(1, 2, 5)] *
  dynamic_scale
persistence <- dynamic_sar_model$coefficients1[1]
neighbour_lag <- dynamic_sar_model$coefficients1[2]
spatial_lag <- dynamic_sar_model$rho1
restriction_value <- neighbour_lag + persistence * spatial_lag
restriction_gradient <- c(spatial_lag, 1, persistence)
restriction_variance <- as.numeric(t(restriction_gradient) %*%
  dynamic_covariance %*%
  restriction_gradient)
wald_statistic <- restriction_value^2 / restriction_variance
cat(
  "eta + gamma rho =", restriction_value, " se =",
  sqrt(restriction_variance), " Wald =", wald_statistic, " p =",
  pchisq(wald_statistic, 1, lower.tail = FALSE), "\n"
)
rm(weights_matrix)
invisible(gc())

# ==================================================
# Часть 11. Географически взвешенная регрессия
# ==================================================

# GWR: своя регрессия в каждом пункте с гауссовым ядром, ширина окна
# (доля ближайших пунктов) подбирается кросс-валидацией
section("11.1. Ширина окна GWR")
gwr_bandwidth <- gwr.sel(model_formula,
  data = cross_section,
  coords = coordinates, longlat = TRUE,
  adapt = TRUE, gweight = gwr.Gauss,
  method = "cv", verbose = FALSE
)
cat(
  "Доля ближайших пунктов:", round(gwr_bandwidth, 4),
  " число соседей:", round(gwr_bandwidth * nrow(cross_section)), "\n"
)

section("11.2. Локальные коэффициенты GWR")
gwr_model <- gwr(model_formula,
  data = cross_section,
  coords = coordinates, longlat = TRUE,
  adapt = gwr_bandwidth, gweight = gwr.Gauss,
  hatmatrix = TRUE, se.fit = TRUE
)
print(gwr_model)
total_sum_squares <- sum((cross_section$log_revenue -
  mean(cross_section$log_revenue))^2)
cat("Квази-R2 GWR:", round(1 - gwr_model$results$rss /
  total_sum_squares, 3), "\n")

# AICc для сравнения всех моделей, у GWR число параметров - tr(S)
corrected_aic <- function(log_likelihood, parameters, observations) {
  -2 * log_likelihood + 2 * parameters +
    2 * parameters * (parameters + 1) / (observations - parameters - 1)
}
observations <- nrow(cross_section)
gwr_parameters <- gwr_model$results$v1
gwr_log_likelihood <- -observations / 2 *
  (log(2 * pi * gwr_model$results$rss / observations) + 1)
cat(
  "AICc: МНК", corrected_aic(
    as.numeric(logLik(ols_model)),
    attr(logLik(ols_model), "df"),
    observations
  ),
  " SDM", corrected_aic(
    as.numeric(logLik(sdm_model)),
    attr(logLik(sdm_model), "df"), observations
  ),
  " GWR", corrected_aic(
    gwr_log_likelihood, gwr_parameters + 1,
    observations
  ), "\n"
)
# Тест Морана остатков GWR, p-значение приближенное
cat("Тест Морана остатков GWR:\n")
print(moran.test(gwr_model$SDF$gwr.e, weights))
# F-тест Леунга, Ми и Чжана на изменчивость каждого коэффициента
# H0: коэффициент одинаков для всех пунктов
print(LMZ.F3GWR.test(gwr_model))

# Значимость локального коэффициента: t-тест в каждом пункте с поправкой
# Бонферрони 0.05 / n
gwr_significance <- function(variable) {
  coefficient <- gwr_model$SDF[[variable]]
  standard_error <- gwr_model$SDF[[paste0(variable, "_se")]]
  p_value <- 2 * pt(-abs(coefficient / standard_error),
    df = gwr_model$results$edf
  )
  ifelse(p_value < 0.05 / nrow(cross_section), coefficient, NA)
}
significant_gwr <- sapply(regressors, gwr_significance)
gwr_summary <- t(sapply(regressors, function(variable) {
  coefficient <- gwr_model$SDF[[variable]]
  significant_coefficient <- significant_gwr[, variable]
  c(quantile(coefficient, c(0, 0.25, 0.5, 0.75, 1)),
    significant = sum(!is.na(significant_coefficient)),
    negative = sum(significant_coefficient < 0, na.rm = TRUE)
  )
}))
print(round(gwr_summary, 3))

settlement_points$gwr_competitors <- significant_gwr[
  ,
  "grocery_competitors"
]
settlement_points$gwr_car_traffic <- significant_gwr[, "log_car_traffic"]
cat("\nСубъекты с значимым локальным эффектом конкурентов:\n")
print(sort(table(cross_section$shop_region[
  !is.na(settlement_points$gwr_competitors)
]), decreasing = TRUE))
cat("\nСубъекты с значимым локальным эффектом автотрафика:\n")
print(sort(table(cross_section$shop_region[
  !is.na(settlement_points$gwr_car_traffic)
]), decreasing = TRUE))
gwr_map <- function(column, legend_title) {
  ordered_points <- settlement_points[order(
    !is.na(settlement_points[[column]])
  ), ]
  ggplot() +
    map_layer() +
    geom_sf(
      data = ordered_points, aes(color = .data[[column]]),
      size = 0.9
    ) +
    scale_color_viridis_c(
      name = legend_title, na.value = "grey80",
      n.breaks = 4,
      labels = scales::label_number(accuracy = 0.001)
    ) +
    map_frame_coordinates() +
    guides(color = guide_colorbar(barwidth = unit(6, "cm")))
}
save_figure(
  gwr_map("gwr_competitors", "Эффект конкурентов"),
  "map_gwr_competitors.png"
)
save_figure(
  gwr_map("gwr_car_traffic", "Эффект автотрафика"),
  "map_gwr_car_traffic.png"
)
