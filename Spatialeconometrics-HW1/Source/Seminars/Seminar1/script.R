library(ggplot2)
library(openxlsx)
library(dplyr)
df <- read.xlsx("new_geo_problem_fantasy.xlsx", sheet = 2)

# task1
ggplot(data = df, aes(x = x, y = y, color = avg_price_per_sqm)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_gradient(low = "green", high = "red", 
                       name = "Средняя цена\n за кв.м") +
  labs(title = "Средняя стоимость квартир за квадратный метр по городам",
       x = "X", 
       y = "Y") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# task2
cor(df$distance_from_leftmost, df$avg_price_per_sqm)

# task3
ggplot(data = df, aes(x = x, y = y, color = avg_price_per_sqm)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_vline(xintercept = 0, color = "red", linewidth = 1.5) +
  scale_color_gradient(low = "green", high = "red", 
                       name = "Средняя цена\n за кв.м") +
  labs(title = "Средняя стоимость квартир за квадратный метр по городам",
       x = "X", 
       y = "Y") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

df_filtered <- df %>% filter(x < 0) 

cor(df_filtered$distance_from_leftmost, df_filtered$avg_price_per_sqm)

# task4
matrix <- as.matrix(read.xlsx("new_geo_problem_fantasy.xlsx", sheet = 3, startRow = 1, colNames = F))
travel_time_from_13 <- as.numeric(matrix["13", ])
travel_time_from_13
cor(travel_time_from_13, df$avg_price_per_sqm)


#########################
"ggplot2", "ggpubr", "ggthemes", "patchwork", # ggplot
"units", "measurements",
"sf", "lwgeom", "s2", # new spatial libraries
"rmapshaper", # simplification
# "sp", "rgdal", "geos", # old spatial libraries
"leaflet", "mapview", "mapdeck", # interactive mapping
"tmap", "mapsf", # static mapping
"spdep", "spatialreg", "rgeoda" # spatial regression

library(ggpubr)
library(ggthemes)
library(sf)
library(lwgeom)
library(s2)
library(spdep)
library(spatialreg)
library(rgeoda)
library(patchwork)
sf_use_s2(TRUE)

ru_reg <- st_read("region_borders_with_data.gpkg")
ru_reg %>% st_geometry() %>% plot()
st_area(ru_reg) %>% sum()
ru_reg_3576 <- ru_reg %>% st_transform(3576)
st_area(ru_reg_3576) %>% sum()
ru_reg_sanfrancisco <- ru_reg %>% st_transform(32610)
st_area(ru_reg_sanfrancisco) %>% sum()

st_area(ru_reg_3576) %>% sum() / st_area(ru_reg) %>% sum()
st_area(ru_reg_sanfrancisco) %>% sum() / st_area(ru_reg) %>% sum()

m1 <- ru_reg %>%
  ggplot(aes(fill = Jan_Temp_Mean_10yr)) +
  geom_sf(cex = 0.1) +
  scale_fill_viridis_c() +
  theme_pubclean() +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        #legend.position = 'none',
        title = element_text(size = 20)) +
  labs(title = "10-Year mean January Temparature")
m1

h1 <- ru_reg %>%
  ggplot(aes(x = Jan_Temp_Mean_10yr, fill = cut(..x.., breaks = seq(min(..x..), max(..x..), by = 1) ))) +
  geom_histogram(binwidth = 2) +
  scale_fill_viridis_d() +
  theme_pubclean() +
  theme(legend.position = 'none', axis.title.y = element_blank(), plot.background = element_blank(), axis.text = element_text(size = 20)) + labs(x = "")

m1+h1
  
OKTMO_NUM <- as.integer(ru_reg$OKTMO_NUM)  
nb <- read.gal(file = "region_borders_with_data.gal", region.id = OKTMO_NUM)
W <- nb2listw(nb, style = "W")
W_mat <- nb2mat(nb)

moran.test(x = ru_reg$Income_per_capita, listw = W)
moran.plot(x = ru_reg$Income_per_capita, listw = W)
lm(W_mat %*% as.matrix(ru_reg$Income_per_capita) ~ Income_per_capita, data = ru_reg)

ru_reg[ru_reg$OKTMO_NUM == 77000000,] %>% select("REG_NAME")
ru_reg[ru_reg$OKTMO_NUM == 45000000,] %>% select("REG_NAME")

moran.test(x = ru_reg$Excess_mortality_apr_feb_per_capita, listw = W)
moran.plot(x = ru_reg$Excess_mortality_apr_feb_per_capita, listw = W)
lm(W_mat %*% as.matrix(ru_reg$Excess_mortality_apr_feb_per_capita) ~ Excess_mortality_apr_feb_per_capita, data = ru_reg)
ru_reg[ru_reg$OKTMO_NUM == 42000000,] %>% select("REG_NAME")

ru_reg$docto
m_slx <- lmSLX(Excess_mortality_apr_feb_per_capita ~ Doctors_per_capita_x1000 +
                 Migr_Inflow_International_3Y_mean_per_capita_x10000 +
                 Migr_Inflow_InterReg_3Y_mean_per_capita_x10000, data = ru_reg,
                 listw = W)
summary(m_slx)

m_sar <- lagsarlm(Excess_mortality_apr_feb_per_capita ~ Doctors_per_capita_x1000 +
                    Migr_Inflow_International_3Y_mean_per_capita_x10000 +
                    Migr_Inflow_InterReg_3Y_mean_per_capita_x10000, data = ru_reg,
                    listw = W)
summary(m_sar)
imp_sar <- impacts(m_sar, listw = W, R = 1000)
summary(imp_sar, zstats = TRUE)

m_sdm <- lagsarlm(Excess_mortality_apr_feb_per_capita ~ Doctors_per_capita_x1000 +
                    Migr_Inflow_International_3Y_mean_per_capita_x10000 +
                    Migr_Inflow_InterReg_3Y_mean_per_capita_x10000, data = ru_reg,
                  listw = W, Durbin = TRUE)
summary(m_sdm)
imp_sdm <- impacts(m_sdm, listw = W, R = 1000)
summary(imp_sdm, zstats = TRUE)
