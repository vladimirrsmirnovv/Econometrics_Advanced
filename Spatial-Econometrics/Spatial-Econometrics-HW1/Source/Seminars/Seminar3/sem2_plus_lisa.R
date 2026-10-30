# installing and loading libraries
install.packages("sf", type = "binary")
install.packages("dplyr", type = "binary")
library(sf)
library(dplyr)

############ working with data ############

# you can find maps (geographical data or polygons) for different countries at https://gadm.org/download_country.html
# we use a geopackage file gadm41_RUS.gpkg
# there are different administrative  layers inside the file
# st_layers() provides info about those layers
st_layers("gadm41_RUS.gpkg")

# regional level is labeled as ADM_ADM_1
# st_read loads the dara
ru_reg <- st_read("gadm41_RUS.gpkg", layer = "ADM_ADM_1")

# .gpkg is basically a dataframe (table), so can work with it using familiar functions
# head() displays first 6 rows of the dataframe
# note that all the info about polygons is stored in the last column "geom"
head(ru_reg)

#  %>%  is a pipe operator
# it applies a function in the right side to the object on the left side
# st_geometry() extracts the data about polygons that can further be plotted with a simple plot()
# note that Chukotka is mapped poorly
ru_reg %>% st_geometry() %>% plot()

################ projections ################

# you can find proper projections for Russia and other countries at https://epsg.io/
# for example, reasonable projections for Russia are coded as 3576 or 5940
# actually all distances an areas depend on the projection used
# you may interact with such inconsistencies at https://www.thetruesize.com/
# st_transform() allows us to project the initial data into a desired coordinates system
ru_reg_3576 <- st_transform(ru_reg, 3576)
ru_reg_5940 <- st_transform(ru_reg, 5940)

# now we have a proper map
ru_reg_3576 %>% st_geometry() %>% plot()

# let's also look at 5940 on its own before comparing all three together below
ru_reg_5940 %>% st_geometry() %>% plot()

# additionally, you can find projections that are not present in R libraries
# for example, we found albers-siberia projection at https://gis-lab.info/qa/gis-lab-projections.html
# it can also be used for Russia
# to use an external projection you need to save the line with its description
crs_albers_siberia <- "+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=105 +x_0=18500000 +y_0=0 +ellps=krass +units=m +towgs84=28,-130,-95,0,0,0,0 +no_defs"
# all other steps are similar
ru_reg_albers_siberia <- st_transform(ru_reg, crs_albers_siberia)
ru_reg_albers_siberia %>% st_geometry() %>% plot()

################ comparing area across projections ################

area_3576 <- ru_reg_3576 %>% st_area() %>% sum()
area_5940 <- ru_reg_5940 %>% st_area() %>% sum()
area_albers <- ru_reg_albers_siberia %>% st_area() %>% sum()
area_albers

area_3576/area_albers
area_5940/area_albers

ru_reg_3171 <- st_transform(ru_reg, 3171)

area_san <- ru_reg_3171 %>% st_area() %>% sum()

area_albers / area_san

################ 2014 regions (with Crimea and Sevastopol) ################

# russia_regions_2014.gpkg merges GADM's Russia admin-1 layer (83 regions) with Crimea and Sevastopol

st_layers("russia_regions_2014.gpkg")

ru_reg_2014 <- st_read("russia_regions_2014.gpkg", layer = "regions_2014")

head(ru_reg_2014)


# from here on, ru_reg_2014 can be projected and used exactly like ru_reg was above
ru_reg_2014_albers <- st_transform(ru_reg_2014, crs_albers_siberia)

ru_reg_2014_albers <- st_make_valid(ru_reg_2014_albers)

ru_reg_2014_albers %>% st_geometry() %>% plot()

################ mapping the added variables ################


install.packages("ggplot2", type = "binary")
install.packages("patchwork", type = "binary")
library(ggplot2)
library(patchwork)

# mortality
map_mortality <- ggplot(ru_reg_2014_albers) +
  geom_sf(aes(fill = Excess_mortality_apr_feb_per_capita), lwd = 0.1, color = "grey40") +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "Excess mortality (Apr-Feb, per capita)")

hist_mortality <- ggplot(ru_reg_2014_albers, aes(x = Excess_mortality_apr_feb_per_capita)) +
  geom_histogram(aes(fill = after_stat(x)), bins = 20, color = "white") +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  labs(x = NULL, y = NULL)

map_mortality + hist_mortality

# mean January temperature
map_temp <- ggplot(ru_reg_2014_albers) +
  geom_sf(aes(fill = Jan_Temp_Mean_10yr), lwd = 0.1, color = "grey40") +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "Mean January temperature")

hist_temp <- ggplot(ru_reg_2014_albers, aes(x = Jan_Temp_Mean_10yr)) +
  geom_histogram(aes(fill = after_stat(x)), bins = 20, color = "white") +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  labs(x = NULL, y = NULL)

map_temp + hist_temp

# urban population share
map_urban <- ggplot(ru_reg_2014_albers) +
  geom_sf(aes(fill = Urban_Population_Share), lwd = 0.1, color = "grey40") +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "Urban population share")

hist_urban <- ggplot(ru_reg_2014_albers, aes(x = Urban_Population_Share)) +
  geom_histogram(aes(fill = after_stat(x)), bins = 20, color = "white") +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  labs(x = NULL, y = NULL)

map_urban + hist_urban

# GRP per capita, 2021
map_grp <- ggplot(ru_reg_2014_albers) +
  geom_sf(aes(fill = grp_per_capita_2021), lwd = 0.1, color = "grey40") +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank()) +
  labs(title = "GRP per capita, 2021")

hist_grp <- ggplot(ru_reg_2014_albers, aes(x = grp_per_capita_2021)) +
  geom_histogram(aes(fill = after_stat(x)), bins = 20, color = "white") +
  scale_fill_viridis_c(name = NULL) +
  theme_minimal() +
  labs(x = NULL, y = NULL)

map_grp + hist_grp

################ spatial weights matrices ################

install.packages("spdep", type = "binary")
library(spdep)

# contiguity ("queen") neighbours, same idea as before, now on the 85-region 2014 dataset
nbr <- poly2nb(ru_reg_2014_albers)
summary(nbr)

# plot the adjacency network on top of the map - the fastest way to see what's disconnected
coord <- ru_reg_2014_albers %>% st_centroid() %>% st_coordinates()
plot(st_geometry(ru_reg_2014_albers), border = "grey70")
plot(nbr, coord, add = TRUE, col = "red", points = FALSE)

# turns out there are 4 disconnected pieces here, not 2: Kaliningrad alone, Sakhalin alone,
# the connected mainland - and Crimea+Sevastopol, which ARE neighbours of each other
# (Sevastopol sits inside Crimea's landmass) but have no land border across the Kerch Strait
# linking them to the rest of Russia
ru_reg_2014_albers$NAME_1[card(nbr) == 0]
nbr[[84]]  # Crimea's only neighbour so far is Sevastopol (85)

# fix Kaliningrad (21) and Sakhalin (61) same as before: SPb/Pskov and Khabarovsk/Primor'ye
# (index 56, not 57 - Pskov and Primor'ye are on opposite ends of the country, easy to mix up)
nbr[[21]] <- as.integer(c(14, 57))    # Kaliningrad -> St. Petersburg, Pskov
nbr[[61]] <- as.integer(c(28, 56))    # Sakhalin -> Khabarovsk, Primor'ye

# Crimea/Sevastopol are a different kind of case from the other two: in reality they ARE
# physically connected to the mainland, via the Kerch Strait Bridge 
nbr[[84]] <- sort(as.integer(c(nbr[[84]], 34)))   # Crimea -> also Krasnodar

# make.sym.nb() automatically adds the reverse link wherever we only edited one side above
# (so SPb, Pskov, Khabarovsk, Primor'ye and Krasnodar all get the edited region added back as
# a neighbour too) - no need to track down and edit every reverse link by hand
nbr <- make.sym.nb(nbr)


# nb2mat() as before - "B" for binary, "W" for row-standardized
W_1 <- nb2mat(nbr, style = "B")
W_2 <- nb2mat(nbr, style = "W")

################ distance-based weights ################

coord <- ru_reg_2014_albers %>% st_centroid() %>% st_coordinates()
coord_alt <- ru_reg_2014_albers %>% st_point_on_surface() %>% st_coordinates()

dist <- as.matrix(dist(coord))
W_3 <- 1 / dist
diag(W_3) <- 0
W_3 <- W_3 / rowSums(W_3)
W_4 <- 1 / (1 + dist)^2
diag(W_4) <- 0
W_4 <- W_4 / rowSums(W_4)

# distance-band neighbours: everyone within 2000 km of each other
nbr_cutoff <- dnearneigh(coord, 0, 1000000)
summary(nbr_cutoff)
W_5 <- nb2mat(nbr_cutoff, style = "W", zero.policy = TRUE)

# k-nearest-neighbours: always exactly k neighbours each, no islands possible
nbr_knn <- knn2nb(knearneigh(coord, k = 3))
W_6 <- nb2mat(nbr_knn, style = "W")

################ economic-distance weights ################

# "spatial" weights don't have to be geographic here, regions with
# similar GRP per capita get a higher weight, regardless of how far apart they actually are
grp <- ru_reg_2014_albers$grp_per_capita_2021
econ_dist <- as.matrix(dist(grp))
W_econ <- 1 / econ_dist
diag(W_econ) <- 0
W_econ <- W_econ / rowSums(W_econ)

# e.g. Moscow City's closest matches here are St. Petersburg, Sakha and Magadan - the last two
# thousands of km away, but economically similar (all high GRP per capita), which a geographic
# W would never pick up
moscow_i <- which(ru_reg_2014_albers$NAME_1 == "Moscow City")
ru_reg_2014_albers$NAME_1[order(W_econ[moscow_i, ], decreasing = TRUE)[1:3]]

################ moran's I ################

# same convention as before: convert each weights matrix into a listw object with mat2listw()
listw_nb <- mat2listw(W_2, style = "W")
listw_dist <- mat2listw(W_4)
listw_econ <- mat2listw(W_econ, style = "W")

set.seed(1)

moran.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_nb )
moran.mc(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_nb, nsim = 999)
moran.plot(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_nb,
           xlab = "mortality", ylab = "W mortality", pch = 16, cex = 0.4, labels = F)

moran.test(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_nb)
moran.mc(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_nb, nsim = 999)
moran.plot(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_nb,
           xlab = "jan temp", ylab = "W jan temp", pch = 16, cex = 0.4, labels = F)

moran.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_nb)
moran.mc(ru_reg_2014_albers$Urban_Population_Share, listw = listw_nb, nsim = 999)
moran.plot(ru_reg_2014_albers$Urban_Population_Share, listw = listw_nb,
           xlab = "urban share", ylab = "W urban share", pch = 16, cex = 0.4, labels = F)

moran.test(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_nb)
moran.mc(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_nb, nsim = 999)
moran.plot(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_nb,
           xlab = "grp", ylab = "Wgrp", pch = 16, cex = 0.4, labels = F)

# --- distance-based weights (distance band, within 2000 km) ---
moran.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_dist, zero.policy = TRUE)
moran.mc(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_dist, nsim = 999, zero.policy = TRUE)
moran.plot(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_dist, zero.policy = TRUE,
           xlab = "mortality", ylab = "W mortality", pch = 16, cex = 0.4, labels = F)

moran.test(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_dist, zero.policy = TRUE)
moran.mc(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_dist, nsim = 999, zero.policy = TRUE)
moran.plot(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_dist, zero.policy = TRUE,
           xlab = "jan temp", ylab = "W jan temp", pch = 16, cex = 0.4, labels = F)

moran.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_dist, zero.policy = TRUE)
moran.mc(ru_reg_2014_albers$Urban_Population_Share, listw = listw_dist, nsim = 999, zero.policy = TRUE)
moran.plot(ru_reg_2014_albers$Urban_Population_Share, listw = listw_dist, zero.policy = TRUE,
           xlab = "urban share", ylab = "W urban share", pch = 16, cex = 0.4, labels = F)

moran.test(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_dist, zero.policy = TRUE)
moran.mc(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_dist, nsim = 999, zero.policy = TRUE)
moran.plot(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_dist, zero.policy = TRUE,
           xlab = "grp", ylab = "Wgrp", pch = 16, cex = 0.4, labels = F)

# --- economic-distance weights (GRP similarity) ---
moran.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_econ)
moran.mc(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_econ, nsim = 999)
moran.plot(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_econ,
           xlab = "mortality", ylab = "W mortality (econ)", pch = 16, cex = 0.4, labels = F)

moran.test(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_econ)
moran.mc(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_econ, nsim = 999)
moran.plot(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_econ,
           xlab = "jan temp", ylab = "W jan temp (econ)", pch = 16, cex = 0.4, labels = F)

moran.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_econ)
moran.mc(ru_reg_2014_albers$Urban_Population_Share, listw = listw_econ, nsim = 999)
moran.plot(ru_reg_2014_albers$Urban_Population_Share, listw = listw_econ,
           xlab = "urban share", ylab = "W urban share (econ)", pch = 16, cex = 0.4, labels = F)

moran.test(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_econ)
moran.mc(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_econ, nsim = 999)
moran.plot(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_econ,
           xlab = "grp", ylab = "Wgrp (econ)", pch = 16, cex = 0.4, labels = F)


################ geary's C ################

geary.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_nb)
geary.mc(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_nb, nsim = 999)

geary.test(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_nb)
geary.mc(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_nb, nsim = 999)

geary.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_nb)
geary.mc(ru_reg_2014_albers$Urban_Population_Share, listw = listw_nb, nsim = 999)

geary.test(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_nb)
geary.mc(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_nb, nsim = 999)

# --- distance-based weights (distance band, within 2000 km) ---
geary.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_dist, zero.policy = TRUE)
geary.mc(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_dist, nsim = 999, zero.policy = TRUE)

geary.test(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_dist, zero.policy = TRUE)
geary.mc(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_dist, nsim = 999, zero.policy = TRUE)

geary.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_dist, zero.policy = TRUE)
geary.mc(ru_reg_2014_albers$Urban_Population_Share, listw = listw_dist, nsim = 999, zero.policy = TRUE)

geary.test(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_dist, zero.policy = TRUE)
geary.mc(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_dist, nsim = 999, zero.policy = TRUE)

# --- economic-distance weights (GRP similarity) ---
geary.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_econ)
geary.mc(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_econ, nsim = 999)

geary.test(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_econ)
geary.mc(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_econ, nsim = 999)

geary.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_econ)
geary.mc(ru_reg_2014_albers$Urban_Population_Share, listw = listw_econ, nsim = 999)

geary.test(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_econ)
geary.mc(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_econ, nsim = 999)

################ getis-ord G ################

globalG.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_nb)
globalG.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_nb)
globalG.test(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_nb)

# --- distance-based weights (distance band, within 2000 km) ---
globalG.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_dist, zero.policy = TRUE)
globalG.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_dist, zero.policy = TRUE)
globalG.test(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_dist, zero.policy = TRUE)

# --- economic-distance weights (GRP similarity) ---
globalG.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_econ)
globalG.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_econ)
globalG.test(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_econ)


################ where do global I/C and G disagree? ################

# actual data
moran.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_econ)
geary.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_econ)
globalG.test(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita, listw = listw_econ)
map_mortality

moran.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_econ)
geary.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_econ)
globalG.test(ru_reg_2014_albers$Urban_Population_Share, listw = listw_econ)
map_urban

#simulated example
set.seed(42)
sim_baseline <- scale(coord[, 2])[, 1] * 10 + rnorm(85, 0, 1.5)   # a smooth south-to-north gradient

# Adygey has exactly one neighbour (Krasnodar) - the smallest possible "local" footprint a shock can
# have, so it only inflates ONE pairwise comparison, even though it still inflates the total
# variance that both I and C divide by
adygey_i <- which(ru_reg_2014_albers$NAME_1 == "Adygey")
sim_shock <- sim_baseline
sim_shock[adygey_i] <- sim_shock[adygey_i] + 1000

moran.test(sim_baseline, listw = listw_nb)
geary.test(sim_baseline, listw = listw_nb)

moran.test(sim_shock, listw = listw_nb)
geary.test(sim_shock, listw = listw_nb)

mp_baseline <- moran.plot(sim_baseline, listw = listw_nb,
                          xlab = "simulated value", ylab = "W simulated value",
                          pch = 16, cex = 0.4, labels = FALSE)
mp_shock <- moran.plot(sim_shock, listw = listw_nb,
                       xlab = "simulated value (with Adygey shock)", ylab = "W simulated value",
                       pch = 16, cex = 0.4, labels = FALSE)

################ LISA: local indicators of spatial association ################

alpha_bonf <- 0.05 / 85

# color scheme for clusters

lisa_colors_moran <- c("High-High" = "#d7301f", "Low-Low" = "#2166ac", "High-Low" = "#fdae61",
                       "Low-High" = "#92c5de", "Not significant" = "grey85")
lisa_colors_geary <- c("High-High" = "#d7301f", "Low-Low" = "#2166ac", "Other Positive" = "#fdae61",
                       "Negative" = "#92c5de", "Not significant" = "grey85")
lisa_colors_g <- c("High" = "#d7301f", "Low" = "#2166ac", "Not significant" = "grey85")

lisa_map_theme <- theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank())

set.seed(1)


(lm_temp_analytic <- localmoran(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_nb))

cluster_temp_analytic <- as.character(attr(lm_temp_analytic, "quadr")$mean)
cluster_temp_analytic[lm_temp_analytic[, "Pr(z != E(Ii))"] >= alpha_bonf] <- "Not significant"
ru_reg_2014_albers$cluster_temp_analytic <- cluster_temp_analytic

map_temp_analytic <- ggplot(ru_reg_2014_albers) +
  geom_sf(aes(fill = cluster_temp_analytic), lwd = 0.1, color = "grey40") +
  scale_fill_manual(values = lisa_colors_moran, name = NULL) +
  lisa_map_theme + labs(title = "Local Moran's I - temperature (closed-form)")
map_temp_analytic

(lm_temp_perm <- localmoran_perm(ru_reg_2014_albers$Jan_Temp_Mean_10yr, listw = listw_nb, nsim = 9999))

cluster_temp_perm <- as.character(attr(lm_temp_perm, "quadr")$mean)
cluster_temp_perm[lm_temp_perm[, "Pr(folded) Sim"] >= alpha_bonf] <- "Not significant"
ru_reg_2014_albers$cluster_temp_perm <- cluster_temp_perm

map_temp_perm <- ggplot(ru_reg_2014_albers) +
  geom_sf(aes(fill = cluster_temp_perm), lwd = 0.1, color = "grey40") +
  scale_fill_manual(values = lisa_colors_moran, name = NULL) +
  lisa_map_theme + labs(title = "Local Moran's I - temperature (permutation)")
map_temp_perm

map_temp_analytic + map_temp_perm






(lm_mortality_perm <- localmoran_perm(ru_reg_2014_albers$Excess_mortality_apr_feb_per_capita,
                                      listw = listw_dist, nsim = 9999, zero.policy = TRUE))

cluster_mortality_perm <- as.character(attr(lm_mortality_perm, "quadr")$mean)
cluster_mortality_perm[lm_mortality_perm[, "Pr(folded) Sim"] >= alpha_bonf] <- "Not significant"
ru_reg_2014_albers$cluster_mortality_perm <- cluster_mortality_perm

map_mortality_lisa <- ggplot(ru_reg_2014_albers) +
  geom_sf(aes(fill = cluster_mortality_perm), lwd = 0.1, color = "grey40") +
  scale_fill_manual(values = lisa_colors_moran, name = NULL) +
  lisa_map_theme + labs(title = "Local Moran's I - mortality (distance-band)")
map_mortality_lisa

(lc_grp_perm <- localC_perm(ru_reg_2014_albers$grp_per_capita_2021, listw = listw_econ, nsim = 9999))

cluster_grp_geary <- as.character(attr(lc_grp_perm, "cluster"))
cluster_grp_geary[attr(lc_grp_perm, "pseudo-p")[, "Pr(folded) Sim"] >= alpha_bonf] <- "Not significant"
ru_reg_2014_albers$cluster_grp_geary <- cluster_grp_geary

map_grp_geary <- ggplot(ru_reg_2014_albers) +
  geom_sf(aes(fill = cluster_grp_geary), lwd = 0.1, color = "grey40") +
  scale_fill_manual(values = lisa_colors_geary, name = NULL) +
  lisa_map_theme + labs(title = "Local Geary's C - GRP per capita (economic-distance)")
map_grp_geary

(lg_urban_perm <- localG_perm(ru_reg_2014_albers$Urban_Population_Share, listw = listw_nb, nsim = 9999))

cluster_urban_g <- as.character(attr(lg_urban_perm, "cluster"))
cluster_urban_g[attr(lg_urban_perm, "internals")[, "Pr(folded) Sim"] >= alpha_bonf] <- "Not significant"
ru_reg_2014_albers$cluster_urban_g <- cluster_urban_g

map_urban_g <- ggplot(ru_reg_2014_albers) +
  geom_sf(aes(fill = cluster_urban_g), lwd = 0.1, color = "grey40") +
  scale_fill_manual(values = lisa_colors_g, name = NULL) +
  lisa_map_theme + labs(title = "Local Getis-Ord G - urban share (contiguity)")
map_urban_g


