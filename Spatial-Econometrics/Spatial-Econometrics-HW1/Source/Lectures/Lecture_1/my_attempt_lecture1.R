# install.packages(c("sf", "spdep"))

library(sf)
library(spdep)

bbox <- st_bbox(
  c(
    xmin=0,
    ymin=0,
    xmax=3,
    ymax=2
  )
)

grid <- st_make_grid(
  st_as_sfc(bbox),
  n = c(3, 2)
)

city <- st_sf(
  district = c("D", "E", "F", "A", "B", "C"),
  unemployment = c(20, 22, 24, 10, 12, 14),
  geometry = grid
)

city

plot(
  city["unemployment"],
  main = "Unemployment"
)

nb_rook <- poly2nb(
  city,
  queen = FALSE
)

city$district[nb_rook[[1]]]

nb_queen <- poly2nb(
  city,
  queen = TRUE
)

coords <- st_coordinates(st_centroid(city))
plot(st_geometry(city))
plot(
  nb_rook,
  coords,
  add = TRUE
)

plot(
  nb_queen,
  coords,
  add = TRUE
)

W_binary <- nb2mat(
  nb_rook,
  style = "B"
)

W <- nb2mat(
  nb_rook,
  style = "W"
)

rowSums(W)

y <- city$unemployment
spatial_lag <- W %*% y

listw <- nb2listw(
  nb_rook,
  style = "W"
)

city$lag_unemployment <- lag.listw(
  listw,
  city$unemployment
)

city$income <- c(
  30, 32, 35,
  50, 55, 60
)

model_ols <- lm(
  unemployment ~ income,
  data = city
)

summary(model_ols)


library(spatialreg)

sar <- lagsarlm(
  unemployment ~ income,
  data = city,
  listw = listw
)

summary(sar)
