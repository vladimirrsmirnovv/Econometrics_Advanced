# Установка пакетов
install.packages("strucchange")
install.packages("car")
install.packages("MASS")
install.packages("lmtest")
install.packages("regclass")
install.packages("skedastic")
install.packages("AER")
install.packages("stargazer")
install.packages("xtable")

# Загрузка библиотек

library(strucchange)
library(car)
library(MASS)
library(lmtest)
library(regclass)
library(skedastic)
library(AER)
library(stargazer)
library(xtable)

df = read.csv("D:/Econometrics_data/data_model.csv")
df = df[df$direction=='export',]
df = df[,c('year','importer','quantity','price','gdp_russia','gdp_partner','dist','cpi_partner','cpi_russia','bilateral_er','iip_russia','iip_partner','contig','unfriendly')]


df$covid = ifelse(df$year >2019, 1, 0)
df$svo = ifelse(df$year >2021, 1, 0)

df = na.omit(df)

colnames(df)



#Модель А
formula_a = formula(quantity~price+dist)
mod_a = lm(formula_a, data=df)
summary(mod_a)

#----
# -------------------------------
# Задание 7: графики взаимосвязи quantity с price и dist
# -------------------------------

library(ggplot2)

COL_RED <- "#EA3338"

# Единая тема оформления (уменьшенные подписи)
theme_python <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      axis.title = element_text(size = 11),
      axis.text  = element_text(size = 10),
      panel.grid.major = element_line(color = "grey80", linewidth = 0.4),
      panel.grid.minor = element_line(color = "grey90", linewidth = 0.2)
    )
}

# Функция для построения графика: scatter + линейный тренд
plot_relation <- function(data, x, y = "quantity",
                          title, xlabel,
                          ylabel = "Объём экспорта (quantity)",
                          point_alpha = 0.45,
                          point_size = 2,
                          line_width = 1.2) {
  
  ggplot(data, aes_string(x = x, y = y)) +
    geom_point(color = "black", alpha = point_alpha, size = point_size) +
    geom_smooth(method = "lm", se = FALSE, color = COL_RED, linewidth = line_width) +
    theme_python() +
    labs(title = title, x = xlabel, y = ylabel)
}

# 1) График: quantity ~ price
p_price <- plot_relation(
  data = df,
  x = "price",
  title  = "Зависимость объёма экспорта от цены",
  xlabel = "Цена (price)"
)
print(p_price)

# 2) График: quantity ~ dist
p_dist <- plot_relation(
  data = df,
  x = "dist",
  title  = "Зависимость объёма экспорта от расстояния",
  xlabel = "Расстояние (dist)"
)
print(p_dist)



#----

# в старгейзер

# ----

#вертикальные выбросы и точни расбалансировки
?studres
n = nrow(df)
df[studres(mod_a)>2,] # стьюдентизированные остатки
df[cooks.distance(mod_a)>4/n,] # расстояние кука
df_outl = df
df_outl$studres = studres(mod_a)
df_outl$cook = cooks.distance(mod_a)
df_outl = df_outl[abs((studres(mod_a))>2) | (abs(cooks.distance(mod_a))>4/n),][,c(1, 2, 17, 18)] 
# делаем табличку, которая показывает выбросы

xtable(df_outl) # это для автматической компиляции в латех (сразу код выдаёт)

# ---
xtable(mod_a)
  
# ---

df1 = df[abs((studres(mod_a))<2) | (abs(cooks.distance(mod_a))<4/987),]

#Модель А без выбросов
mod_aa = lm(formula_a, data=df1)
summary(mod_aa)

stargazer(mod_a, mod_aa, title = 'Сводная таблица первых моделей (С выбросами и без)') #вывод моделей в latex

#____

#Тесты Чоу
sctest(formula_a, data=df1[order(df1$unfriendly, decreasing = TRUE), ], #по недружественным
       type = "Chow", point = nrow(df1[df1$unfriendly == 1,]))

sctest(formula_a, data=df1[order(df$contig, decreasing = TRUE), ], #по границам
       type = "Chow", point = nrow(df1[df1$contig == 1,]))

#Тест Рамсея на спецификацию
?resettest
resettest(formula_a, data=df1)

#-----

#для недруж
formula_a1 = formula(quantity~price+dist+contig+unfriendly)
mod_a1 = lm(formula_a1, data=df1)
summary(mod_a1)

#-----

#Чоу в новой спецификации. 
sctest(formula_a1, data=df1[order(df$unfriendly, decreasing = TRUE), ], #недруж
       type = "Chow", point = nrow(df1[df1$unfriendly == 1,]))
sctest(formula_a1, data=df1[order(df$contig, decreasing = TRUE), ], #граница
       type = "Chow", point = nrow(df1[df1$contig == 1,]))

#-----

linearHypothesis(mod_a1, c("unfriendly = 0", "contig = 0")) #f-test на совместную незначимость

#-----quantity~price+dist+contig+unfriendly

#Логарифмируем

# ПОТЕРЯЛАСЬ CONTIG-ДАММИ!!! Далее везде нужно ее вернуть...
# quantity~price+dist+contig+unfriendly

formula_log = formula( log(quantity)~I(log(price+1)) + I(log(dist)) + contig + unfriendly  ) #новая спецификация с логарифмами

mod_a2 = lm(formula_log, data=df1)
summary(mod_a2)

stargazer(mod_a1, mod_a2, title = 'Сводная таблица всех модификаций первой модели')

#-----


#Выбираем лучшее
petest(formula_a1,formula_log, data=df1) #PE-тест для выбора между линейной и нелинейной моделью


#-----

?pe
shapiro.test(mod_a1$residuals)
shapiro.test(mod_a2$residuals)
?replace

#Модель B
df2 = df1
#df2$log_CPI_p <- log(df1$cpi_partner+1)
#df2$log_CPI_r <- log(df1$cpi_r+1)

df2$log_price <- log(df1$price+1)

df2$log_gdp_r <- log(df1$gdp_russia)
df2$log_gdp_p <- log(df1$gdp_partner)

colnames(df2)

formula_b = formula(log(quantity)~I(log(price+1)) + 
                      I(log(price+1)*unfriendly)+
                  
                      I(svo*log(price+1)) + 
                      I(log(dist)) + 
                      unfriendly + 
                      contig +
                      dist + 
                      log_gdp_r + 
                      log_gdp_p + 
                      log_gdp_r+cpi_partner+svo+covid
                    )

mod_b = lm(formula_b, data=df2)
summary(mod_b)
stargazer(mod_a2, mod_b, title = 'Сводная таблица всех модификаций первой модели')

resettest(mod_b)

#мультиколлинеарность
VIF(mod_b)
mod_c <- mod_b

#эндогенность

# РЕГРЕССИЯ С ИНСТРУМЕНТАМИ

df2$log_dist <- log(df2$dist)

formula_b_new = formula(log(quantity)~I(log(price+1)) + 
                      # I(log(price+1)*unfriendly)+
                      
                      # I(svo*log(price+1)) + 
                      # I(log(dist)) + 
                      unfriendly + 
                      contig +
                      # dist + 
                      log_gdp_r + 
                      log_gdp_p + 
                      log_gdp_r+cpi_partner+svo+covid +
                      log_dist
)

colnames(df2)
teend_mod_c <- ivreg(
  formula    = formula_b_new,
  instruments = ~ log_dist +
    unfriendly +
   
   # dist +
    contig +
    log_gdp_r +
    log_gdp_p +
    cpi_partner +
    svo +
    covid +
    cpi_russia +
    iip_russia +
    iip_partner,
  data = df2
)

summary(teend_mod_c, vcov = sandwich, diagnostics = TRUE)
df3 = df2
df3$resid = teend_mod_c$residuals

stargazer(mod_c, teend_mod_c, title = 'Сводная таблица всех модификаций первой модели')


# гетероскедастичность для C

white(mod_c)


#коррекция
?coeftest
coeftest(mod_c, vcov = vcovHC(mod_c, type = "HC0"))
stargazer(mod_c,mod_c, title = 'Сводная таблица всех модификаций первой модели')

