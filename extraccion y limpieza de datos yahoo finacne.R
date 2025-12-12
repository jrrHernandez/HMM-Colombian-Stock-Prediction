#creacion de fracciones desde yahoofinance
rm(list=ls())
require(pacman)
p_load(dplyr, rio, quantmod, tibble, openxlsx)
data <-  getSymbols(Symbols = "TERPEL.CL", auto.assign = FALSE, from = "2022-01-2", to= "2025-10-28")
data <- data %>% as.data.frame() %>% rename(
 px_last= "TERPEL.CL.Open", px_high = "TERPEL.CL.High",  px_low= "TERPEL.CL.Low" ,    px_close_1d=  "TERPEL.CL.Close" , volume=   "TERPEL.CL.Volume" , aja=  "TERPEL.CL.Adjusted"
)

dataterpel <- rownames_to_column(data, var = "date")
dataterpel$date <- as.Date(dataterpel$date)

export(dataterpel, "dataTERPEL.csv")

var <- c( "px_last" , "px_high" ,    "px_low"   ,   "px_close_1d")
logreturns_df_cibest <- data.frame(Date = dateCibest)
dateCibest <- datacibest$date[-1]
for (nombre_columna in var) {
  
  # Calcula el rendimiento logarítmico: diff(log(Precios))
  # Se usa $ para acceder a la columna por nombre en R
  rendimiento_log <- diff(log(datacibest[[nombre_columna]]))
  
  # Agrega la nueva columna al data frame de resultados
  # La nueva columna se llamará como la original con el prefijo "logret_"
  logreturns_df_cibest[[paste0("logret_", nombre_columna)]] <- rendimiento_log
  
}

logreturns_df_cibest <- logreturns_df_cibest %>% 
  rename("px_last" = logret_px_last,
         "px_high" = logret_px_high,
         "px_low" = logret_px_low,
         "px_close_1d" = logret_px_close_1d, 
         date=Date)
setwd("C:/Users/richa/OneDrive - Universidad de los Andes/Universidad/Maestria/MIS/proyecto")
export(logreturns_df_cibest, "CibestLogReturns.csv")
