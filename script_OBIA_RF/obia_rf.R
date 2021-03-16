library(sf)
library(caret)
library(ranger)
library(dplyr)
library(vip)
library(dplyr)
## joint gmk_Obia and obia rename 
    
allsh = st_read("/Volumes/Meng_Mac/obia/temp/joind_within.shp")
allsh$GEOCODE2 = as.character(allsh$GEOCODE2)
 
adf = allsh %>%dplyr:: mutate(group_M =  case_when(startsWith(GEOCODE2, "O") ~ "O", 
                                                   startsWith(GEOCODE2, "P1b") ~ "P1b",
                                startsWith(GEOCODE2, "P2b") ~ "P2b",
                          
                                startsWith(GEOCODE2, "P2c") ~ "P2c", 
                                startsWith(GEOCODE2, "P2a") ~ "P2a", 
                                startsWith(GEOCODE2, "S1a") ~ "S1a",
                                startsWith(GEOCODE2, "S1c") ~ "S1c",
                                startsWith(GEOCODE2, "S2") ~ "S2", 
                                startsWith(GEOCODE2, "P1a1") ~ "P1a1",
                                startsWith(GEOCODE2, "P1a2") ~ "P1a2", 
                                startsWith(GEOCODE2, "S3") ~ "S3",
                                startsWith(GEOCODE2, "P2d") ~ "P2d",
                                startsWith(GEOCODE2, "H1") ~ "H1",
                                startsWith(GEOCODE2, "H2") ~ "H2",
                                startsWith(GEOCODE2, "K") ~ "K",
                                startsWith(GEOCODE2, "D") ~ "D",))
st_write(adf, "/Volumes/Meng_Mac/obia/GMKOBIA.gpkg")

############# joint gmk and obia rename 
#all = st_read("/Volumes/Meng_mac/obia/joind.shp")
#all = st_read("~/Downloads/Elisabeth/obia/joind_within.shp")
#names(all)
#head(all%>%select("GEOCODE2", "OMS_GEOCOD"))

a1 = all%>%filter(!OMS_GEOCOD %in% c('Plaat/slik met water (bodem onzichtbaar)',
                               'Overig waterberging','Overig getijdenhaven','Overig zanddam')) 
 
a1$OMS_GEOCOD = as.character(a1$OMS_GEOCOD)
adf = a1 %>% mutate(group_M =
                     case_when(startsWith(OMS_GEOCOD, "Begroeid schor/strand (gesloten, > 50 % bedekking) natuurlijke (kwelder)") ~ "S1a", 
                               startsWith(OMS_GEOCOD, "Overig plateau/verhoging (antropogeen)") ~ "O", 
                               startsWith(OMS_GEOCOD, "Overig wegen/pade") ~ "O", 
                               startsWith(OMS_GEOCOD, "Begroeid schor/strand (gesloten, > 50 % bedekking) open plek in het kwelder") ~ "S3", 
                               startsWith(OMS_GEOCOD, "Begroeid schor/strand (zeer open, 2-10% bedekking en/of pollenstructuur") ~ "S2", 
                               startsWith(OMS_GEOCOD, "Begroeid schor/strand (open, 10-50% bedekking)") ~ "S2",
                               startsWith(OMS_GEOCOD, "Natuurlijk meanderende kreek (5-250m breed, onbegroeid) op schor/kwelder en groen strand") ~ "S1c",
                               startsWith(OMS_GEOCOD, "Laag energetische vlakke plaat, slibrijk zand") ~ "P1a2",
                               startsWith(OMS_GEOCOD, "Laag energetische vlakke plaat, zand") ~ "P1a1",
                               
                               startsWith(OMS_GEOCOD, "Hoog energetische vlakke plaat") ~ "P2c",
                               startsWith(OMS_GEOCOD, "Geisoleerde") ~ "P2c",
                               startsWith(OMS_GEOCOD,"Hoog energetische plaat") ~ "P2b",
                               startsWith(OMS_GEOCOD,"Hard substraat harde") ~ "H1",
                               startsWith(OMS_GEOCOD,"Hard substraat antropogeen") ~ "H2" ) )

adf = na.omit(adf) 
st_write(adf, "~/Downloads/Elisabeth/reclassified_ml_within.shp")

adf = st_read("/Volumes/Meng_mac/obia/reclasified_ml/reclassified_ml.shp")
table(adf$group_M) 
#H1    H2     O (1) P1a1 (4)  P1a2 (6)   P2b   P2c   S1a (0)   S1c    S2    S3 
#1936  4953  1159   20327     14471      21768 15742 71556   14783  8814  3422 

nrow(adf) #181098 
unique(adf$group_M)
names(adf)
nrow(adf)
 

#adf= adf%>%group_by(group_M) #178931
#adf$group_M                                  
#st_geometry(training) <- NULL

#a =st_read("~/Downloads/Elisabeth/ObjectShapes/Westerschelde_2016_obj_creation.v2.shp")
#a =st_read("~/Downloads/Elisabeth/ObjectShapes/Westerschelde_2016_obj_creation.v2.shp")
#a2 =st_read("~/Downloads/Elisabeth/ObjectShapes_2016_3_3/Westerschelde_2016_obj_creation.v2.shp")
#a3 =rbind(a,a2)
#cla =st_read("~/Downloads/Elisabeth/2016_GMK/e_GMK_Westerschelde2016.shp")
#cla_f = st_join(cla, a3, left =F)
#nrow(cla_f)

#cla_s= cla_f %>%select(matches("GLCM"), 
#                OMS_GEOCOD,
#                matches("Rel"), matches("Number_of_"), Shape_inde)
#                 Brightness,Compactnes,
#Shape_Leng Shape_Area
#st_geometry(cla_s) <- NULL

adf= adf %>%
     select(matches('density|Dif_|EF|GLCM|LW|max_SI|mean_|mn_|num|obj_|SI_|std|tot'), group_M) %>% 
    filter_if(~is.numeric(.), all_vars(!is.infinite(.)))%>%  
    filter_if(~is.numeric(.), all_vars(.<100000))
adf
summary(adf)
training <- sample(1:nrow(adf),nrow(adf)*0.9)                             #Craete training dataset containing about 90 % of the total observations
testing <- setdiff(1:nrow(adf),training)                         #Create testing dataset containing the remaining 10%
training <- adf[training,]
testing <- adf[testing ,]
st_geometry(training) <- NULL
 
#training = droplevels(training)

#                matches("Rel"), matches("Number_of_"), Shape_inde)
#                 Brightness,Compactnes,
r1=ranger(group_M~., data = training, importance = "impurity", num.trees = 1000)
r1
geomtest= st_geometry(testing) 
st_geometry(testing) = NULL
#testing = droplevels(testing)

st_geometry(testing) = geomtest
 
 
importance(r1)%>%names
#barplot(sort(importance(r1), decreasing = F), horiz = T, las  =1 )
vip (r1,width = 0.2, num_features = 20, geom = "point", aesthetics = list(fill = "green3", shape = 17, size = 4))
#sort(importance(r1), decreasing = T)
#droplevels.sfc = function(x, except, exclude, ...) 
#testing =   droplevels.sfc(testing)  
pred = predict(r1, data = data.frame(testing))
 

image(table(testing$group_M, predictions(pred)))

#table(testing$OMS_GEOCOD, predictions(pred))
cm = confusionMatrix(data = predictions(pred), reference = as.factor(testing$group_M), mode = "prec_recall")
r1
cm$overall
plot(predictions(pred))
testing$pred = predictions(pred)
#cm$byClass
# library(cowplot)
#libary(ggplot)
 
library(mapview)
library(RColorBrewer)
mapviewOptions(
  basemaps = c("OpenStreetMap.Mapnik","Esri.OceanBasemap")
  , raster.palette = colorRampPalette(brewer.pal(9, "Set2"))
  , vector.palette = colorRampPalette(brewer.pal(9, "Set2"))
  , na.color = "gray"
  , layers.control.pos = "topleft"
  , viewer.suppress = TRUE # open browser
)

test_simple = testing%>%select(pred, group_M)
st_write(test_simple, "test_simple.shp")
st_write(testing, "testresult.shp")
#a = mapview(testing[1:100,"pred"])+ mapview(testing[1:100,"group_M"], zcol = "group_M")
#mapshot(a , url = "~/Downloads/OBRF_wtiles.html")   
#pa = ggplot() + 
#  geom_sf(data =testing , aes(fill =  pred))
#pb = ggplot() + 
#  geom_sf(data =testing , aes(fill =OMS_GEOCOD))
#plot_grid(pa, pb, nrow = 2, rel_widths = c(2.3, 1))


#plot(testing["pred"])

#plot(testing["OMS_GEOCOD"], key.pos =4,)


