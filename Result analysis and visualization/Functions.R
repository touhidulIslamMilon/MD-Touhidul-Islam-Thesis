processYouGov <- function(file, sheet)
{
  
  widedata <- read_excel(file, sheet)
  
  names(widedata)[1] <- "mood"
  N <- ncol(widedata)
  
  dateStrings <- names(widedata)[2:N]
  #  dateStrings <- sapply(strsplit(dateStrings, " - ", fixed = T), function(x) {x[2]})
  dateStrings <- gsub("January", "01", dateStrings)
  dateStrings <- gsub("Jan", "01", dateStrings)
  dateStrings <- gsub("February", "02", dateStrings)
  dateStrings <- gsub("Feb", "02", dateStrings)
  dateStrings <- gsub("March", "03", dateStrings)
  dateStrings <- gsub("Mar", "03", dateStrings)
  dateStrings <- gsub("April", "04", dateStrings)
  dateStrings <- gsub("Apr", "04", dateStrings)
  dateStrings <- gsub("May", "05", dateStrings)
  dateStrings <- gsub("June", "06", dateStrings)
  dateStrings <- gsub("Jun", "06", dateStrings)
  dateStrings <- gsub("July", "07", dateStrings)
  dateStrings <- gsub("Jul", "07", dateStrings)
  dateStrings <- gsub("August", "08", dateStrings)
  dateStrings <- gsub("Aug", "08", dateStrings)
  dateStrings <- gsub("September", "09", dateStrings)
  dateStrings <- gsub("Sep", "09", dateStrings)
  dateStrings <- gsub("October", "10", dateStrings)
  dateStrings <- gsub("Oct", "10", dateStrings)
  dateStrings <- gsub("November", "11", dateStrings)
  dateStrings <- gsub("Nov", "11", dateStrings)
  dateStrings <- gsub("December", "12", dateStrings)
  dateStrings <- gsub("Dec", "12", dateStrings)
  
  dateStrings <- gsub("th", "", dateStrings)
  dateStrings <- gsub("st", "", dateStrings)
  dateStrings <- gsub("nd", "", dateStrings)
  dateStrings <- gsub("rd", "", dateStrings)
  dateStrings <- gsub(" ", "-", dateStrings, fixed = T)
  dateStrings <- as.character(as.Date(as.POSIXlt(dateStrings, format = "%d-%m-%Y")))
  
  names(widedata)[2:N] <- dateStrings
  
  longdf <- NULL
  for (ds in dateStrings)
  {
    v <- as.character(widedata[[ds]])
    v <- as.numeric(gsub("%","", v))
    longdf <- rbind(longdf, data.frame(date=ds, measure=widedata$mood, count=v))
  }
  
  longdf %>% filter(measure=="Happy") %>% select(date, count) -> happydf
  longdf %>% filter(measure=="Energetic") %>% select(date, count) -> energeticdf
  longdf %>% filter(measure=="Inspired") %>% select(date, count) -> inspireddf
  longdf %>% filter(measure=="Optimistic") %>% select(date, count) -> optimisticdf
  longdf %>% filter(measure=="Content") %>% select(date, count) -> contentdf
  longdf %>% filter(measure=="Sad") %>% select(date, count) -> saddf
  longdf %>% filter(measure=="Scared") %>% select(date, count) -> scareddf
  longdf %>% filter(measure=="Frustrated") %>% select(date, count) -> frustrateddf
  longdf %>% filter(measure=="Stressed") %>% select(date, count) -> stresseddf
  longdf %>% filter(measure=="Lonely") %>% select(date, count) -> lonelydf
  longdf %>% filter(measure=="Bored") %>% select(date, count) -> boreddf
  longdf %>% filter(measure=="Apathetic") %>% select(date, count) -> apatheticdf
  
  ygdf <- data.frame(date=happydf$date, happy=happydf$count, sad=saddf$count, 
                     energetic=energeticdf$count, apathetic=apatheticdf$count,
                     scared=scareddf$count, frustrated=frustrateddf$count,
                     inspired=inspireddf$count, 
                     content=contentdf$count, optimistic=optimisticdf$count,
                     stressed=stresseddf$count, lonely=lonelydf$count, bored=boreddf$count)
  return(ygdf)
  
}
plotTS2 <- function(df1, df2, var1, var2, var3, col1, col2, col3, label1, label2, label3, title="Time series comparison", ylab="measurement", ymin=NA, ymax=NA)
{
  w1 <- as.numeric(df1[[var1]])
  w2 <- as.numeric(df2[[var2]])
  w3 <- as.numeric(df2[[var3]])
  
  # Ensure 'day' columns are in Date format
  df1$day <- as.Date(df1$day)
  df2$day <- as.Date(df2$day)
  
  # Create data frames for each series
  ddf1 <- data.frame(day=df1$day, y1=w1)
  ddf2 <- data.frame(day=df2$day, y2=w2, y3=w3)
  
  # Merge data frames on the 'day' column
  df <- merge(ddf1, ddf2, by="day", all=FALSE)  # all=FALSE ensures only matching rows
  
  # Reshape data for plotting
  df <- data.frame(day=rep(df$day, 3),
                   y=c(df$y1, df$y2, df$y3),
                   dataset=c(rep(label1, nrow(df)), rep(label2, nrow(df)), rep(label3, nrow(df))))
  #print(df)
  # Scale the values for each dataset
  df$y[df$dataset==label1] <- scale(df$y[df$dataset==label1])
  df$y[df$dataset==label2] <- scale(df$y[df$dataset==label2])
  df$y[df$dataset==label3] <- scale(df$y[df$dataset==label3])
  print(df)
  # Create the plot
  plt <- ggplot(df, aes(x=as.Date(day), y=y, group=dataset, color=dataset)) + 
    geom_line() + 
    theme_bw() + 
    theme(legend.position="bottom") +
    scale_color_manual(values=c(col1, col2, col3)) + 
    xlab("Date") + 
    ylab(ylab) + 
    ggtitle(title) + 
    geom_vline(xintercept = as.Date("2023-03-16"), color=rgb(0,0,0,0.5))
  
  # Apply y-axis limits if specified
  if (!is.na(ymin) & !is.na(ymax)) {
    plt <- plt + ylim(ymin, ymax)
  }
  
  return(plt)
}
plotTS3 <- function(df, var1, var2, var3, var4, 
                    col1, col2, col3, col4, 
                    label1, label2, label3, label4, 
                    title = "Time series comparison", 
                    ylab = "measurement", ymin = NA, ymax = NA, 
                    plot_name = "No name") {
  # Extract and scale variables
  w1 <- scale(as.numeric(df[[var1]]))
  w2 <- scale(as.numeric(df[[var2]]))
  w3 <- scale(as.numeric(df[[var3]]))
  w4 <- scale(as.numeric(df[[var4]]))
  
  # Reshape data for plotting
  plot_df <- data.frame(
    day = rep(as.Date(df$day), 4),
    y = c(w1, w2, w3, w4),
    dataset = c(rep(label1, nrow(df)), 
                rep(label2, nrow(df)), 
                rep(label3, nrow(df)), 
                rep(label4, nrow(df)))
  )
  
  # Create the plot
  plt <- ggplot(plot_df, aes(x = day, y = y, group = dataset, color = dataset)) + 
    geom_line() + 
    theme_bw() + 
    theme(legend.position = "bottom") +
    scale_color_manual(values = c(col1, col2, col3, col4)) + 
    xlab("Date") + 
    ylab(ylab) + 
    ggtitle(title) + 
    geom_vline(xintercept = as.Date("2023-03-16"), color = rgb(0, 0, 0, 0.5))
  
  # Apply y-axis limits if specified
  if (!is.na(ymin) & !is.na(ymax)) {
    plt <- plt + ylim(ymin, ymax)
  }
  
  # Save plot to file
  plot_name <- paste0("plot/", plot_name, ".jpg")
  ggsave(plot_name, plot = plt, width = 9, height = 5, units = "in")
  
  return(plt)
}
plotTS <- function(df1, df2, var1, var2, col1, col2, label1, label2, title="Time series comparison", ylab="measurement", ymin=NA, ymax=NA,plot_name)
{
  w1 <- as.numeric(df1[[var1]])
  w2 <- as.numeric(df2[[var2]])

  ddf1 <- data.frame(day=as.Date(df1$day), y1=w1)
  ddf2 <- data.frame(day=as.Date(df2$day), y2=w2)
  df <- inner_join(ddf1, ddf2, by="day")
  df <- data.frame(day=rep(df$day,2), y=c(df$y1,df$y2), dataset=c(rep(label1, nrow(df)), rep(label2, nrow(df))))
  
  df$y[df$dataset==label1] <- scale(df$y[df$dataset==label1])
  df$y[df$dataset==label2] <- scale(df$y[df$dataset==label2])
  
  plt <- ggplot(df, aes(x=as.Date(day), y=y, group=dataset, color=dataset)) + geom_line() + 
    theme_bw() + theme(legend.position="bottom") +
    scale_color_manual(values=c(col1, col2))+ xlab("Date") + ylab(ylab)  + ggtitle(title) + geom_vline(xintercept = as.Date("2020-11-01"), color=rgb(0,0,0,0.5))
  plot_name = paste("plot/",plot_name,".jpg")
  if (!is.na(ymin) & ! is.na(ymax))
  {
    plt <- plt + ylim(ymin, ymax)
    ggsave(plot_name, plot = plt,width = 9, height = 5, units = "in")
  }
  return(plt)
}


dcca.test <- function(ts1, ts2, nu=1, m=12, R=10000)
{
  rnd <- NULL
  for (i in seq(1,R))
  {
    rnd <- rbind(rnd, rhodcca(sample(ts1), ts2, nu=nu, m=m)$rhodcca)
  }
  
  return(rnd)
}

permcor <- function(X,Y, R=10000)
{
  rnd <- NULL
  for (i in seq(1,R))
  {
    rnd <- rbind(rnd,cor(sample(X),Y))
  }
  
  return(rnd)
  
}
getfilename <- function(epoch, query_type, similarity){
  filename= paste("Result/",similarity,query_type,"_quary_epoch",epoch,".xlsx",sep = "")
  print(filename)
  return(filename)
}
analyses <- function(Twittersel, Yougovdf, R,label)
{
  
  f <- as.Date(Twittersel$day) < as.Date("2023-01-05")
  X <- Twittersel$x[f]
  Y <- Yougovdf$y[f]
 
  print("----normal Corelation----")
  print("Correlation up to Mar 16st 2023")
  ct <- cor.test(X,Y)
  print(ct)
  
  
  
  print("------Permutation COrelation--- ")
  rndR <- permcor(X,Y,R) # do a permutation test 
  pperm <- (sum(rndR>=cor(X,Y))+1)/R 
  print("Corr permation test p-val:")
  print(pperm)
  results <- data.frame(rh = ct$estimate, rhlow = ct$conf.int[1], rhhigh=ct$conf.int[2], rhp = ct$p.value, rhpperm = pperm)  
  
  print("DCCA:")
  rho <- rhodcca(X, Y, nu=1, m=12)$rhodcca
  rndR <- dcca.test(X, Y, nu=1, m=12, R=R)
  print(rho)
  print("DCCA p-val:")
  pval <- (sum(rndR>=rho)+1)/R
  print(pval)

  results <- cbind(results, data.frame(dcca=rho, dccap=pval))  
  
  print("----Another permutaion---")
  print("Prediction correlation")
  f <- as.Date(Twittersel$day) >= as.Date("2023-01-05")
  X <- Twittersel$x[f]
  Y <- Yougovdf$y[f]
  ct <- cor.test(X,Y)
  print(ct)
  rndR <- permcor(X,Y,R) # do a permutation test 
  pperm <- (sum(rndR>=cor(X,Y))+1)/R 
  print(pperm)
  
  results <- cbind(results, data.frame(rp = ct$estimate, rplow = ct$conf.int[1], rphigh=ct$conf.int[2], rpp = ct$p.value, rppperm = pperm))  

  f <- as.Date(Twittersel$day) < as.Date("2023-01-05")
  X <- Twittersel$x[f]
  Y <- Yougovdf$y[f]
  
  print("Lagged model:")
  Y <- scale(Y)
  X <- scale(X)
  model <- lm(Y ~ X + lag(Y,1))
  ct <- coeftest(model, vcov=vcovHAC(model))
  print(ct)
  kpt <- kpss.test(model$residuals)
  print(kpt)
  results <- cbind(results, data.frame(beta=ct[2,1], betap = ct[2,3], KSp=kpt$p.value))
  rownames(results) <- NULL
  filename= paste("output/",label,'.txt')
  msescale<-mean((X - Y)^2)
  
  f <- as.Date(Twittersel$day) < as.Date("2023-01-05")
  X <- Twittersel$x[f]
  Y <- Yougovdf$y[f]
  mses<-mean((X - Y)^2)
  print("Mean squire error: ")
  print("Normal")
  print(mses)
  print("scaled: ")
  print(msescale)
  print(filename)
  
  # Fit a linear model
  model <- lm(X ~ Y)
  
  # Extract R^2 from the summary
  model_summary <- summary(model)
  print(summary(model))
  # Extract the Adjusted R-squared value
  adjusted_r2 <- model_summary$adj.r.squared
  
  # Print the Adjusted R-squared value
  print("Adjusted R2")
  print(adjusted_r2)
  
  
  
  capture.output({
    print(label)
    print(ct)
    print("Corr perm test p-val:")
    print(pperm)
    print(rho)
    print("DCCA:")
    print(pval)
    print("Prediction correlation")
    print(ct)
    print(pperm)
    print("Lagged model:")
    print(ct)
    print(kpt)
    print(filename)
    print("Mean squire error: ")
    print("Normal")
    print(mses)
    print("scaled: ")
    print(msescale)
    print("R2: ")
    print(summary(model))
    
  }, file = filename)
  return(results)  
}


