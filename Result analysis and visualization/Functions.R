---
title: "Comparative Approach: Temporal Question Analysis"
date: "November 2024"
author: "Md Touhidul Islam"

---
processYouGov <- function(file, sheet)
{
  # Read the Excel file data
  widedata <- read_excel(file, sheet)
  
  # Rename first column to 'mood'
  names(widedata)[1] <- "mood"
  N <- ncol(widedata)
  
  # Extract date strings from column headers
  dateStrings <- names(widedata)[2:N]
  
  # Convert month names to numeric format (01-12)
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
  
  # Remove ordinal suffixes (th, st, nd, rd) from day numbers
  dateStrings <- gsub("th", "", dateStrings)
  dateStrings <- gsub("st", "", dateStrings)
  dateStrings <- gsub("nd", "", dateStrings)
  dateStrings <- gsub("rd", "", dateStrings)
  
  # Convert to standard date format
  dateStrings <- gsub(" ", "-", dateStrings, fixed = T)
  dateStrings <- as.character(as.Date(as.POSIXlt(dateStrings, format = "%d-%m-%Y")))
  
  # Replace column names with standardized dates
  names(widedata)[2:N] <- dateStrings
  
  # Transform data from wide to long format
  longdf <- NULL
  for (ds in dateStrings)
  {
    # Extract values for each date, remove percentage signs and convert to numeric
    v <- as.character(widedata[[ds]])
    v <- as.numeric(gsub("%","", v))
    longdf <- rbind(longdf, data.frame(date=ds, measure=widedata$mood, count=v))
  }
  
  # Extract each emotion category into separate dataframes
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
  
  # Combine all emotion measures into a wide format dataframe
  ygdf <- data.frame(date=happydf$date, happy=happydf$count, sad=saddf$count, 
                     energetic=energeticdf$count, apathetic=apatheticdf$count,
                     scared=scareddf$count, frustrated=frustrateddf$count,
                     inspired=inspireddf$count, 
                     content=contentdf$count, optimistic=optimisticdf$count,
                     stressed=stresseddf$count, lonely=lonelydf$count, bored=boreddf$count)
  return(ygdf)
  
}
#' Create a Time Series Plot with Three Variables
#'
#' This function creates a comparison plot of three time series, with the first variable
#' from one dataframe and the other two from a second dataframe. All values are normalized
#' (scaled) to allow for comparison regardless of their original units or ranges.
#'
#' @param df1 First dataframe containing the first time series
#' @param df2 Second dataframe containing the second and third time series
#' @param var1 Variable name in df1 to plot
#' @param var2 First variable name in df2 to plot
#' @param var3 Second variable name in df2 to plot
#' @param col1 Color for the first time series
#' @param col2 Color for the second time series
#' @param col3 Color for the third time series
#' @param label1 Legend label for the first time series
#' @param label2 Legend label for the second time series
#' @param label3 Legend label for the third time series
#' @param title Plot title, defaults to "Time series comparison"
#' @param ylab Y-axis label, defaults to "measurement"
#' @param ymin Minimum value for y-axis (optional)
#' @param ymax Maximum value for y-axis (optional)
#' @return A ggplot2 object containing the time series plot
#'
#' @author MD Touhidul Islam
plotTS2 <- function(df1, df2, var1, var2, var3, col1, col2, col3, label1, label2, label3, title="Time series comparison", ylab="measurement", ymin=NA, ymax=NA)
{
  # Extract numeric values from the dataframes for each variable
  w1 <- as.numeric(df1[[var1]])
  w2 <- as.numeric(df2[[var2]])
  w3 <- as.numeric(df2[[var3]])
  
  # Ensure 'day' columns are in Date format for proper plotting
  df1$day <- as.Date(df1$day)
  df2$day <- as.Date(df2$day)
  
  # Create data frames for each series with consistent column names
  ddf1 <- data.frame(day=df1$day, y1=w1)
  ddf2 <- data.frame(day=df2$day, y2=w2, y3=w3)
  
  # Merge data frames on the 'day' column, keeping only matching dates
  df <- merge(ddf1, ddf2, by="day", all=FALSE)  # all=FALSE ensures only matching rows
  
  # Reshape data into long format required by ggplot2
  df <- data.frame(day=rep(df$day, 3),
                   y=c(df$y1, df$y2, df$y3),
                   dataset=c(rep(label1, nrow(df)), rep(label2, nrow(df)), rep(label3, nrow(df))))
  
  # Scale (normalize) the values separately for each dataset to enable comparison
  df$y[df$dataset==label1] <- scale(df$y[df$dataset==label1])
  df$y[df$dataset==label2] <- scale(df$y[df$dataset==label2])
  df$y[df$dataset==label3] <- scale(df$y[df$dataset==label3])
  print(df)  # Print the processed data frame for debugging
  
  # Create the plot using ggplot2
  plt <- ggplot(df, aes(x=as.Date(day), y=y, group=dataset, color=dataset)) + 
    geom_line() + 
    theme_bw() + 
    theme(legend.position="bottom") +
    scale_color_manual(values=c(col1, col2, col3)) + 
    xlab("Date") + 
    ylab(ylab) + 
    ggtitle(title) + 
    geom_vline(xintercept = as.Date("2023-03-16"), color=rgb(0,0,0,0.5))  # Add vertical line for reference date
  
  # Apply y-axis limits if specified
  if (!is.na(ymin) & !is.na(ymax)) {
    plt <- plt + ylim(ymin, ymax)
  }
  
  return(plt)
}
#' Plot Four Time Series Variables from a Single DataFrame
#'
#' @param df DataFrame containing time series data
#' @param var1-var4 Names of variables to plot
#' @param col1-col4 Colors for each variable
#' @param label1-label4 Labels for legend
#' @param plot_name Name for saved plot file
#' @return ggplot object
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
#' Plot Two Time Series Variables from Different DataFrames
#'
#' @param df1,df2 Input dataframes with time series data
#' @param var1,var2 Variable names to plot from each dataframe
#' @param col1,col2 Colors for each time series
#' @param label1,label2 Labels for the legend
#' @param plot_name Name for the saved plot file
#' @return ggplot object with normalized time series comparison
plotTS <- function(df1, df2, var1, var2, col1, col2, label1, label2, title="Time series comparison", ylab="measurement", ymin=NA, ymax=NA,plot_name)
{
  # Extract numeric values from both dataframes
  w1 <- as.numeric(df1[[var1]])
  w2 <- as.numeric(df2[[var2]])

  # Create dataframes with consistent column names and ensure dates are in Date format
  ddf1 <- data.frame(day=as.Date(df1$day), y1=w1)
  ddf2 <- data.frame(day=as.Date(df2$day), y2=w2)
  
  # Join dataframes on matching dates only
  df <- inner_join(ddf1, ddf2, by="day")
  
  # Reshape to long format for ggplot
  df <- data.frame(day=rep(df$day,2), y=c(df$y1,df$y2), dataset=c(rep(label1, nrow(df)), rep(label2, nrow(df))))
  
  # Scale each dataset separately for normalized comparison
  df$y[df$dataset==label1] <- scale(df$y[df$dataset==label1])
  df$y[df$dataset==label2] <- scale(df$y[df$dataset==label2])
  
  # Create plot with ggplot2
  plt <- ggplot(df, aes(x=as.Date(day), y=y, group=dataset, color=dataset)) + geom_line() + 
    theme_bw() + theme(legend.position="bottom") +
    scale_color_manual(values=c(col1, col2))+ xlab("Date") + ylab(ylab)  + ggtitle(title) + 
    geom_vline(xintercept = as.Date("2020-11-01"), color=rgb(0,0,0,0.5))  # Add reference line
  
  # Create file path for saving
  plot_name = paste("plot/",plot_name,".jpg")
  
  # Apply y-axis limits if specified and save the plot
  if (!is.na(ymin) & ! is.na(ymax))
  {
    plt <- plt + ylim(ymin, ymax)
    ggsave(plot_name, plot = plt,width = 9, height = 5, units = "in")
  }
  return(plt)
}


#' Perform Detrended Cross-Correlation Analysis (DCCA) Test
#'
#' @param ts1,ts2 Time series to analyze
#' @param nu,m DCCA parameters
#' @param R Number of random permutations
#' @return Matrix of DCCA results
dcca.test <- function(ts1, ts2, nu=1, m=12, R=10000)
{
  # Initialize results matrix
  rnd <- NULL
  # Run permutation test R times
  for (i in seq(1,R))
  {
    # For each permutation, randomly shuffle ts1 and calculate DCCA with ts2
    rnd <- rbind(rnd, rhodcca(sample(ts1), ts2, nu=nu, m=m)$rhodcca)
  }
  
  return(rnd)
}

#' Permutation Correlation Test
#'
#' @param X,Y Vectors to analyze correlation between
#' @param R Number of permutations
#' @return Matrix of correlation results
permcor <- function(X,Y, R=10000)
{
  # Initialize results
  rnd <- NULL
  # Perform R permutations
  for (i in seq(1,R))
  {
    # For each iteration, randomly shuffle X and calculate correlation with Y
    rnd <- rbind(rnd,cor(sample(X),Y))
  }
  
  return(rnd)
}

#' Generate Standardized Filename for Results
#'
#' @param epoch Model epoch number
#' @param query_type Type of query
#' @param similarity Similarity measure
#' @return Formatted filename string
getfilename <- function(epoch, query_type, similarity){
  # Construct standardized filename
  filename= paste("Result/",similarity,query_type,"_quary_epoch",epoch,".xlsx",sep = "")
  print(filename)
  return(filename)
}
#' Comprehensive Statistical Analysis of Time Series Data
#'
#' @param Twittersel DataFrame with Twitter data
#' @param Yougovdf DataFrame with YouGov survey data
#' @param R Number of permutations for tests
#' @param label Label for output file
#' @return DataFrame with statistical results
analyses <- function(Twittersel, Yougovdf, R,label)
{
  
  # Filter data up to specific date
  f <- as.Date(Twittersel$day) < as.Date("2023-01-05")
  X <- Twittersel$x[f]  # Twitter data
  Y <- Yougovdf$y[f]    # YouGov data
 
  print("----normal Corelation----")
  print("Correlation up to Mar 16st 2023")
  ct <- cor.test(X,Y)  # Standard correlation test
  print(ct)
  
  
  
  # Permutation-based correlation test
  print("------Permutation COrelation--- ")
  rndR <- permcor(X,Y,R) # do a permutation test 
  pperm <- (sum(rndR>=cor(X,Y))+1)/R  # Calculate p-value from permutations
  print("Corr permation test p-val:")
  print(pperm)
  
  # Store initial results in dataframe
  results <- data.frame(rh = ct$estimate, rhlow = ct$conf.int[1], rhhigh=ct$conf.int[2], rhp = ct$p.value, rhpperm = pperm)  
  
  # Detrended Cross-Correlation Analysis (DCCA)
  print("DCCA:")
  rho <- rhodcca(X, Y, nu=1, m=12)$rhodcca  # Calculate DCCA coefficient
  rndR <- dcca.test(X, Y, nu=1, m=12, R=R)  # Permutation test for DCCA
  print(rho)
  print("DCCA p-val:")
  pval <- (sum(rndR>=rho)+1)/R  # Calculate p-value
  print(pval)

  # Add DCCA results to the dataframe
  results <- cbind(results, data.frame(dcca=rho, dccap=pval))  
  
  # Prediction evaluation on data after cutoff date
  print("----Another permutaion---")
  print("Prediction correlation")
  f <- as.Date(Twittersel$day) >= as.Date("2023-01-05")  # Filter newer data
  X <- Twittersel$x[f]  # Twitter data (newer)
  Y <- Yougovdf$y[f]    # YouGov data (newer)
  ct <- cor.test(X,Y)    # Correlation test on newer data
  print(ct)
  rndR <- permcor(X,Y,R) # do a permutation test 
  pperm <- (sum(rndR>=cor(X,Y))+1)/R  # Permutation p-value
  print(pperm)
  
  # Add prediction correlation results to dataframe
  results <- cbind(results, data.frame(rp = ct$estimate, rplow = ct$conf.int[1], rphigh=ct$conf.int[2], rpp = ct$p.value, rppperm = pperm))  

  # Refilter data for the lagged model
  f <- as.Date(Twittersel$day) < as.Date("2023-01-05")
  X <- Twittersel$x[f]
  Y <- Yougovdf$y[f]
  
  # Lagged regression model analysis
  print("Lagged model:")
  Y <- scale(Y)  # Standardize variables
  X <- scale(X)
  model <- lm(Y ~ X + lag(Y,1))  # Regression with lagged dependent variable
  ct <- coeftest(model, vcov=vcovHAC(model))  # HAC robust standard errors
  print(ct)
  kpt <- kpss.test(model$residuals)  # Test for stationarity of residuals
  print(kpt)
  
  # Add lagged model results
  results <- cbind(results, data.frame(beta=ct[2,1], betap = ct[2,3], KSp=kpt$p.value))
  rownames(results) <- NULL
  
  # Prepare output filename
  filename= paste("output/",label,'.txt')
  msescale<-mean((X - Y)^2)  # MSE with scaled values
  
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


