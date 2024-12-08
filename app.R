library(shiny)
library(kknn)
library(shinydashboard)
library(shinyBS)
library(ggplot2)
library(plotly)
library(caret)

# read data from gtzan dataset
data <- read.csv("features_30_sec.csv")

# feature groups from dataset
feature_groups <- list(
  "MFCC Features" = grep("^mfcc", colnames(data), value = TRUE),
  "Spectral Features" = c("spectral_centroid_mean", "spectral_bandwidth_mean", 
                          "rolloff_mean", "zero_crossing_rate_mean"),
  "RMS & Chroma Features" = c("rms_mean", "chroma_stft_mean"),
  "All Features" = colnames(data)[-c(1, ncol(data))]  # excludin label and filename
)

# ui design 
ui <- dashboardPage(
  dashboardHeader(title = "KNN Music Genre Classifier"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Application", tabName = "application", icon = icon("cog")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    )
  ),
  dashboardBody(
    tabItems(
      # App 
      tabItem(
        tabName = "application",
        fluidRow(
          box(
            title = "Model Settings",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            sliderInput("k", "Number of Neighbors (k):", min = 1, max = 20, value = 5),
            selectInput("distance", "Distance Metric:",
                        choices = c("Euclidean" = 2, "Manhattan" = 1)), #needs to be numerical for kknn
            selectInput("weighting", "Weighting Scheme:",
                        choices = c("Uniform" = "rectangular", "Triangular" = "triangular", 
                                    "Gaussian" = "gaussian", "Epanechnikov" = "epanechnikov")),
            selectInput("featureGroup", "Feature Groups:", 
                        choices = names(feature_groups), selected = "All Features"),
            sliderInput("cvFolds", "Number of Cross-Validation Folds:", min = 2, max = 15, value = 5),
            actionButton("train", "Train k-NN Classifier"),
            actionButton("sweep", "Run Parameter Sweep"),
            actionButton("setOptimal", "Use Optimal Parameters", enabled = FALSE),
            
            bsTooltip("train", "Train the k-NN classifier with the current parameters.", placement = "right"),
            bsTooltip("sweep", "Run a parameter sweep to find the best k value.", placement = "right"),
            bsTooltip("setOptimal", "Set the optimal parameters discovered by the parameter sweep.", placement = "right")
          
          ),
          box(
            title = "Model Performance with Confusion Matrix",
            status = "success",
            solidHeader = TRUE,
            collapsible = TRUE,
            verbatimTextOutput("accuracy"),
            tableOutput("confMatrix")
          )
        ),
        fluidRow(
          box(
            title = "Confusion Matrix Heatmap",
            status = "warning",
            solidHeader = TRUE,
            collapsible = TRUE,
            plotlyOutput("heatmap")
          ),
          box(
            title = "Class Accuracy",
            status = "info",
            solidHeader = TRUE,
            collapsible = TRUE,
            plotlyOutput("classAccuracyPlot")
          )
        ),
        fluidRow(
          box(
            title = "Scoreboard (Run History)",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            tableOutput("scoreboard")
          ),
          box(
            title = "Parameter Sweep Results",
            status = "info",
            solidHeader = TRUE,
            collapsible = TRUE,
            plotlyOutput("sweep2DPlot", height = "500px")
          )
        ),
      ),
      # About Tab
      tabItem(
        tabName = "about",
        uiOutput("about_content")
      )
    )
  )
)

# application logic
server <- function(input, output, session) {
  # store run results 
  run_history <- reactiveVal(data.frame(
    Run = integer(),
    k = integer(),
    Distance = character(),
    Weighting = character(),
    Accuracy = numeric(),
    Time = character()
  ))
  
  
  # store parameter sweep results
  sweep_results <- reactiveVal(data.frame(
    k = integer(),
    Accuracy = numeric()
  ))
  
  # store confusin matrix and accuracy
  model_results <- reactiveValues(
    confMatrix = NULL,
    accuracy = NULL,
    classAccuracy = NULL
  )
  
  # function to ensure numerical values
  get_numeric_features <- function(feature_group) {
    selected_features <- data[, feature_groups[[feature_group]], drop = FALSE]
    numeric_features <- selected_features[, sapply(selected_features, is.numeric)]
    return(numeric_features)
  }
  
  # function to train and evaluate k-NN using kknn with cross-validation
  train_knn_cv <- function(k, selected_features, labels, distance_metric, kernel, folds) {
    set.seed(123)
    labels <- as.factor(labels) 
    control <- trainControl(method = "cv", number = folds)
    
    model <- train(
      x = scale(selected_features),
      y = labels,
      method = "kknn",
      tuneGrid = data.frame(kmax = k, distance = distance_metric, kernel = kernel),
      trControl = control
    )
    confMatrix <- confusionMatrix(model)
    mean_accuracy <- model$results$Accuracy
    class_accuracy <- diag(prop.table(confMatrix$table, 2))
    return(list(accuracy = mean_accuracy, confMatrix = confMatrix$table, classAccuracy = class_accuracy))
  }
  
  # train k-NN with cross-validation 
  observeEvent(input$train, {
    selected_features <- get_numeric_features(input$featureGroup)
    labels <- data$label
    
    withProgress(message = 'Training k-NN Classifier', value = 0, {
      total_steps <- input$cvFolds
      results <- train_knn_cv(
        input$k,
        selected_features,
        labels,
        as.numeric(input$distance),
        input$weighting,
        input$cvFolds
      )
      # increment pgrogress bar
      for (i in seq_len(total_steps)) {
        incProgress(1 / total_steps, detail = paste("Processing fold", i))
        Sys.sleep(0.1)
      }
      
      model_results$accuracy <- results$accuracy
      model_results$confMatrix <- results$confMatrix
      model_results$classAccuracy <- results$classAccuracy
    })
    
    # update run history
    current_history <- run_history()
    new_run <- data.frame(
      Run = paste0("#", as.integer(nrow(current_history) + 1)),
      k = input$k,
      Distance = input$distance,
      Weighting = input$weighting,
      Accuracy = round(results$accuracy * 100, 2),
      Time = format(Sys.time(), "%H:%M:%S") 
    )
    run_history(rbind(current_history, new_run))
    
    # update outputs
    output$accuracy <- renderText({
      paste("Overall Cross-Validation Accuracy:", round(model_results$accuracy * 100, 2), "%")
    })
    output$confMatrix <- renderTable({
      as.data.frame.matrix(model_results$confMatrix)
    })
    # confusion matrix as heatmap
    output$heatmap <- renderPlotly({
      heatmap_data <- as.data.frame(as.table(model_results$confMatrix))
      colnames(heatmap_data) <- c("Actual", "Predicted", "Count")
      ggplot(heatmap_data, aes(x = Predicted, y = Actual, fill = Count)) +
        geom_tile() +
        scale_fill_gradient(low = "white", high = "blue") +
        labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Actual") +
        theme_minimal()
    })
    # individual genre accuracy plot
    output$classAccuracyPlot <- renderPlotly({
      accuracy_data <- data.frame(
        Class = names(model_results$classAccuracy),
        Accuracy = model_results$classAccuracy * 100
      )
      ggplot(accuracy_data, aes(x = Class, y = Accuracy, fill = Class)) +
        geom_bar(stat = "identity") +
        labs(title = "Class Accuracy", x = "Class", y = "Accuracy (%)") +
        theme_minimal()
    })
    output$scoreboard <- renderTable({
      run_history()
    })
  })
  
  output$about_content <- renderUI({
    includeHTML("www/about.html") # include rmd render
  })
  
  
  # run parameter sweep to find optimal k
  observeEvent(input$sweep, {
    selected_features <- get_numeric_features(input$featureGroup)
    labels <- data$label
    results <- expand.grid(k = 1:20)
    
    withProgress(message = 'Running Parameter Sweep', value = 0, {
      total <- nrow(results)
      results$Accuracy <- sapply(seq_len(total), function(i) {
        incProgress(1 / total, detail = paste("Evaluating k =", results$k[i]))
        train_knn_cv(results$k[i], selected_features, labels, 
                     as.numeric(input$distance), input$weighting, input$cvFolds)$accuracy
      })
    })
    
    sweep_results(results)
    updateActionButton(session, "setOptimal", disabled = FALSE)
  })
  
  # set optimal parameters
  observeEvent(input$setOptimal, {
    best <- sweep_results()[which.max(sweep_results()$Accuracy), ]
    updateSliderInput(session, "k", value = best$k)
  })
  
  # parameter sweep result plot
  output$sweep2DPlot <- renderPlotly({
    results <- sweep_results()
    plot_ly(
      data = results,
      x = ~k,
      y = ~Accuracy,
      type = "scatter",
      mode = "lines+markers",
      marker = list(size = 8),
      line = list(width = 2)
    ) %>%
      layout(
        title = "Parameter Sweep: Accuracy vs. k",
        xaxis = list(title = "k (Neighbors)"),
        yaxis = list(title = "Accuracy"),
        margin = list(t = 40)
      )
  })
}

shinyApp(ui = ui, server = server)
