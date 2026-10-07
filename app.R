# ==============================================================================
# Moderate-to-severe AKI risk predictor after Type A aortic dissection (Shiny)
# ------------------------------------------------------------------------------
# Model source: validation-best model trained by 03_run_8models.R
#           model/BEST_XGBoost_model.rds (caret::train object, xgbTree)
# Outcome definition: moderate-to-severe AKI (KDIGO stages 2-3);
#           outputs the predicted probability of this outcome (0-100%)
# Features:
#   - All input widgets are generated automatically from the model's embedded
#     training data (column names / ranges / factor levels), so swapping in
#     another .rds model (other algorithms in the same project) also works;
#   - Optional SHAP individual-level explanation (click button, takes a few
#     seconds).
# Disclaimer: this app is a research tool; it does not replace clinical
#   judgment.
# ==============================================================================

# --------------------------- Package dependency check ------------------------
needed <- c("shiny", "bslib", "ggplot2", "shapviz", "kernelshap", "thematic")
missing <- setdiff(needed, rownames(installed.packages()))
if (length(missing) > 0) {
  install.packages(missing, repos = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
}

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(ggplot2)
  library(shapviz)
  library(kernelshap)
  library(thematic)
})

# ------------------------------- Configuration ------------------------------
MODEL_FILE     <- file.path("model", "BEST_XGBoost_model.rds")
RISK_CUT_LOW   <- 0.30      # low / intermediate risk cut-off
RISK_CUT_HIGH  <- 0.60      # intermediate / high risk cut-off
SHAP_BG_N      <- 100       # SHAP background sample size (larger = slower)
SHAP_SEED      <- 20260425  # random seed for background sampling

FAMILY <- ""   # system default font (app is in English)
theme_plot <- theme_bw(base_size = 12, base_family = FAMILY) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

# ------------------------------- Load model ---------------------------------
if (!file.exists(MODEL_FILE)) {
  stop("Model file not found: ", MODEL_FILE,
       "\nPlease keep app.R together with its sibling model/ subdirectory.")
}
mdl <- readRDS(MODEL_FILE)

# Extract training data from the model object (original column names, value
# ranges, factor levels)
tr_data     <- mdl$trainingData
outcome_col <- attr(tr_data, "outcomeName")
if (is.null(outcome_col)) outcome_col <- ".outcome"
feat_names  <- setdiff(names(tr_data), outcome_col)
outcome_vec <- tr_data[[outcome_col]]
positive    <- levels(outcome_vec)[2]   # positive class (Severe)
negative    <- levels(outcome_vec)[1]

pred_prob <- function(model, newdata) {
  predict(model, newdata = newdata, type = "prob")[[positive]]
}

# Feature metadata: type / value range / median (used to generate input widgets
# and default values)
feat_meta <- lapply(feat_names, function(v) {
  x <- tr_data[[v]]
  if (is.factor(x)) {
    list(name = v, type = "factor", levels = levels(x))
  } else {
    x <- as.numeric(x)
    list(name = v, type = "numeric",
         min = min(x, na.rm = TRUE),
         max = max(x, na.rm = TRUE),
         median = round(median(x, na.rm = TRUE), 1))
  }
})
names(feat_meta) <- feat_names

# SHAP background data (sampled at startup for reproducibility)
set.seed(SHAP_SEED)
bg_df <- tr_data[sample(nrow(tr_data), min(SHAP_BG_N, nrow(tr_data))), feat_names]

# Risk stratification
risk_level <- function(p) {
  if (p < RISK_CUT_LOW) "Low risk" else if (p < RISK_CUT_HIGH) "Intermediate risk" else "High risk"
}
risk_color <- function(p) {
  if (p < RISK_CUT_LOW) "success" else if (p < RISK_CUT_HIGH) "warning" else "danger"
}

# Build a new-patient data frame matching the training data structure (identical
# column order, types, and factor levels)
build_patient <- function(input) {
  out <- lapply(feat_names, function(v) {
    m <- feat_meta[[v]]
    if (m$type == "factor") factor(input[[v]], levels = m$levels)
    else as.numeric(input[[v]])
  })
  names(out) <- feat_names
  as.data.frame(out, stringsAsFactors = FALSE)
}

# Dynamically generate input widgets (factor -> dropdown; numeric -> numeric input)
make_input_ui <- function() {
  lapply(feat_names, function(v) {
    m <- feat_meta[[v]]
    if (m$type == "factor") {
      lv <- as.character(m$levels)
      if (identical(lv, c("0", "1"))) {
        choices <- setNames(lv, c("No (0)", "Yes (1)"))
      } else {
        choices <- setNames(lv, lv)
      }
      selectInput(v, label = v, choices = choices, selected = lv[1])
    } else {
      step <- if ((m$max - m$min) <= 5) 0.1 else 1
      numericInput(v, label = v, value = m$median,
                   min = m$min, max = m$max, step = step)
    }
  })
}

# ------------------------------- UI -----------------------------------------
ui <- page_sidebar(
  title = "Moderate-to-Severe AKI Risk Predictor after Type A Aortic Dissection",
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  sidebar = sidebar(
    width = 360,
    accordion(
      open = TRUE,
      accordion_panel(
        "Clinical parameters",
        icon = NULL,
        uiOutput("input_panel")
      )
    ),
    layout_column_wrap(
      width = 1/2,
      fill = FALSE,
      actionButton("reset", "Reset to defaults", width = "100%"),
      actionButton("example", "Random example", width = "100%")
    ),
    p(style = "font-size: 12px; color: #6c757d; margin-top: 8px;",
      "Numeric input ranges are taken from the observed ranges in the training set.")
  ),
  layout_column_wrap(
    width = 1/3,
    fill = FALSE,
    uiOutput("box_prob"),
    uiOutput("box_level"),
    uiOutput("box_prev")
  ),
  layout_column_wrap(
    width = 1/2,
    card(
      full_screen = TRUE,
      card_header("Predicted probability"),
      plotOutput("gauge", height = "260px")
    ),
    card(
      full_screen = TRUE,
      card_header(
        layout_column_wrap(
          width = 1,
          fill = FALSE,
          "Individual feature contributions (SHAP)",
          actionButton("shap_btn", "Compute explanation", class = "btn-sm")
        )
      ),
      p(style = "font-size: 12px; color: #6c757d;",
        "Computed from training-set background samples; re-click after changing parameters."),
      plotOutput("shap_plot", height = "380px")
    )
  ),
  card(
    full_screen = TRUE,
    card_header("Interpretation"),
    uiOutput("interpret")
  ),
  card(
    full_screen = TRUE,
    card_header("Model info & disclaimer"),
    uiOutput("model_info")
  )
)

# ------------------------------- Server -------------------------------------
server <- function(input, output, session) {
  thematic_shiny(font = FAMILY)

  output$input_panel <- renderUI(make_input_ui())

  patient <- reactive(build_patient(input))

  prob <- reactive({
    ok <- vapply(feat_names, function(v) !is.null(input[[v]]) && !is.na(input[[v]]),
                 logical(1))
    req(all(ok))
    pred_prob(mdl, patient())
  })

  # ---- Top metric boxes ----
  output$box_prob <- renderUI({
    p <- prob()
    value_box(
      title = "Moderate-to-severe AKI probability",
      value = sprintf("%.1f%%", 100 * p),
      theme = risk_color(p),
      full_screen = FALSE
    )
  })
  output$box_level <- renderUI({
    p <- prob()
    value_box(
      title = "Risk stratum",
      value = risk_level(p),
      theme = risk_color(p)
    )
  })
  output$box_prev <- renderUI({
    value_box(
      title = "Cohort prevalence (training set)",
      value = sprintf("%.1f%%", 100 * mean(outcome_vec == positive)),
      theme = "info"
    )
  })

  # ---- Probability gauge ----
  output$gauge <- renderPlot({
    p <- prob()
    bands <- data.frame(
      xmin = c(0, RISK_CUT_LOW, RISK_CUT_HIGH),
      xmax = c(RISK_CUT_LOW, RISK_CUT_HIGH, 1),
      lab  = c("Low risk", "Intermediate risk", "High risk")
    )
    ggplot() +
      geom_rect(data = bands,
                aes(xmin = xmin, xmax = xmax, ymin = 0.1, ymax = 0.9, fill = lab),
                alpha = 0.25) +
      scale_fill_manual(values = c("Low risk" = "#2ca02c",
                                   "Intermediate risk" = "#ff9f1c",
                                   "High risk" = "#d62728")) +
      annotate("segment", x = p, xend = p, y = 0, yend = 1.12,
               linewidth = 1.8, colour = "#222222") +
      annotate("point", x = p, y = 1.12, size = 4, colour = "#222222") +
      annotate("text", x = p, y = 1.3, label = sprintf("%.1f%%", 100 * p),
               size = 6, fontface = "bold", colour = "#222222") +
      scale_x_continuous(limits = c(0, 1),
                         breaks = c(0, 0.25, 0.5, 0.75, 1),
                         labels = c("0%", "25%", "50%", "75%", "100%")) +
      coord_cartesian(ylim = c(0, 1.45)) +
      labs(x = NULL, y = NULL) +
      theme_plot +
      theme(axis.text.y = element_blank(),
            axis.ticks.y = element_blank())
  })

  # ---- SHAP individual explanation (computed on click) ----
  shap_obj <- eventReactive(input$shap_btn, {
    withProgress(message = "Computing SHAP explanation...", value = 0.2, {
      incProgress(0.4, detail = "Computing contributions from background samples")
      ks <- kernelshap(mdl, X = patient(), pred_fun = pred_prob,
                       bg_X = bg_df, verbose = FALSE)
      incProgress(0.3, detail = "Generating plot")
      shapviz(ks, X = patient())
    })
  })

  output$shap_plot <- renderPlot({
    req(shap_obj())
    sv_waterfall(shap_obj(), row_id = 1) +
      labs(title = "SHAP decomposition of this patient's predicted probability (left = decreases risk / right = increases risk)") +
      theme_plot
  })

  # ---- Interpretation ----
  output$interpret <- renderUI({
    p <- prob()
    tags$ul(
      tags$li(sprintf("Predicted probability of moderate-to-severe AKI (KDIGO stages 2-3) after surgery for this patient: %.1f%%.", 100 * p)),
      tags$li(sprintf("Classified as '%s' under the current thresholds (<30%% low risk; 30-60%% intermediate risk; >=60%% high risk).",
                      risk_level(p))),
      tags$li("Risk thresholds are only to aid interpretation and have not been prospectively validated; please combine with clinical judgment.")
    )
  })

  # ---- Model info ----
  output$model_info <- renderUI({
    tags$ul(
      tags$li(sprintf("Model: %s (caret xgbTree, best model on validation set)", basename(MODEL_FILE))),
      tags$li(sprintf("%d features: %s", length(feat_names),
                      paste(feat_names, collapse = ", "))),
      tags$li(sprintf("Training sample size: %d; positive class (moderate-to-severe AKI) proportion: %.1f%%.",
                      nrow(tr_data), 100 * mean(outcome_vec == positive))),
      tags$li("Validation AUC and other performance metrics are in the OUTPUT/ folder produced by 03_run_8models.R."),
      tags$li("This tool is for research purposes only and does not replace clinical diagnosis or treatment decisions.")
    )
  })

  # ---- Reset / Example ----
  observeEvent(input$reset, {
    for (v in feat_names) {
      m <- feat_meta[[v]]
      if (m$type == "factor") {
        updateSelectInput(session, v, selected = as.character(m$levels[1]))
      } else {
        updateNumericInput(session, v, value = m$median)
      }
    }
  })

  observeEvent(input$example, {
    ex <- tr_data[sample(nrow(tr_data), 1), feat_names]
    for (v in feat_names) {
      m <- feat_meta[[v]]
      if (m$type == "factor") {
        updateSelectInput(session, v, selected = as.character(ex[[v]]))
      } else {
        updateNumericInput(session, v, value = as.numeric(ex[[v]]))
      }
    }
  })
}

shinyApp(ui, server)
