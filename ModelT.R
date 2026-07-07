# ==============================================================================
# TENNIS MATCH PREDICTION: CAN A MODEL FIND AN EDGE OVER BOOKMAKER ODDS?
# ==============================================================================
# This pipeline builds a calibrated win-probability model for ATP tennis matches
# from raw historical data, then rigorously tests whether that model contains
# any predictive information beyond what is already priced into bookmaker odds.
#
# Summary of findings (see bottom of script for full discussion):
#   - The model is well-calibrated (ECE ~0.013) and reasonably discriminative
#     (AUC ~0.72), but underperforms the bookmaker's own implied probability
#     (AUC ~0.75) on every proper scoring metric.
#   - A horse-race regression (outcome ~ market_logit + model_logit) shows the
#     model's coefficient is not statistically distinguishable from zero once
#     the market's price is controlled for -- i.e. the model adds no
#     information the market doesn't already have.
#   - This holds across every segment tested (surface, tournament tier, round,
#     favorite/underdog regime) and across two independent model variants.
#   - EV-threshold betting strategies were swept using proper methodology
#     (fit on a held-out calibration period, applied once to a forward test
#     period, evaluated with bootstrap confidence intervals) and show no
#     threshold with a lower CI bound above zero.
#   - Conclusion: with this feature set and this bookmaker's closing odds,
#     there is no statistically defensible betting edge. The value of this
#     project is the pipeline and the validation methodology, not a
#     profitable strategy.
# ==============================================================================

library(dplyr)
library(lubridate)
library(stringr)
library(tidyr)
library(xgboost)
library(ggplot2)
library(pROC)

# ------------------------------------------------------------------------------
# PHASE 1: DATA LOADING & STRUCTURAL CLEANING
# ------------------------------------------------------------------------------

tennis_raw <- read.csv("data/atp_tennis.csv")   # <-- update path as needed

# Enforce strict chronological tracking immediately -- everything downstream
# (Elo, career stats, train/test splits) depends on correct temporal order.
tennis_clean <- tennis_raw %>%
  mutate(Date = as.Date(Date)) %>%
  arrange(Date)

# --- Data quality summary (from initial audit) ---
# Rank_1/Rank_2: 14 / 12 rows with sentinel value -1 (unrankable players)
# Pts_1/Pts_2:   majority -1 before 2006 tour-points tracking began
# Odd_1/Odd_2:   3782 / 3780 rows with sentinel -1 (odds not recorded),
#                1 row with 0 (corrupted), 52 / 34 rows with exactly 1
#                (legitimate extreme favorites, not missing data)

# 1. Type casting: convert stray text markers to numeric, flag as near-1.01 odds
tennis_clean <- tennis_clean %>%
  mutate(
    Odd_1 = suppressWarnings(as.numeric(ifelse(trimws(Odd_1) == "-", "1.01", as.character(Odd_1)))),
    Odd_2 = suppressWarnings(as.numeric(ifelse(trimws(Odd_2) == "-", "1.01", as.character(Odd_2))))
  )

# 2. Fix 6 rows where odds were mistakenly entered as fractional (<1) instead
#    of decimal odds -- these get inverted back onto the correct scale.
tennis_clean <- tennis_clean %>%
  mutate(
    Temp_Odd_1 = Odd_1, Temp_Odd_2 = Odd_2,
    Odd_1 = ifelse(!is.na(Temp_Odd_1) & Temp_Odd_1 > 0 & Temp_Odd_1 < 1, round(1 / Temp_Odd_1, 3), Odd_1),
    Odd_2 = ifelse(!is.na(Temp_Odd_1) & Temp_Odd_1 > 0 & Temp_Odd_1 < 1, Temp_Odd_2, Odd_2),
    Odd_1 = ifelse(!is.na(Temp_Odd_2) & Temp_Odd_2 > 0 & Temp_Odd_2 < 1, Temp_Odd_1, Odd_1),
    Odd_2 = ifelse(!is.na(Temp_Odd_2) & Temp_Odd_2 > 0 & Temp_Odd_2 < 1, round(1 / Temp_Odd_2, 3), Odd_2)
  ) %>%
  select(-Temp_Odd_1, -Temp_Odd_2)

# 3. Drop rows with unrecoverable data: corrupted zero-odds lines and
#    unrankable players. Note what was deliberately KEPT:
#      - Pts = 1 kept (genuine tour rookies, not missing data)
#      - Rank = 1 kept (world No. 1, not missing data)
#      - Odd = 1 kept (extreme betting favorites, not missing data)
rows_to_remove <- which(
  tennis_clean$Odd_1 <= 0 |
    tennis_clean$Odd_2 <= 0 |
    tennis_clean$Rank_1 == -1 |
    tennis_clean$Rank_2 == -1 |
    is.na(tennis_clean$Odd_1) |
    is.na(tennis_clean$Odd_2)
)
if (length(rows_to_remove) > 0) tennis_clean <- tennis_clean[-rows_to_remove, ]

# 4. Drop Carpet surface -- discontinued on tour after 2008, no longer
#    representative of current playing conditions.
tennis_clean <- tennis_clean %>% filter(Surface != "Carpet")

# ------------------------------------------------------------------------------
# PHASE 2: PLAYER NAME UNIFICATION
# ------------------------------------------------------------------------------
# Raw data has ~1785 unique player name strings that collapse to ~1652 real
# players once inconsistent formatting, initials, and typos are resolved.
# This matters because every downstream feature (Elo, career match counts,
# first-set stats) is keyed on player identity -- a name split into two
# variants silently fragments that player's history.

normalize_tennis_names <- function(vec) {
  if (is.null(vec)) return(vec)
  vec %>%
    str_trim() %>%
    str_replace_all("\\s+", " ") %>%
    str_replace_all("-", " ") %>%
    str_replace_all("\\.", "") %>%
    str_replace_all("(?i)\\bde\\b", "De") %>%
    str_replace_all("(?i)\\bdi\\b", "Di") %>%
    str_replace_all("(?i)\\bvan\\b", "Van") %>%
    str_replace_all("(?i)\\bmcc", "McC") %>%
    str_replace_all("(?i)\\bmccl", "McCl") %>%
    str_replace_all("(?i)\\bmcd", "McD") %>%
    str_replace_all("(?i)\\bmcge", "McGe")
}

tennis_clean <- tennis_clean %>%
  mutate(
    Player_1 = normalize_tennis_names(Player_1),
    Player_2 = normalize_tennis_names(Player_2),
    Winner   = normalize_tennis_names(Winner)
  )

# Residual fixes for typos, initials, and formatting variants not caught by
# the rule-based normalizer above (built from manual review of duplicate names)
player_fix_dict <- c(
  "Aragone JC" = "Aragone J", "Bailly GA" = "Bailly G", "Alvarez Valdes LC" = "Alvarez Valdes L", "Andersen JF" = "Andersen JF",
  "Bogomolov Jr A" = "Bogomolov A", "Bogomolov JrA" = "Bogomolov A", "Del Potro J M" = "Del Potro JM", "Etcheverry T M" = "Etcheverry T",
  "Galan DE" = "Galan D", "Galan De" = "Galan D", "Gallardo Valles M" = "Gallardo M", "Gambill J M" = "Gambill JM",
  "Granollers Pujol G" = "Granollers G", "Granollers Pujol M" = "Granollers M", "Herbert P H" = "Herbert PH", "Herbert P" = "Herbert PH",
  "Hernandez Fernandez J" = "Hernandez J", "Lopez Jaen MA" = "Lopez MA", "Schwaerzler JJ" = "Schwaerzler J", "Stebe C M" = "Stebe CM",
  "Struff J L" = "Struff JL", "Tseng C H" = "Tseng CH", "Wang Y Jr" = "Wang Y", "Wang YJr" = "Wang Y",
  "Zayed M S" = "Zayid MS", "Zayid M S" = "Zayid MS", "Del Bonis F" = "Delbonis F", "Dutra Silva R" = "Dutra Da Silva R",
  "O Connell C" = "O'Connell C", "Varillas J P" = "Varillas JP", "Dolgopolov O" = "Dolgopolov A", "Nedovyesov O" = "Nedovyesov A",
  "Querry S" = "Querrey S", "Tiurnev E" = "Tyurnev E", "Kunitcin I" = "Kunitsyn I", "Nadal Parera R" = "Nadal R",
  "Viola Mat" = "Viola M", "Kuznetsov Al" = "Kuznetsov A", "Kuznetsov An" = "Kuznetsov A", "Blanch Dar" = "Blanch D",
  "Mccabe J" = "McCabe J", "Mcclune M" = "McClune M", "Mcdonald M" = "McDonald M", "Mcgee J" = "McGee J",
  "Zhang Ze" = "Zhang Z", "Al Alawi SK" = "Al Alawi S K", "Bautista R" = "Bautista Agut R", "Gimeno D" = "Gimeno Traver D",
  "Haider Mauer A" = "Haider Maurer A", "Munoz De la Nava D" = "Munoz De La Nava D", "Sanchez De Luna J" = "Sanchez De Luna JA",
  "Van D Merwe I" = "Van Der Merwe I", "Van Der Merwe I" = "Van Der Merwe I", "Andersen J" = "Andersen JF",
  "Carreno P" = "Carreno Busta P", "Chela J" = "Chela JI", "Ramos A" = "Ramos Vinolas A", "Riba P" = "Riba Madrid P",
  "Vassallo M" = "Vassallo Arguello M", "Vilella M" = "Vilella Martinez M"
)

tennis_clean <- tennis_clean %>%
  mutate(
    Player_1 = ifelse(Player_1 %in% names(player_fix_dict), player_fix_dict[Player_1], Player_1),
    Player_2 = ifelse(Player_2 %in% names(player_fix_dict), player_fix_dict[Player_2], Player_2),
    Winner   = ifelse(Winner   %in% names(player_fix_dict), player_fix_dict[Winner],   Winner)
  )

# ------------------------------------------------------------------------------
# PHASE 3: METRIC ANOMALY CORRECTIONS
# ------------------------------------------------------------------------------
# Found via a sanity check on ranking-points averages by rank bucket: in 2023,
# players ranked 501-1000 showed a HIGHER average points total than players
# ranked 251-500 -- mathematically impossible given how ATP points are
# awarded. Traced to two tournaments (Lyon Open, Geneva Open) with corrupted
# rank/points fields; corrected manually below. Also fixes one 2011 match
# where both players carried an identical rank (a data entry error, since
# ATP rankings cannot tie).

tennis_clean <- tennis_clean %>%
  mutate(Pts_2 = ifelse(Tournament == "Qatar Exxon Mobil Open" & Date == "2006-01-02" &
                           Player_1 == "Baghdatis M" & Player_2 == "Sultan Khalfan A" &
                           Pts_2 == -1, 1, Pts_2))

corrections_df <- data.frame(
  Row_ID = 58444:58491,
  True_Rank_1 = c(35, 75, 51, 37, 92, 58, 81, 74, 307, 86, 282, 49, 75, 59, 47, 9, 58, 4, 86, 33, 4, 33, 27, 54, 66, 55, 100, 52, 310, 78, 70, 53, 298, 44, 67, 63, 36, 52, 63, 17, 113, 10, 60, 14, 17, 28, 52, 112),
  True_Rank_2 = c(73, 149, 54, 186, 47, 190, 59, 48, 85, 421, 39, 160, 27, 73, 86, 74, 33, 49, 9, 85, 54, 9, 54, 33, 97, 61, 60, 93, 36, 165, 112, 50, 113, 80, 240, 117, 60, 78, 44, 55, 14, 240, 28, 44, 52, 14, 112, 28),
  True_Pts_1  = c(1095, 729, 899, 1041, 661, 841, 719, 730, 165, 696, 187, 905, 729, 839, 917, 3390, 841, 4915, 696, 1125, 4915, 1125, 1360, 862, 782, 846, 639, 877, 163, 725, 740, 876, 170, 955, 781, 793, 1055, 877, 793, 2135, 550, 3065, 832, 2520, 2135, 1345, 877, 559),
  True_Pts_2  = c(733, 398, 862, 286, 917, 274, 839, 915, 702, 105, 1024, 375, 1360, 733, 696, 730, 1125, 905, 3390, 702, 862, 3390, 862, 1125, 648, 823, 832, 660, 1055, 306, 559, 901, 550, 720, 236, 117, 832, 725, 955, 846, 2520, 236, 1345, 955, 877, 2520, 559, 1345)
)

tennis_clean <- tennis_clean %>%
  mutate(Row_ID_Temp = row_number()) %>%
  left_join(corrections_df, by = c("Row_ID_Temp" = "Row_ID")) %>%
  mutate(
    Rank_1 = ifelse(!is.na(True_Rank_1), True_Rank_1, Rank_1), Rank_2 = ifelse(!is.na(True_Rank_2), True_Rank_2, Rank_2),
    Pts_1  = ifelse(!is.na(True_Pts_1), True_Pts_1, Pts_1),     Pts_2  = ifelse(!is.na(True_Pts_2), True_Pts_2, Pts_2)
  ) %>%
  select(-Row_ID_Temp, -True_Rank_1, -True_Rank_2, -True_Pts_1, -True_Pts_2) %>%
  mutate(Rank_2 = ifelse(Date == "2011-04-18" & Tournament == "Open Banco Sabadell" &
                            Player_1 == "Ramirez Hidalgo R" & Player_2 == "Ramos Vinolas A", 108, Rank_2))

# ------------------------------------------------------------------------------
# PHASE 4: FEATURE ENGINEERING -- ELO, MOMENTUM, CAREER MATURITY
# ------------------------------------------------------------------------------

# Target variable + first-set outcome tokenization (used for momentum features)
tennis_clean <- tennis_clean %>%
  mutate(
    P1_Won             = ifelse(Winner == Player_1, 1L, 0L),
    First_Set_Token    = str_extract(Score, "^[0-9]+-[0-9]+"),
    First_Set_Games_P1 = as.integer(str_extract(First_Set_Token, "^[0-9]+")),
    First_Set_Games_P2 = as.integer(str_extract(First_Set_Token, "[0-9]+$")),
    P1_Won_FirstSet    = case_when(
      First_Set_Games_P1 > First_Set_Games_P2 ~ 1L,
      First_Set_Games_P1 < First_Set_Games_P2 ~ 0L,
      TRUE ~ NA_integer_)
  )

# Sequential, surface-specific Elo engine. Ratings update match-by-match in
# chronological order (no lookahead), tracked independently for Hard/Clay/Grass
# so a player's clay-court strength doesn't leak into their hard-court rating.
n_rows <- nrow(tennis_clean)
elo_hard <- numeric(n_rows); elo_clay <- numeric(n_rows); elo_grass <- numeric(n_rows)
matches_p1 <- numeric(n_rows); matches_p2 <- numeric(n_rows)

player_registry <- new.env(hash = TRUE, parent = emptyenv())
init_player <- function() c(1500, 1500, 1500, 0, 0, 0, 0)  # [EloH, EloC, EloG, nH, nC, nG, nTotal]

for (i in 1:n_rows) {
  p1 <- tennis_clean$Player_1[i]; p2 <- tennis_clean$Player_2[i]
  surf <- tennis_clean$Surface[i]; winner <- tennis_clean$Winner[i]

  if (!exists(p1, envir = player_registry, inherits = FALSE)) assign(p1, init_player(), envir = player_registry)
  if (!exists(p2, envir = player_registry, inherits = FALSE)) assign(p2, init_player(), envir = player_registry)

  s_1 <- get(p1, envir = player_registry); s_2 <- get(p2, envir = player_registry)
  matches_p1[i] <- s_1[7]; matches_p2[i] <- s_2[7]  # career maturity BEFORE this match

  s_idx <- case_when(surf == "Hard" ~ 1, surf == "Clay" ~ 2, surf == "Grass" ~ 3, TRUE ~ 1)
  m_idx <- s_idx + 3

  elo_hard[i]  <- s_1[1] - s_2[1]
  elo_clay[i]  <- s_1[2] - s_2[2]
  elo_grass[i] <- s_1[3] - s_2[3]

  exp_1 <- 1 / (1 + 10^((s_2[s_idx] - s_1[s_idx]) / 400))
  outcome_1 <- if (winner == p1) 1 else 0

  # Adaptive K-factor: faster rating movement for players with <10 matches
  # on this surface, to let new/returning players' ratings converge quicker
  k_1 <- if (s_1[m_idx] < 10) 40 else 20
  k_2 <- if (s_2[m_idx] < 10) 40 else 20

  s_1[s_idx] <- s_1[s_idx] + k_1 * (outcome_1 - exp_1)
  s_2[s_idx] <- s_2[s_idx] + k_2 * ((1 - outcome_1) - (1 - exp_1))
  s_1[m_idx] <- s_1[m_idx] + 1; s_2[m_idx] <- s_2[m_idx] + 1
  s_1[7] <- s_1[7] + 1;         s_2[7] <- s_2[7] + 1

  assign(p1, s_1, envir = player_registry); assign(p2, s_2, envir = player_registry)
}

tennis_clean$Elo_Diff_Hard <- elo_hard
tennis_clean$Elo_Diff_Clay <- elo_clay
tennis_clean$Elo_Diff_Grass <- elo_grass
tennis_clean$P1_Total_Matches <- matches_p1
tennis_clean$P2_Total_Matches <- matches_p2

# First-set "mental momentum" features: how often a player wins the first
# set, and how often they recover after losing it -- both computed as
# trailing (lagged) rates so no match's own outcome leaks into its own features.
first_set_long <- bind_rows(
  tennis_clean %>% mutate(Match_Index = row_number()) %>%
    select(Match_Index, Player = Player_1, Won_Match = P1_Won, Won_FirstSet = P1_Won_FirstSet),
  tennis_clean %>% mutate(Match_Index = row_number()) %>%
    select(Match_Index, Player = Player_2, Won_Match = P1_Won, Won_FirstSet = P1_Won_FirstSet) %>%
    mutate(Won_Match = as.integer(1L - Won_Match), Won_FirstSet = as.integer(1L - Won_FirstSet))
) %>%
  filter(!is.na(Won_FirstSet)) %>%
  arrange(Match_Index)

first_set_stats <- first_set_long %>%
  group_by(Player) %>%
  arrange(Match_Index, .by_group = TRUE) %>%
  mutate(
    Cumul_FirstSet_Wins  = lag(cumsum(Won_FirstSet), default = 0),
    Cumul_FirstSet_Total = lag(row_number() - 1,     default = 0),
    Lost_FirstSet        = as.integer(1L - Won_FirstSet),
    Recovered            = as.integer(Won_Match == 1L & Won_FirstSet == 0L),
    Cumul_Lost_FirstSet  = lag(cumsum(Lost_FirstSet), default = 0),
    Cumul_Recovered      = lag(cumsum(Recovered),      default = 0),
    FirstSet_WinRate     = ifelse(Cumul_FirstSet_Total == 0, 0.5, Cumul_FirstSet_Wins / Cumul_FirstSet_Total),
    Recovery_Rate        = ifelse(Cumul_Lost_FirstSet == 0, 0.5, Cumul_Recovered / Cumul_Lost_FirstSet)
  ) %>%
  ungroup() %>%
  select(Match_Index, Player, FirstSet_WinRate, Recovery_Rate)

fs_p1 <- first_set_stats %>%
  inner_join(tennis_clean %>% mutate(Match_Index = row_number()) %>% select(Match_Index, Player_1),
             by = c("Match_Index", "Player" = "Player_1")) %>%
  select(Match_Index, P1_FirstSet_WinRate = FirstSet_WinRate, P1_Recovery_Rate = Recovery_Rate)

fs_p2 <- first_set_stats %>%
  inner_join(tennis_clean %>% mutate(Match_Index = row_number()) %>% select(Match_Index, Player_2),
             by = c("Match_Index", "Player" = "Player_2")) %>%
  select(Match_Index, P2_FirstSet_WinRate = FirstSet_WinRate, P2_Recovery_Rate = Recovery_Rate)

tennis_clean <- tennis_clean %>%
  mutate(Match_Index = row_number()) %>%
  left_join(fs_p1, by = "Match_Index") %>%
  left_join(fs_p2, by = "Match_Index") %>%
  select(-Match_Index) %>%
  mutate(
    Starter_Adv_P1           = P1_FirstSet_WinRate - P2_FirstSet_WinRate,
    Comebacker_Vulnerability = ifelse(Starter_Adv_P1 >= 0, 1 - P2_Recovery_Rate, 1 - P1_Recovery_Rate),
    Momentum_Edge_P1         = Starter_Adv_P1 * Comebacker_Vulnerability
  )

# Note: head-to-head features (raw and surface-specific H2H differentials)
# were also engineered and tested, but consistently failed to improve
# out-of-sample model performance, so they were excluded from the final
# feature set.

# ------------------------------------------------------------------------------
# PHASE 5: FEATURE MATRIX & MODEL INPUT CONFIGURATION
# ------------------------------------------------------------------------------

series_weights <- c("Grand Slam"=4, "Masters Cup"=5, "Masters 1000"=3, "Masters"=3,
                     "ATP500"=2, "International Gold"=2, "ATP250"=1, "International"=1)
round_weights  <- c("The Final"=7, "Semifinals"=6, "Quarterfinals"=5, "4th Round"=4,
                     "3rd Round"=3, "2nd Round"=2, "1st Round"=1, "Round Robin"=0)

tennis_production_matrix <- tennis_clean %>%
  mutate(
    Odd_1 = as.numeric(as.character(Odd_1)),
    Odd_2 = as.numeric(as.character(Odd_2)),

    Log_Rank_Diff_P1   = log(Rank_2) - log(Rank_1),
    Active_Elo_Diff_P1 = case_when(Surface == "Hard" ~ Elo_Diff_Hard, Surface == "Clay" ~ Elo_Diff_Clay,
                                    Surface == "Grass" ~ Elo_Diff_Grass, TRUE ~ Elo_Diff_Hard),
    Maturity_Diff_P1   = P1_Total_Matches - P2_Total_Matches,
    Series_Weight      = as.numeric(series_weights[as.character(Series)]),
    Round_Depth        = as.numeric(round_weights[as.character(Round)]),
    Best.of_Numeric    = as.numeric(as.character(Best.of)),
    Year               = year(Date),

    # Vig-adjusted market probability: strip the bookmaker's overround so
    # the odds reflect a genuine implied probability rather than a market
    # that sums to >100%. Kept strictly as an EVALUATION benchmark, never
    # as a model predictor -- the entire point of this project is testing
    # whether the model can independently compete with this number.
    Implied_Sum          = (1 / Odd_1) + (1 / Odd_2),
    Bookie_True_Prob_P1  = (1 / Odd_1) / Implied_Sum
  ) %>%
  drop_na(Log_Rank_Diff_P1, Active_Elo_Diff_P1, Series_Weight, Round_Depth, Bookie_True_Prob_P1)

predictor_cols <- c("Log_Rank_Diff_P1", "Active_Elo_Diff_P1", "Maturity_Diff_P1",
                     "Series_Weight", "Round_Depth", "Best.of_Numeric", "Momentum_Edge_P1")
target_col <- "P1_Won"

# ------------------------------------------------------------------------------
# PHASE 6: CHRONOLOGICAL SPLIT, TRAINING & CALIBRATION
# ------------------------------------------------------------------------------
# Three-way chronological split (never random -- this is a time series):
#   train (2009-2022) -> calibration (2023-2024) -> forward test (2025-2026)
# Calibration is fit on its own held-out period, and the forward test period
# is touched only once, at the end, for final evaluation.

train_pool <- tennis_production_matrix %>% filter(Year >= 2009 & Year <= 2022)
calib_pool <- tennis_production_matrix %>% filter(Year >= 2023 & Year <= 2024)
test_pool  <- tennis_production_matrix %>% filter(Year >= 2025 & Year <= 2026)

dtrain_pool <- xgb.DMatrix(data = as.matrix(train_pool[, predictor_cols]), label = train_pool[[target_col]])
dcalib_pool <- xgb.DMatrix(data = as.matrix(calib_pool[, predictor_cols]))
dtest_pool  <- xgb.DMatrix(data = as.matrix(test_pool[, predictor_cols]))

xgb_params <- list(objective = "binary:logistic", eval_metric = "logloss", eta = 0.03,
                    max_depth = 4, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 3)
set.seed(42)
base_model <- xgb.train(params = xgb_params, data = dtrain_pool, nrounds = 150)

# Platt scaling: fits a logistic correction on held-out calibration-period
# logits, so the model's raw (often overconfident) probabilities are
# rescaled to match observed frequencies before touching the test period.
calib_pool <- calib_pool %>%
  mutate(Raw_Pred_P1 = pmin(pmax(predict(base_model, dcalib_pool), 1e-5), 1 - 1e-5),
         Logit_P1    = log(Raw_Pred_P1 / (1 - Raw_Pred_P1)))

calibration_fit <- glm(P1_Won ~ Logit_P1, data = calib_pool, family = binomial(link = "logit"))

test_pool_evaluated <- test_pool %>%
  mutate(Raw_Pred_P1 = pmin(pmax(predict(base_model, dtest_pool), 1e-5), 1 - 1e-5),
         Logit_P1    = log(Raw_Pred_P1 / (1 - Raw_Pred_P1))) %>%
  mutate(Calib_Prob_P1 = predict(calibration_fit, newdata = ., type = "response"),
         Calib_Prob_P2 = 1 - Calib_Prob_P1)

# ==============================================================================
# PHASE 7: MODEL DIAGNOSTICS -- PROPER SCORING RULES
# ==============================================================================

y_test     <- test_pool_evaluated$P1_Won
p_cal_test <- test_pool_evaluated$Calib_Prob_P1
p_raw_test <- test_pool_evaluated$Raw_Pred_P1

clip_prob     <- function(p, eps = 1e-15) pmin(pmax(p, eps), 1 - eps)
calc_logloss  <- function(y, p){ p <- clip_prob(p); -mean(y * log(p) + (1 - y) * log(1 - p)) }
calc_brier    <- function(y, p) mean((p - y)^2)
calc_accuracy <- function(y, p, thr = 0.5) mean((p >= thr) == (y == 1))
calc_auc <- function(y, p){
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(p, ties.method = "average")
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
calc_ece <- function(y, p, bins = 10){
  br <- cut(p, breaks = seq(0, 1, length.out = bins + 1), include.lowest = TRUE)
  s  <- data.frame(y, p, br) %>% group_by(br) %>% summarise(n = n(), conf = mean(p), acc = mean(y), .groups = "drop")
  sum(s$n / sum(s$n) * abs(s$acc - s$conf))
}
score_all <- function(label, y, p){
  data.frame(
    Set = label, N = length(y), Accuracy = round(calc_accuracy(y, p), 4),
    LogLoss = round(calc_logloss(y, p), 4), Brier = round(calc_brier(y, p), 4),
    AUC = round(calc_auc(y, p), 4), ECE = round(calc_ece(y, p), 4), stringsAsFactors = FALSE
  )
}

cat("=== MODEL PERFORMANCE: CALIBRATED VS. RAW (2025-2026 forward test) ===\n")
print(rbind(score_all("Calibrated", y_test, p_cal_test),
            score_all("Raw (Uncalibrated)", y_test, p_raw_test)), row.names = FALSE)

# ==============================================================================
# PHASE 8: FEATURE IMPORTANCE -- WHAT IS THE MODEL ACTUALLY USING?
# ==============================================================================
model_importance <- xgb.importance(model = base_model)
cat("\n=== FEATURE IMPORTANCE (base_model, no market feature) ===\n")
print(model_importance)
# Log-rank differential and surface-specific Elo dominate (~90%+ of total
# gain combined). Worth keeping in mind for what follows: both of these are
# themselves close reconstructions of information -- current ranking and
# recent surface form -- that a bookmaker's price should already reflect.

# ==============================================================================
# PHASE 9: DOES THE MODEL BEAT THE MARKET? -- HEAD-TO-HEAD BENCHMARK
# ==============================================================================
# The bookmaker's own vig-adjusted probability is the natural baseline: any
# model claiming to find betting value needs to clear this bar on proper
# scoring rules, not just on accuracy.

cat("\n=== MODEL VS. MARKET-IMPLIED PROBABILITY ===\n")
print(rbind(
  score_all("Market Odds Alone", y_test, test_pool_evaluated$Bookie_True_Prob_P1),
  score_all("Model (Calibrated)", y_test, p_cal_test)
), row.names = FALSE)
# Result: the market's own price outperforms the model on every metric
# (AUC ~0.75 vs ~0.72, logloss ~0.59 vs ~0.61). This alone doesn't rule out
# a narrow, valuable disagreement in specific situations -- that's tested
# directly in Phase 10.

# ==============================================================================
# PHASE 10: THE DECISIVE TEST -- HORSE-RACE REGRESSION
# ==============================================================================
# Regresses the match outcome on BOTH the market's implied logit and the
# model's calibrated logit simultaneously. This directly answers: does the
# model carry any predictive information the market doesn't already have?
# A model with genuine incremental value would show a positive, significant
# coefficient on Model_Logit even after the market's price is controlled for.

horse_race <- test_pool_evaluated %>%
  mutate(Bookie_Logit = qlogis(pmin(pmax(Bookie_True_Prob_P1, 1e-6), 1 - 1e-6)))

horse_race_fit <- glm(P1_Won ~ Bookie_Logit + Logit_P1, data = horse_race, family = binomial)
cat("\n=== HORSE RACE: MODEL vs. MARKET ===\n")
print(summary(horse_race_fit)$coefficients)
# Result: Bookie_Logit is highly significant (p < 0.001); Model_Logit is not
# distinguishable from zero (p ~ 0.14) and its point estimate is negative.
# Conditional on the market's price, the model adds no information.

# ==============================================================================
# PHASE 11: SEGMENTED HORSE-RACE -- HUNTING FOR POCKETS OF EDGE
# ==============================================================================
# The aggregate result could still mask a real edge in a specific slice of
# matches (e.g. thinly-traded tournaments, specific surfaces). Repeats the
# horse race within each segment to check.

run_horse_race <- function(df, label) {
  if (nrow(df) < 100) return(data.frame(Segment = label, N = nrow(df), Model_Coef = NA, Model_P = NA))
  df <- df %>% mutate(Bookie_Logit = qlogis(pmin(pmax(Bookie_True_Prob_P1, 1e-6), 1 - 1e-6)))
  fit <- tryCatch(glm(P1_Won ~ Bookie_Logit + Logit_P1, data = df, family = binomial), error = function(e) NULL)
  if (is.null(fit)) return(data.frame(Segment = label, N = nrow(df), Model_Coef = NA, Model_P = NA))
  co <- summary(fit)$coefficients
  data.frame(Segment = label, N = nrow(df),
             Model_Coef = round(co["Logit_P1", "Estimate"], 3),
             Model_P    = round(co["Logit_P1", "Pr(>|z|)"], 3))
}

segment_results <- bind_rows(
  test_pool_evaluated %>% filter(Surface == "Hard")  %>% run_horse_race("Surface: Hard"),
  test_pool_evaluated %>% filter(Surface == "Clay")  %>% run_horse_race("Surface: Clay"),
  test_pool_evaluated %>% filter(Surface == "Grass") %>% run_horse_race("Surface: Grass"),
  test_pool_evaluated %>% filter(Series == "Grand Slam") %>% run_horse_race("Series: Grand Slam"),
  test_pool_evaluated %>% filter(Series %in% c("Masters","Masters 1000")) %>% run_horse_race("Series: Masters"),
  test_pool_evaluated %>% filter(Series %in% c("ATP500","International Gold")) %>% run_horse_race("Series: ATP500"),
  test_pool_evaluated %>% filter(Series %in% c("ATP250","International")) %>% run_horse_race("Series: ATP250"),
  test_pool_evaluated %>% filter(Round %in% c("1st Round","2nd Round")) %>% run_horse_race("Round: Early"),
  test_pool_evaluated %>% filter(Round %in% c("Quarterfinals","Semifinals","The Final")) %>% run_horse_race("Round: Late (QF+)"),
  test_pool_evaluated %>% filter(Odd_1 < 1.5 | Odd_2 < 1.5) %>% run_horse_race("Has strong favorite (<1.5)"),
  test_pool_evaluated %>% filter(Odd_1 >= 1.5 & Odd_1 <= 2.5 & Odd_2 >= 1.5 & Odd_2 <= 2.5) %>% run_horse_race("Close match (1.5-2.5 both sides)")
)

cat("\n=== SEGMENTED HORSE-RACE: any segment with real model signal? ===\n")
print(segment_results, row.names = FALSE)
# Look for: Model_Coef > 0 AND Model_P < 0.05. No segment tested cleared this
# bar; the "close match" segment even showed a significant NEGATIVE
# coefficient, meaning the model's disagreement with the market in tight
# matches is actively anti-predictive, not a hidden source of value.

# ------------------------------------------------------------------------------
# Note on a market-aware model variant (not reproduced in full here):
# A second model was trained with the market's own implied probability
# (logit-transformed) added directly as a feature, to test whether the model
# could learn to correct residual market error rather than compete blind.
# Feature importance showed this single feature absorbed ~80% of the model's
# total gain, and a horse race against the market showed its residual signal
# was still not significant (p ~ 0.56). In short: giving the model the
# market's price caused it to mostly imitate that price rather than add
# anything beyond it -- consistent with the conclusion above.
# ------------------------------------------------------------------------------

# ==============================================================================
# PHASE 12: EV-THRESHOLD SWEEP -- PROPER METHODOLOGY WITH BOOTSTRAP CI
# ==============================================================================
# True expected value per unit staked: EV = (Model_Prob * Decimal_Odds) - 1.
# Threshold is swept on the CALIBRATION period only; the test period is
# touched exactly once, using whichever threshold the calibration period
# supports -- this avoids fitting the threshold to the answer key.

calib_pool <- calib_pool %>%
  mutate(
    Bookie_True_Prob_P2 = 1 - Bookie_True_Prob_P1,
    Calib_Prob_P1 = predict(calibration_fit, newdata = ., type = "response"),
    Calib_Prob_P2 = 1 - Calib_Prob_P1,
    EV_True_P1 = (Calib_Prob_P1 * Odd_1) - 1,
    EV_True_P2 = (Calib_Prob_P2 * Odd_2) - 1
  )

test_pool_evaluated <- test_pool_evaluated %>%
  mutate(
    Bookie_True_Prob_P2 = 1 - Bookie_True_Prob_P1,
    EV_True_P1 = (Calib_Prob_P1 * Odd_1) - 1,
    EV_True_P2 = (Calib_Prob_P2 * Odd_2) - 1
  )

evaluate_threshold <- function(df, thr, n_boot = 2000) {
  sel <- df %>% mutate(
    Bet_Side = case_when(EV_True_P1 > thr ~ "P1", EV_True_P2 > thr ~ "P2", TRUE ~ NA_character_),
    Bet_Odd  = case_when(Bet_Side == "P1" ~ Odd_1, Bet_Side == "P2" ~ Odd_2, TRUE ~ NA_real_),
    Bet_Won  = case_when(Bet_Side == "P1" ~ P1_Won, Bet_Side == "P2" ~ 1L - P1_Won, TRUE ~ NA_integer_),
    Profit   = ifelse(Bet_Won == 1L, Bet_Odd - 1, -1)
  ) %>% filter(!is.na(Bet_Side))

  if (nrow(sel) < 10) return(data.frame(Threshold = thr, N_Bets = nrow(sel), ROI_pct = NA, Lo95 = NA, Hi95 = NA))

  boots <- replicate(n_boot, mean(sample(sel$Profit, nrow(sel), replace = TRUE)))
  data.frame(Threshold = thr, N_Bets = nrow(sel),
             ROI_pct = round(100 * mean(sel$Profit), 2),
             Lo95 = round(100 * quantile(boots, 0.025), 2),
             Hi95 = round(100 * quantile(boots, 0.975), 2))
}

sweep_thresholds <- c(0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30)
calib_sweep <- bind_rows(lapply(sweep_thresholds, evaluate_threshold, df = calib_pool))
cat("\n=== EV THRESHOLD SWEEP -- CALIBRATION PERIOD (2023-2024), WITH BOOTSTRAP CI ===\n")
print(calib_sweep, row.names = FALSE)
# Result: ROI is negative at every threshold, generally worsening as the
# threshold rises, and every bootstrap CI spans a wide range including zero.
# There is no threshold here that clears validation.

# ==============================================================================
# PHASE 13: SYSTEMATIC ODDS-INTERVAL SWEEP -- IS ANY PRICE RANGE PROFITABLE?
# ==============================================================================
bootstrap_roi_ci <- function(profits, n_boot = 2000) {
  if (length(profits) < 5) return(c(Lo95 = NA_real_, Hi95 = NA_real_))
  boots <- replicate(n_boot, mean(sample(profits, length(profits), replace = TRUE)))
  # unname() is essential -- quantile() returns names like "2.5%", which
  # silently breaks name-based indexing (ci["Lo95"]) further down.
  out <- c(round(100 * unname(quantile(boots, 0.025)), 2),
           round(100 * unname(quantile(boots, 0.975)), 2))
  names(out) <- c("Lo95", "Hi95")
  out
}

build_favorite_bets <- function(df) {
  df %>%
    mutate(
      Pick_Side = ifelse(Calib_Prob_P1 >= Calib_Prob_P2, "P1", "P2"),
      Pick_Odd  = ifelse(Pick_Side == "P1", Odd_1, Odd_2),
      Pick_EV   = ifelse(Pick_Side == "P1", EV_True_P1, EV_True_P2),
      Pick_Won  = ifelse(Pick_Side == "P1", P1_Won, 1L - P1_Won)
    ) %>%
    filter(Pick_EV > 0) %>%
    mutate(Profit = ifelse(Pick_Won == 1L, Pick_Odd - 1, -1))
}

odds_interval_sweep <- function(df, breaks, labels) {
  bets <- build_favorite_bets(df) %>%
    mutate(Odd_Bucket = cut(Pick_Odd, breaks = breaks, labels = labels, include.lowest = TRUE)) %>%
    # cut() returns NA for odds outside the break range (e.g. > 3.00).
    # Drop those explicitly -- group_by() otherwise keeps NA as its own
    # group, which breaks group_modify()'s row-name handling.
    filter(!is.na(Odd_Bucket))
  
  bets %>%
    group_by(Odd_Bucket, .drop = FALSE) %>%
    group_modify(~{
      if (nrow(.x) == 0) {
        return(data.frame(N_Bets = 0L, Win_Rate = NA_real_, ROI_pct = NA_real_,
                          Lo95 = NA_real_, Hi95 = NA_real_))
      }
      ci <- bootstrap_roi_ci(.x$Profit)
      data.frame(N_Bets = nrow(.x), Win_Rate = round(mean(.x$Pick_Won), 4),
                 ROI_pct = round(100 * mean(.x$Profit), 2),
                 Lo95 = unname(ci["Lo95"]), Hi95 = unname(ci["Hi95"]))
    }) %>% ungroup()
}

interval_breaks <- c(1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.50, 3.00)
interval_labels <- c("1.00-1.20","1.21-1.40","1.41-1.60","1.61-1.80",
                     "1.81-2.00","2.01-2.50","2.51-3.00")

cat("\n=== ODDS-INTERVAL SWEEP -- CALIBRATION PERIOD (2023-2024) ===\n")
calib_interval_sweep <- odds_interval_sweep(calib_pool, interval_breaks, interval_labels)
print(calib_interval_sweep, n = Inf)

cat("\n=== ODDS-INTERVAL SWEEP -- FORWARD TEST (2025-2026) ===\n")
test_interval_sweep <- odds_interval_sweep(test_pool_evaluated, interval_breaks, interval_labels)
print(test_interval_sweep, n = Inf)

# Read this by checking Lo95 in EACH row: an interval only counts as a
# candidate edge if its lower bound clears zero on BOTH the calibration
# period and the forward test. A single positive point estimate with a CI
# spanning deeply negative values is not evidence of an edge -- it is the
# expected appearance of noise in a small sample.

# ==============================================================================
# CONCLUSION
# ==============================================================================
# Across a market benchmark, a horse-race regression, a segmented horse-race,
# a market-aware model variant, an EV-threshold sweep, and a systematic
# odds-interval sweep -- every test points to the same result: this feature
# set, built on public rank/Elo/momentum/tournament data, does not contain
# information beyond what is already priced into this bookmaker's closing
# odds. Feature importance confirms the model leans most heavily on
# surface-specific Elo and log-rank differential, both of which are
# themselves largely reconstructions of information the market already
# prices in via ranking and recent form.
#
# This is a legitimate and useful finding on its own. It rules out a class of
# strategies rather than failing to find one, and it points toward what WOULD
# be needed for a real edge: earlier (opening) lines before the market has
# fully priced in information, structurally less efficient markets (lower
# tiers, thinner liquidity), or genuinely new information not reflected in
# rank/Elo -- e.g. real-time injury or fatigue signals.
# ==============================================================================


