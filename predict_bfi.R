# ==============================================================================
# predict_bfi.R
#
# Predict Big Five (BFI-44) trait scores for each participant from a transcript
# of their conversation with an AI, using Claude.
#
# INPUT: one spreadsheet (.xlsx or .csv) with three columns, one row per turn:
#
#     participant_id | ai_message              | participant_message
#     -------------- | ----------------------- | -------------------------
#     101            | How are you doing?      | Pretty good thanks, you?
#     101            | What did you do today?  | Went for a long walk.
#     102            | How are you doing?      | Busy. Lots on.
#
#   Rows are grouped by participant and kept in the order they appear in the file,
#   so the file should already be in conversational order.
#
# OUTPUT: one .csv, one row per participant:
#
#     participant_id, extraversion, agreeableness, conscientiousness,
#     neuroticism, openness        (trait means on the 1-5 BFI scale)
#
# The model is asked to rate the 44 BFI items as if it were the participant; those
# ratings are then scored with the standard BFI-44 key (reverse-scored items are
# flipped, then items are averaged within trait).
#
# Requirements: httr2, jsonlite (and readxl only if the input is .xlsx)
#   install.packages(c("httr2", "jsonlite", "readxl"))
#
# One API call per participant. 
# ==============================================================================

rm(list = ls())

# Set your Anthropic API key before running:
#   Sys.setenv(ANTHROPIC_API_KEY = "sk-ant-...")

library(httr2)
library(jsonlite)

# ── Settings ──────────────────────────────────────────────────────────────────

INPUT_FILE  <- "Experiment1_data.xlsx"   # .xlsx or .csv
INPUT_SHEET <- "Conversations"           # ignored for .csv

# Column names in the input file.
COL_ID    <- "userid"                    # participant identifier
COL_AI    <- "GPTmessage"                # what the AI said
COL_HUMAN <- "usermessage"               # what the participant said

OUTPUT_FILE <- "bfi_predictions_test.csv"

MODEL       <- "claude-sonnet-4-6"
TEMPERATURE <- 1
MAX_TOKENS  <- 1200

# Set to a number to try a few participants first; NA runs all of them.
MAX_PARTICIPANTS <- NA

# TRUE builds the prompts and reports what would be sent, without calling the API
# or spending anything. Useful for checking that your file is read correctly.
DRY_RUN <- FALSE

# ── BFI-44 items and scoring key ──────────────────────────────────────────────
# Order matters: it defines item_1 ... item_44 in the prompt and the response.

bfi_items <- c(
  "Is talkative", "Tends to find fault with others", "Does a thorough job",
  "Is depressed, blue", "Is original, comes up with new ideas", "Is reserved",
  "Is helpful and unselfish with others", "Can be somewhat careless",
  "Is relaxed, handles stress well", "Is curious about many different things",
  "Is full of energy", "Starts quarrels with others", "Is a reliable worker",
  "Can be tense", "Is ingenious, a deep thinker", "Generates a lot of enthusiasm",
  "Has a forgiving nature", "Tends to be disorganized", "Worries a lot",
  "Has an active imagination", "Tends to be quiet", "Is generally trusting",
  "Tends to be lazy", "Is emotionally stable, not easily upset", "Is inventive",
  "Has an assertive personality", "Can be cold and aloof",
  "Perseveres until the task is finished", "Can be moody",
  "Values artistic, aesthetic experiences", "Is sometimes shy, inhibited",
  "Is considerate and kind to almost everyone", "Does things efficiently",
  "Remains calm in tense situations", "Prefers work that is routine",
  "Is outgoing, sociable", "Is sometimes rude to others",
  "Makes plans and follows through with them", "Gets nervous easily",
  "Likes to reflect, play with ideas", "Has few artistic interests",
  "Likes to cooperate with others", "Is easily distracted",
  "Is sophisticated in art, music, or literature"
)

scoring_key <- list(
  extraversion      = list(items = c(1, 6, 11, 16, 21, 26, 31, 36),
                           reverse = c(6, 21, 31)),
  agreeableness     = list(items = c(2, 7, 12, 17, 22, 27, 32, 37, 42),
                           reverse = c(2, 12, 27, 37)),
  conscientiousness = list(items = c(3, 8, 13, 18, 23, 28, 33, 38, 43),
                           reverse = c(8, 18, 23, 43)),
  neuroticism       = list(items = c(4, 9, 14, 19, 24, 29, 34, 39),
                           reverse = c(9, 24, 34)),
  openness          = list(items = c(5, 10, 15, 20, 25, 30, 35, 40, 41, 44),
                           reverse = c(35, 41))
)

# 44 ratings (1-5) -> five trait means. Reverse-keyed items become 6 - rating.
score_bfi <- function(ratings) {
  sapply(scoring_key, function(key) {
    vals <- ratings[key$items]
    vals[key$items %in% key$reverse] <- 6 - vals[key$items %in% key$reverse]
    mean(vals)
  })
}

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT <- paste0(
  "You are skilled at deeply inhabiting another person's perspective based on how ",
  "they express themselves in writing. When given samples of someone's language, ",
  "you can accurately simulate how that person thinks, feels, and would respond to ",
  "questions about themselves."
)

build_prompt <- function(transcript) {
  items_str <- paste(seq_along(bfi_items), bfi_items, sep = ". ", collapse = "\n")
  paste0(
    "Below is a conversation between an AI and a participant. ",
    "Use the AI's questions as context for interpreting the participant's replies, ",
    "but base your personality assessment solely on what the PARTICIPANT says - ",
    "their word choice, topics, emotional tone, and interpersonal style.\n\n",
    "CONVERSATION:\n", transcript, "\n\n",
    "---\n\n",
    "STEP 1 - Per-trait observations:\n",
    "For each of the five personality traits below, write one sentence describing ",
    "what the user's messages specifically reveal about that trait. ",
    "Be precise and evidence-based - cite actual language or behaviour from the transcript.\n\n",
    "  Extraversion: How talkative, energetic, assertive, and socially engaged is the user? ",
    "Note message length, enthusiasm, and how much they volunteer information unprompted.\n",
    "  Agreeableness: How cooperative, warm, and considerate is the user? ",
    "Agreeableness varies widely - look carefully for signs of LOW agreeableness too: ",
    "criticising others, bluntness, self-interest, pushback, or reluctance to cooperate. ",
    "Do not default to rating people as warm and agreeable.\n",
    "  Conscientiousness: How organised, thorough, and disciplined is the user? ",
    "Look for planning language, attention to detail, structured writing, ",
    "mentions of goals or routines, and whether they complete thoughts carefully.\n",
    "  Neuroticism: How emotionally reactive, anxious, or easily stressed is the user? ",
    "Look for worry expressions, self-doubt, negative affect, and emotional volatility.\n",
    "  Openness: How curious, imaginative, and intellectually engaged is the user? ",
    "Look for abstract language, creative expression, intellectual curiosity, ",
    "and willingness to explore tangents.\n\n",
    "STEP 2 - BFI-44 ratings:\n",
    "Imagining you ARE this user, rate how much you agree with each of the 44 statements ",
    "below. Use this scale:\n",
    "  1 = Disagree strongly\n",
    "  2 = Disagree a little\n",
    "  3 = Neither agree nor disagree\n",
    "  4 = Agree a little\n",
    "  5 = Agree strongly\n\n",
    "IMPORTANT RATING RULES:\n",
    "- Use the full 1-5 range. A realistic personality profile will have substantial ",
    "variation across items - ratings clustered around 3 are almost certainly wrong.\n",
    "- Do not hedge. If the evidence points clearly in one direction, commit to a 1 or 5.\n",
    "- Base every rating on what the transcript actually shows, not on what seems ",
    "socially desirable.\n\n",
    "Statements:\n", items_str, "\n\n",
    "Format your response as:\n",
    "- First, your five trait observations (plain text, one sentence each).\n",
    "- Then, on its own line, a single JSON object with keys item_1 through item_44, ",
    "each an integer 1-5. Example: {\"item_1\": 4, \"item_2\": 2, ...}\n",
    "Output the JSON on a single line starting with {."
  )
}

# ── API call and response parsing ─────────────────────────────────────────────

call_claude <- function(prompt, api_key) {
  resp <- request("https://api.anthropic.com/v1/messages") |>
    req_headers("x-api-key" = api_key,
                "anthropic-version" = "2023-06-01",
                "content-type" = "application/json") |>
    req_body_json(list(
      model       = MODEL,
      max_tokens  = MAX_TOKENS,
      temperature = TEMPERATURE,
      system      = SYSTEM_PROMPT,
      messages    = list(list(role = "user", content = prompt))
    )) |>
    req_retry(max_tries = 3, backoff = ~ 2) |>
    req_perform()

  body <- resp_body_json(resp)
  if (is.null(body$content) || length(body$content) == 0)
    stop("Empty response (stop_reason = ", body$stop_reason, ")")
  body$content[[1]]$text
}

# Pull the 44 ratings out of the JSON line the model returns.
parse_ratings <- function(text) {
  lines <- strsplit(text, "\n")[[1]]
  json  <- lines[grepl("^\\{", trimws(lines))]
  if (length(json) == 0) stop("No JSON object found in the response.")
  vals <- as.numeric(unlist(fromJSON(json[1])[paste0("item_", 1:44)]))
  if (length(vals) != 44 || any(is.na(vals))) stop("Did not get 44 valid ratings.")
  vals
}

# ── Read the conversations ────────────────────────────────────────────────────

if (!file.exists(INPUT_FILE)) stop("Cannot find ", INPUT_FILE)

if (grepl("\\.xlsx?$", INPUT_FILE)) {
  dat <- as.data.frame(readxl::read_excel(INPUT_FILE, sheet = INPUT_SHEET,
                                          .name_repair = "minimal"))
} else {
  dat <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)
}

missing <- setdiff(c(COL_ID, COL_AI, COL_HUMAN), names(dat))
if (length(missing) > 0)
  stop("Missing column(s): ", paste(missing, collapse = ", "),
       "\nColumns found: ", paste(names(dat), collapse = ", "))

dat <- dat[, c(COL_ID, COL_AI, COL_HUMAN)]
names(dat) <- c("id", "ai", "human")
dat$ai    <- ifelse(is.na(dat$ai), "", dat$ai)
dat$human <- ifelse(is.na(dat$human), "", dat$human)

ids <- unique(dat$id)
if (!is.na(MAX_PARTICIPANTS)) ids <- head(ids, MAX_PARTICIPANTS)

cat(sprintf("%d participants, %d conversation turns\n", length(ids), nrow(dat)))

# One transcript per participant: the turns in file order.
transcript_for <- function(id) {
  rows <- dat[dat$id == id, ]
  paste(sprintf("AI: %s\nParticipant: %s", rows$ai, rows$human), collapse = "\n\n")
}

# ── Predict ───────────────────────────────────────────────────────────────────

api_key <- Sys.getenv("ANTHROPIC_API_KEY")
if (!DRY_RUN && !nzchar(api_key))
  stop("Set ANTHROPIC_API_KEY before running (or set DRY_RUN <- TRUE to test the input).")

traits  <- names(scoring_key)
results <- data.frame(participant_id = ids)
results[traits] <- NA_real_

# Resume: carry over any participant already scored in a previous run. Rows whose
# scores are still NA (a failed call) are retried, so re-running fills the gaps.
if (!DRY_RUN && file.exists(OUTPUT_FILE)) {
  prev <- read.csv(OUTPUT_FILE, stringsAsFactors = FALSE)
  if (all(c("participant_id", traits) %in% names(prev))) {
    hit <- match(results$participant_id, prev$participant_id)
    done <- !is.na(hit) & !is.na(prev[hit, "extraversion"])
    results[done, traits] <- prev[hit[done], traits]
    cat(sprintf("Resuming from %s: %d already scored, %d to do\n",
                OUTPUT_FILE, sum(done), sum(!done)))
  }
}

for (i in seq_along(ids)) {
  id <- ids[i]
  if (!DRY_RUN && !is.na(results[i, "extraversion"])) next   # already scored
  prompt <- build_prompt(transcript_for(id))

  if (DRY_RUN) {
    cat(sprintf("[%d/%d] %s | %d turns | prompt %d characters (dry run)\n",
                i, length(ids), id, sum(dat$id == id), nchar(prompt)))
    next
  }

  cat(sprintf("[%d/%d] %s ... ", i, length(ids), id))
  scores <- tryCatch({
    score_bfi(parse_ratings(call_claude(prompt, api_key)))
  }, error = function(e) {
    cat("FAILED:", conditionMessage(e), "\n")
    NULL
  })

  if (!is.null(scores)) {
    results[i, traits] <- scores
    cat(paste(sprintf("%s %.2f", substr(traits, 1, 1), scores), collapse = "  "), "\n")
  }
  write.csv(results, OUTPUT_FILE, row.names = FALSE)   # save after every call
  Sys.sleep(0.5)                                      
}

# ── Save ──────────────────────────────────────────────────────────────────────

if (!DRY_RUN) {
  write.csv(results, OUTPUT_FILE, row.names = FALSE)
  ok <- sum(!is.na(results$extraversion))
  cat(sprintf("\nSaved %s (%d of %d participants scored)\n",
              OUTPUT_FILE, ok, nrow(results)))
  if (ok < nrow(results))
    cat("Run the script again to retry the ones that failed.\n")
}
