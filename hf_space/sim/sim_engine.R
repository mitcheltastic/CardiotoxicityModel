# sim_engine.R
# Single-cell (fixed-parameter) CiPA AP simulation for Cardivex.
# Logic is a faithful port of FDA/CiPA chantest_AP_simulation/AP_simulation.R
# (Lab_Specific_Validation_Calibration_2020 branch) and funs/get_pars.R,
# restructured as a function that takes parameters directly instead of
# reading per-drug result folders.
#
# Usage: Rscript sim_engine.R <input.json> <output.json>
#   input.json: {"cmax_nM":..., "dose":..., "herg":{Kmax,Ku,n,halfmax,Vhalf},
#                "ic50":{INa_IC50,INa_h,INaL_IC50,INaL_h,ICaL_IC50,ICaL_h}}
#   cmax_nM = 0 or "control": true runs the drug-free control.

suppressMessages({ library(deSolve); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript sim_engine.R <input.json> <output.json>")
args[1] <- normalizePath(args[1], mustWork = TRUE)
args[2] <- file.path(normalizePath(dirname(args[2]), mustWork = TRUE), basename(args[2]))
script_dir <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
setwd(script_dir)

FEATURES <- c("qNet","dvdtmax","vmax","vrest","APD50","APD90","max_dv","camax","carest","CaTD50","CaTD90")
CL <- 2000; BEATS <- 1000; NSAVE <- 250

ext <- ifelse(Sys.info()[["sysname"]] == "Windows", ".dll", ".so")
dyn.load(paste0("models/newordherg_qNet", ext))

find_dxdtMax <- function(t, x){ d <- diff(x)/diff(t); i <- which.max(d)[1]; list(value=d[i], t=t[i]) }
find_max <- function(t, x){ i <- which.max(x)[1]; list(value=x[i], t=t[i]) }
find_rep <- function(t, x, xrest, idx_max, pct){
  xrep <- xrest + (x[idx_max]-xrest)*(1-pct)
  list(value=xrep, t=t[which(x <= xrep & t > t[idx_max])[1]])
}

# Mirrors get_pars(): fixed (isamp = 0) parameters, optimized scaling on.
build_pars <- function(conc, herg, ic50, control){
  tmp <- read.table("models/newordherg_pars.txt", col.names=c("param","value"))
  pars <- setNames(tmp$value, tmp$param)
  pars["GKrfc"] <- 1.1
  pars["T"] <- 37
  if (control) {
    pars["Kmax"] <- 0; pars["Kt"] <- 0
  } else {
    for (nm in c("Kmax","Ku","n","halfmax","Vhalf")) pars[nm] <- as.numeric(herg[[nm]])
    pars["Kt"] <- 3.5e-5
    if (conc > 0) {
      sc <- function(ch) {
        v <- ic50[[paste0(ch,"_IC50")]]; h <- ic50[[paste0(ch,"_h")]]
        if (is.null(v) || is.null(h) || is.na(v) || is.na(h)) return(NA)
        1 + (conc/as.numeric(v))^as.numeric(h)
      }
      s <- sc("INa");  if (!is.na(s)) pars["GNafc"]  <- s
      s <- sc("INaL"); if (!is.na(s)) pars["GNaLfc"] <- s
      s <- sc("ICaL"); if (!is.na(s)) pars["PCafc"]  <- s
    }
  }
  pars["GKrfc"]  <- pars["GKrfc"]/1.114
  pars["GNaLfc"] <- pars["GNaLfc"]/2.661
  pars["GKsfc"]  <- pars["GKsfc"]/1.870
  pars["GK1fc"]  <- pars["GK1fc"]/1.698
  pars["PCafc"]  <- pars["PCafc"]/1.007
  pars
}

simulate <- function(conc, herg, ic50, control){
  pars <- build_pars(conc, herg, ic50, control)
  tmp <- read.table(sprintf("models/newordherg_states_CL%d.txt", CL), col.names=c("param","value"))
  states <- setNames(tmp$value, tmp$param)
  states[["D"]] <- conc
  states <- c(states, qNet=0)
  fulltimes <- 0:CL
  params <- c("INa","INaL","Ito","ICaL","IKr","IKs","IK1","dv")
  beatdf <- data.frame(); traces <- list()

  for (i in 1:BEATS) {
    out <- ode(states, fulltimes, "derivs", pars, dllname="newordherg_qNet", initfunc="initmod",
               nout=length(params), rtol=1e-6, atol=1e-6, method="lsoda")
    if (nrow(out) != length(fulltimes) || any(is.nan(out))) stop("solving error at beat ", i)
    colnames(out)[1+length(states)+seq_along(params)] <- params

    if (i > BEATS-NSAVE) {
      tmpdf <- data.frame(beat=i, qNet=out[nrow(out),"qNet"])
      catd50 <- NA; catd90 <- NA; apd50 <- NA; apd90 <- NA
      t <- out[,"time"]; v <- out[,"v"]; ca <- out[,"cai"]
      t_start <- t[length(t)]; t_end <- t[length(t)]
      p <- find_dxdtMax(t, v); t_up <- p$t; tmpdf$dvdtmax <- p$value
      p <- find_max(t, v); t_max <- p$t; vmax <- p$value; i_max <- which(t == t_max)
      tmpdf$vmax <- vmax
      if (vmax >= 0) t_start <- t_max
      vrest <- min(v[t < t_max]); tmpdf$vrest <- vrest
      if (vmax >= 0 && vrest < -40) {
        t_rep <- find_rep(t, v, vrest, i_max, 0.3)$t
        if (!is.na(t_rep)) {
          t_start <- t_rep
          t_rep <- find_rep(t, v, vrest, i_max, 0.5)$t
          if (!is.na(t_rep)) {
            apd50 <- t_rep - t_up
            t_rep <- find_rep(t, v, vrest, i_max, 0.9)$t
            if (!is.na(t_rep)) { apd90 <- t_rep - t_up; t_end <- t_rep }
          }
        }
      }
      tmpdf$APD50 <- apd50; tmpdf$APD90 <- apd90
      tmpdf$max_dv <- ifelse(t_end - t_start > 1, max(out[t > t_start & t < t_end, "dv"]), NA)
      p <- find_max(t, ca); t_max <- p$t; camax <- p$value; i_max <- which(t == t_max)
      tmpdf$camax <- camax
      carest <- min(ca[t < t_max]); tmpdf$carest <- carest
      caup <- carest + (camax-carest)*0.9
      t_up <- t[which(ca > caup)[1]]
      t_rep <- find_rep(t, ca, carest, i_max, 0.5)$t
      if (!is.na(t_rep)) {
        catd50 <- t_rep - t_up
        t_rep <- find_rep(t, ca, carest, i_max, 0.9)$t
        if (!is.na(t_rep)) catd90 <- t_rep - t_up
      }
      tmpdf$CaTD50 <- catd50; tmpdf$CaTD90 <- catd90
      beatdf <- rbind(beatdf, tmpdf)
      keep_idx <- seq(1, length(t), by=2)
      traces[[i-(BEATS-NSAVE)]] <- list(time=t[keep_idx], voltage=v[keep_idx], cai=ca[keep_idx])
    }
    pidx <- match(names(states), colnames(out), nomatch=0)
    states[pidx != 0] <- out[nrow(out), pidx]
    states[length(states)] <- 0
  }

  max_vec <- beatdf$max_dv
  nadv <- is.na(max_vec)
  keep <- if (all(nadv)) nrow(beatdf) else which.max(max_vec)
  list(metrics = as.list(beatdf[keep, FEATURES]),
       beat = beatdf$beat[keep],
       EADs = sum(max_vec[!nadv] > 0),
       noDepols = sum(nadv),
       noRepols = sum(is.na(beatdf$APD90)),
       trace = traces[[keep]])
}

main <- function(){
  inp <- fromJSON(args[1], simplifyVector = TRUE)
  control <- isTRUE(inp$control)
  conc <- if (control) 0 else as.numeric(inp$cmax_nM) * as.numeric(inp$dose)
  t0 <- proc.time()[["elapsed"]]
  res <- simulate(conc, inp$herg, inp$ic50, control)
  res$conc_nM <- conc
  res$elapsed_s <- round(proc.time()[["elapsed"]] - t0, 1)
  write(toJSON(res, auto_unbox=TRUE, digits=NA, na="null"), args[2])
}
main()
