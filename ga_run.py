import os
import sys
import csv
import functools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ga_main import genetic_algorithm, POPULATION_SIZE

FIXED_POPULATION_SIZE = 70
FIXED_GENERATIONS     = 50
N_RUNS                = 10

BASELINE = {
    "mutation_rate"  : 0.10,
    "crossover_rate" : 0.70,
}

SENSITIVITY_GRIDS = {
    "mutation_rate"  : [0.10, 0.20, 0.30],
    "crossover_rate" : [0.70, 0.80, 0.90],
}

DIR_CONVERGENCE = "output_convergence"
DIR_SENSITIVITY = "output_sensitivity"
DIR_PSO         = "output_pso"

for _d in [DIR_CONVERGENCE, DIR_SENSITIVITY, DIR_PSO]:
    os.makedirs(_d, exist_ok=True)




def _run_one_trial(mutation_rate, crossover_rate):

    import ga_main as _m
    from ga_main import (init_population, fitness as _fitness,
                           check, roulette_wheel, binary_to_desimal)

    orig_crossover = _m.crossover
    orig_mutation  = _m.mutation
    _m.crossover   = functools.partial(orig_crossover, crossover_rate=crossover_rate)
    _m.mutation    = functools.partial(orig_mutation,  mutation_rate=mutation_rate)

    devnull = open(os.devnull, 'w')
    sys.stdout = devnull

    try:
        pop_size    = FIXED_POPULATION_SIZE
        _generations = FIXED_GENERATIONS

        population_val = _fitness(init_population(pop_size))

        iteration      = 0
        fitness_hist   = []
        cappv_hist     = []
        ebess_hist     = []
        tc_hist        = []
        cpv_hist       = []
        cbess_hist     = []

        while iteration < _generations:
            iteration += 1
            result = check(population_val)

            if result is False:
                population_val = _fitness(init_population(pop_size))
                _generations  += 1
                continue

            population_check = result
            new_population   = []

            ix = -1
            ix, p1 = roulette_wheel(population_check, ix)
            ix, p2 = roulette_wheel(population_check, ix)

            while len(new_population) < pop_size // 2:
                o1c, o2c, o1e, o2e = _m.crossover(p1, p2)
                o1c = _m.mutation(o1c)
                o2c = _m.mutation(o2c)
                o1e = _m.mutation(o1e)
                o2e = _m.mutation(o2e)

                d1c = binary_to_desimal(o1c); d2c = binary_to_desimal(o2c)
                d1e = binary_to_desimal(o1e); d2e = binary_to_desimal(o2e)

                nc1 = dict(cappv_bits=o1c, cappv_desimal=d1c,
                           ebess_bits=o1e, ebess_desimal=d1e)
                nc2 = dict(cappv_bits=o2c, cappv_desimal=d2c,
                           ebess_bits=o2e, ebess_desimal=d2e)

                tmp   = _fitness([nc1, nc2])
                valid = check(tmp)

                if valid is not False:
                    new_population += [nc1, nc2]
                    new_population_val = _fitness(new_population)
                    iy = -1
                    iy, p1 = roulette_wheel(new_population_val, iy)
                    iy, p2 = roulette_wheel(new_population_val, iy)

            population_val = _fitness(new_population)

            def _rv(v):
                return float(v.real if hasattr(v, 'real') else v)

            best_ch = min(population_val,
                          key=lambda c: _rv(c['fitness']))

            cur_f = _rv(best_ch['fitness'])

            if not fitness_hist or cur_f <= fitness_hist[-1]:
                fitness_hist.append(cur_f)
                cappv_hist.append(_rv(best_ch['cappv_desimal']))
                ebess_hist.append(_rv(best_ch['ebess_desimal']))
                tc_hist.append(_rv(best_ch['TC']))
                cpv_hist.append(_rv(best_ch['cpv']))
                cbess_hist.append(_rv(best_ch['cbess']))
            else:
                fitness_hist.append(fitness_hist[-1])
                cappv_hist.append(cappv_hist[-1])
                ebess_hist.append(ebess_hist[-1])
                tc_hist.append(tc_hist[-1])
                cpv_hist.append(cpv_hist[-1])
                cbess_hist.append(cbess_hist[-1])

    finally:
        sys.stdout = sys.__stdout__
        devnull.close()
        _m.crossover = orig_crossover
        _m.mutation  = orig_mutation

    best_fitness = fitness_hist[-1]
    best_cappv   = cappv_hist[-1]
    best_ebess   = ebess_hist[-1]
    best_tc      = tc_hist[-1]

    return best_fitness, best_cappv, best_ebess, best_tc, fitness_hist


def run_main_convergence(filename="aci_1year"):

    print(f"CONVERGENCE RUN  [{filename}]")
    print(f"mutation_rate={BASELINE['mutation_rate']}  "
          f"crossover_rate={BASELINE['crossover_rate']}  "
          f"pop={FIXED_POPULATION_SIZE}  gen={FIXED_GENERATIONS}")

    log_path = os.path.join(DIR_CONVERGENCE, f"{filename}_log.txt")
    orig_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as f:
        sys.stdout = f
        try:
            fitness_all, cappv_all, ebess_all, total_cost_all, cpv_all, cbess_all = \
                genetic_algorithm(POPULATION_SIZE)
        finally:
            sys.stdout = orig_stdout

    def _rv(v):
        return float(v.real if hasattr(v, 'real') else v)

    csv_path = os.path.join(DIR_CONVERGENCE, f"convergence_{filename}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Iteration", "Fitness", "CAPPV", "CPV",
                    "EBESS", "CBESS", "Total_Cost"])
        for i in range(len(fitness_all)):
            w.writerow([i + 1,
                        round(_rv(fitness_all[i]),    4),
                        round(_rv(cappv_all[i]),      4),
                        round(_rv(cpv_all[i]),        4),
                        round(_rv(ebess_all[i]),      4),
                        round(_rv(cbess_all[i]),      4),
                        round(_rv(total_cost_all[i]), 4)])

    graph_path = os.path.join(DIR_CONVERGENCE, f"convergence_{filename}.jpg")
    _plot_convergence(fitness_all, cappv_all, ebess_all, total_cost_all,
                      title=f"GA Convergence — {filename} "
                            f"(mr={BASELINE['mutation_rate']}, "
                            f"cr={BASELINE['crossover_rate']})",
                      save_path=graph_path)

    print(f"Log              → {log_path}")
    print(f"Convergence CSV  → {csv_path}")
    print(f"Convergence Graph→ {graph_path}")
    print(f"Best Fitness     = {round(_rv(fitness_all[-1]),    4)}")
    print(f"Best CAPPV       = {round(_rv(cappv_all[-1]),      4)}")
    print(f"Best EBESS       = {round(_rv(ebess_all[-1]),      4)}")
    print(f"Best Total Cost  = {round(_rv(total_cost_all[-1]), 4)}")

    return fitness_all, cappv_all, ebess_all, total_cost_all, cpv_all, cbess_all


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4.2 — SENSITIVITY ANALYSIS
#  Only mutation_rate and crossover_rate are swept.
#  Each configuration is run N_RUNS=10 times (No Free Lunch principle).
# ═══════════════════════════════════════════════════════════════════

def run_sensitivity_analysis():
    print(f"Fixed: population_size={FIXED_POPULATION_SIZE}, "
          f"generations={FIXED_GENERATIONS}")
    print(f"Swept: mutation_rate, crossover_rate")
    print(f"Runs per config: {N_RUNS}  (No Free Lunch)")
    print(f"{'='*60}")


    all_results  = {}
    summary_rows = []

    for param_name, grid in SENSITIVITY_GRIDS.items():
        print(f"\n  ▶ Sweeping: {param_name}  grid={grid}")
        param_results = []

        for val in grid:
            cfg = dict(BASELINE)
            cfg[param_name] = val

            run_fitnesses  = []
            run_cappvs     = []
            run_ebesses    = []
            run_tcs        = []
            run_histories  = []

            for run_idx in range(1, N_RUNS + 1):
                print(f"     {param_name}={val}  run {run_idx}/{N_RUNS} ...",
                      end=" ", flush=True)
                bf, bc, be, bt, hist = _run_one_trial(**cfg)
                run_fitnesses.append(bf)
                run_cappvs.append(bc)
                run_ebesses.append(be)
                run_tcs.append(bt)
                run_histories.append(hist)
                print(f"fitness={bf:.4f}")

            # aggregate across N_RUNS
            mean_f = float(np.mean(run_fitnesses))
            std_f  = float(np.std(run_fitnesses, ddof=1))
            min_f  = float(np.min(run_fitnesses))
            max_f  = float(np.max(run_fitnesses))

            mean_cappv = float(np.mean(run_cappvs))
            mean_ebess = float(np.mean(run_ebesses))
            mean_tc    = float(np.mean(run_tcs))

            # median-length history padded/trimmed to same length for plotting
            target_len   = int(np.median([len(h) for h in run_histories]))
            norm_hists   = [_pad_history(h, target_len) for h in run_histories]
            mean_history = list(np.mean(norm_hists, axis=0))
            std_history  = list(np.std(norm_hists,  axis=0, ddof=1))

            entry = {
                "param_name"     : param_name,
                "param_value"    : val,
                "mean_fitness"   : mean_f,
                "std_fitness"    : std_f,
                "min_fitness"    : min_f,
                "max_fitness"    : max_f,
                "mean_cappv"     : mean_cappv,
                "mean_ebess"     : mean_ebess,
                "mean_total_cost": mean_tc,
                "mean_history"   : mean_history,
                "std_history"    : std_history,
                "all_histories"  : norm_hists,
            }
            param_results.append(entry)
            summary_rows.append({k: v for k, v in entry.items()
                                  if k not in ("mean_history",
                                               "std_history",
                                               "all_histories")})

        all_results[param_name] = param_results

        csv_path = os.path.join(DIR_SENSITIVITY, f"sensitivity_{param_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["param_name", "param_value",
                          "mean_fitness", "std_fitness",
                          "min_fitness",  "max_fitness",
                          "mean_cappv", "mean_ebess", "mean_total_cost"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in param_results:
                w.writerow({k: round(v, 6) if isinstance(v, float) else v
                             for k, v in r.items() if k in fieldnames})
        print(f"\nCSV saved → {csv_path}")

        graph_path = os.path.join(DIR_SENSITIVITY, f"sensitivity_{param_name}.jpg")
        _plot_sensitivity_param(param_name, grid, param_results, graph_path)
        print(f"Graph saved → {graph_path}")

    master_csv = os.path.join(DIR_SENSITIVITY, "sensitivity_summary.csv")
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["param_name", "param_value",
                      "mean_fitness", "std_fitness",
                      "min_fitness",  "max_fitness",
                      "mean_cappv", "mean_ebess", "mean_total_cost"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v
                         for k, v in r.items() if k in fieldnames})
    print(f"\nMaster summary CSV → {master_csv}")

    overview_path = os.path.join(DIR_SENSITIVITY, "sensitivity_overview.jpg")
    _plot_sensitivity_overview(all_results, overview_path)
    print(f"Overview graph    → {overview_path}")



def _pad_history(hist, target_len):
    """Trim or pad a fitness history list to exactly target_len entries."""
    if len(hist) >= target_len:
        return hist[:target_len]
    return hist + [hist[-1]] * (target_len - len(hist))


def _plot_convergence(fitness_all, cappv_all, ebess_all, total_cost_all,
                      title, save_path):
    def _real(arr):
        return [float(v.real if hasattr(v, 'real') else v) for v in arr]

    fa  = _real(fitness_all)
    ca  = _real(cappv_all)
    ea  = _real(ebess_all)
    tca = _real(total_cost_all)
    x   = list(range(1, len(fa) + 1))

    fig, axs = plt.subplots(4, 1, figsize=(12, 14))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    specs = [
        (fa,  "Fitness",    "Fitness Value",   "#E63946"),
        (ca,  "CAPPV",      "CAPPV (W)",       "#2A9D8F"),
        (ea,  "EBESS",      "EBESS (Wh)",      "#E9C46A"),
        (tca, "Total Cost", "Cost (currency)", "#457B9D"),
    ]
    for ax, (data, label, ylabel, color) in zip(axs, specs):
        ax.plot(x, data, color=color, linewidth=1.8, marker='o',
                markersize=3, markevery=max(1, len(x) // 20))
        ax.fill_between(x, data, alpha=0.12, color=color)
        ax.set_title(f"{label} Over Generations", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Generation", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.annotate(f"Final: {data[-1]:.2f}",
                    xy=(x[-1], data[-1]),
                    xytext=(-60, 10), textcoords='offset points',
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_sensitivity_param(param_name, grid, results, save_path):
    """
    Three panels per parameter:
      Left  : bar chart of mean ± std best fitness per grid value
      Middle: convergence curves (mean ± std band) per grid value
      Right : min/max range plot (robustness view)
    """
    means  = [r["mean_fitness"] for r in results]
    stds   = [r["std_fitness"]  for r in results]
    mins   = [r["min_fitness"]  for r in results]
    maxs   = [r["max_fitness"]  for r in results]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(grid)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Sensitivity Analysis — {param_name}\n"
        f"(pop={FIXED_POPULATION_SIZE}, gen={FIXED_GENERATIONS}, "
        f"N_runs={N_RUNS} per config — No Free Lunch)",
        fontsize=12, fontweight='bold'
    )

    # panel 1: mean ± std bar chart
    x_pos = np.arange(len(grid))
    bars  = ax1.bar(x_pos, means, yerr=stds, color=colors,
                    edgecolor='black', linewidth=0.6,
                    error_kw=dict(ecolor='black', capsize=5, linewidth=1.2))
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(v) for v in grid])
    ax1.set_title(f"Mean ± Std Best Fitness\n({N_RUNS} runs each)")
    ax1.set_xlabel(param_name)
    ax1.set_ylabel("Best Fitness")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + (max(means) * 0.005),
                 f"{m:.2f}", ha='center', va='bottom', fontsize=7.5)

    # Highlight best mean bar
    best_idx = int(np.argmin(means))
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2.2)
    ax1.annotate("Best\nmean",
                 xy=(x_pos[best_idx], means[best_idx]),
                 xytext=(0, 28), textcoords='offset points',
                 ha='center', fontsize=8, color='red',
                 arrowprops=dict(arrowstyle='->', color='red'))

    # baseline vertical line
    if BASELINE[param_name] in grid:
        bl_idx = grid.index(BASELINE[param_name])
        ax1.axvline(bl_idx, color='gray', linestyle=':', linewidth=1.5)
        ax1.text(bl_idx + 0.05, ax1.get_ylim()[0], "baseline",
                 fontsize=7.5, color='gray', va='bottom')

    # panel 2: mean convergence + std band
    for r, val, col in zip(results, grid, colors):
        mh  = r["mean_history"]
        sh  = r["std_history"]
        mh_arr = np.array(mh)
        sh_arr = np.array(sh)
        xs  = list(range(1, len(mh) + 1))
        ax2.plot(xs, mh_arr, label=f"{param_name}={val}",
                 color=col, linewidth=1.6)
        ax2.fill_between(xs, mh_arr - sh_arr, mh_arr + sh_arr,
                         alpha=0.15, color=col)

    ax2.set_title(f"Mean Convergence ± Std\n({N_RUNS} runs, shaded = ±1σ)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Best Fitness")
    ax2.legend(fontsize=7.5, loc='upper right')
    ax2.grid(linestyle='--', alpha=0.4)

    # panel 3: min–max robustness range
    for i, (val, col, mn, mx, me) in enumerate(
            zip(grid, colors, mins, maxs, means)):
        ax3.plot([val, val], [mn, mx], color=col, linewidth=3, solid_capstyle='round')
        ax3.scatter([val], [me], color=col, s=60, zorder=5)
        ax3.scatter([val], [mn], marker='^', color=col, s=40, zorder=5)
        ax3.scatter([val], [mx], marker='v', color=col, s=40, zorder=5)

    ax3.set_title(f"Min / Mean / Max Fitness\n(robustness across {N_RUNS} runs)")
    ax3.set_xlabel(param_name)
    ax3.set_ylabel("Best Fitness")
    ax3.grid(linestyle='--', alpha=0.4)

    if BASELINE[param_name] in grid:
        ax3.axvline(BASELINE[param_name], color='gray',
                    linestyle=':', linewidth=1.5, label='baseline')
        ax3.legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_sensitivity_overview(all_results, save_path):
    """
    1×2 overview: mean ± std fitness vs. parameter value for both swept params.
    Makes it easy to compare sensitivity of mutation_rate vs. crossover_rate.
    """
    params  = list(SENSITIVITY_GRIDS.keys())   # exactly 2
    palette = ["#E63946", "#2A9D8F"]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Sensitivity Overview — Mutation Rate vs Crossover Rate\n"
        f"(pop={FIXED_POPULATION_SIZE}, gen={FIXED_GENERATIONS}, fixed | "
        f"{N_RUNS} runs/config — No Free Lunch)",
        fontsize=12, fontweight='bold'
    )

    for ax, param, color in zip(axs, params, palette):
        results = all_results[param]
        vals    = [r["param_value"]  for r in results]
        means   = [r["mean_fitness"] for r in results]
        stds    = [r["std_fitness"]  for r in results]
        means_a = np.array(means)
        stds_a  = np.array(stds)

        ax.plot(vals, means_a, marker='o', color=color, linewidth=2,
                markersize=7, markerfacecolor='white', markeredgewidth=2,
                label='mean fitness')
        ax.fill_between(vals, means_a - stds_a, means_a + stds_a,
                        alpha=0.18, color=color, label='±1σ')

        ax.set_title(f"Effect of {param}", fontsize=11)
        ax.set_xlabel(param, fontsize=9)
        ax.set_ylabel("Best Fitness (mean ± std)", fontsize=9)
        ax.grid(linestyle='--', alpha=0.4)
        ax.legend(fontsize=8)

        baseline_val = BASELINE[param]
        if baseline_val in vals:
            ax.axvline(baseline_val, color='gray', linestyle=':', linewidth=1.4)
            bl_y = means[vals.index(baseline_val)]
            ax.annotate("baseline",
                        xy=(baseline_val, bl_y),
                        xytext=(8, -14), textcoords='offset points',
                        fontsize=7.5, color='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  MULTI-RUN HELPERS  (GA & PSO — N_RUNS each for fair comparison)
# ═══════════════════════════════════════════════════════════════════

def _run_one_trial_pso():
    from pso_main import pso, N_PARTICLES

    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    try:
        fitness_hist, cappv_hist, ebess_hist, tc_hist, _, _ = pso(N_PARTICLES)
    finally:
        sys.stdout = sys.__stdout__
        devnull.close()

    return fitness_hist[-1], cappv_hist[-1], ebess_hist[-1], tc_hist[-1], fitness_hist


def _multirun_stats(run_fitnesses, run_cappvs, run_ebesses, run_tcs, run_histories):
    mean_f     = float(np.mean(run_fitnesses))
    std_f      = float(np.std(run_fitnesses,  ddof=1))
    min_f      = float(np.min(run_fitnesses))
    max_f      = float(np.max(run_fitnesses))
    mean_cappv = float(np.mean(run_cappvs))
    mean_ebess = float(np.mean(run_ebesses))
    mean_tc    = float(np.mean(run_tcs))

    target_len   = int(np.median([len(h) for h in run_histories]))
    norm_hists   = [_pad_history(h, target_len) for h in run_histories]
    mean_history = list(np.mean(norm_hists, axis=0))
    std_history  = list(np.std(norm_hists,  axis=0, ddof=1))

    return mean_history, std_history, {
        "mean_fitness": mean_f, "std_fitness": std_f,
        "min_fitness":  min_f,  "max_fitness":  max_f,
        "mean_cappv":   mean_cappv,
        "mean_ebess":   mean_ebess,
        "mean_tc":      mean_tc,
        "all_histories": norm_hists,
    }


def _save_multirun_csv(path, run_fitnesses, run_cappvs, run_ebesses, run_tcs, stats):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Run", "Best_Fitness", "Best_CAPPV", "Best_EBESS", "Best_TC"])
        for i, (bf, bc, be, bt) in enumerate(
                zip(run_fitnesses, run_cappvs, run_ebesses, run_tcs), 1):
            w.writerow([i, round(bf, 4), round(bc, 4), round(be, 4), round(bt, 4)])
        w.writerow(["MEAN", round(stats["mean_fitness"], 4),
                    round(stats["mean_cappv"],   4),
                    round(stats["mean_ebess"],   4),
                    round(stats["mean_tc"],      4)])
        w.writerow(["STD",  round(stats["std_fitness"], 4), "", "", ""])
        w.writerow(["MIN",  round(stats["min_fitness"], 4), "", "", ""])
        w.writerow(["MAX",  round(stats["max_fitness"], 4), "", "", ""])


def run_ga_multiple(filename="ga_aci_multirun"):
    print(f"GA MULTI-RUN  [{filename}]")
    print(f"mutation_rate={BASELINE['mutation_rate']}  "
          f"crossover_rate={BASELINE['crossover_rate']}  "
          f"pop={FIXED_POPULATION_SIZE}  gen={FIXED_GENERATIONS}  N_RUNS={N_RUNS}")
    print(f"{'='*60}")

    run_fitnesses, run_cappvs, run_ebesses, run_tcs, run_histories = [], [], [], [], []

    for run_idx in range(1, N_RUNS + 1):
        print(f"  run {run_idx}/{N_RUNS} ...", end=" ", flush=True)
        bf, bc, be, bt, hist = _run_one_trial(**BASELINE)
        run_fitnesses.append(bf);  run_cappvs.append(bc)
        run_ebesses.append(be);    run_tcs.append(bt)
        run_histories.append(hist)
        print(f"fitness={bf:.4f}")

    mean_history, std_history, stats = _multirun_stats(
        run_fitnesses, run_cappvs, run_ebesses, run_tcs, run_histories)

    csv_path = os.path.join(DIR_CONVERGENCE, f"ga_multirun_{filename}.csv")
    _save_multirun_csv(csv_path, run_fitnesses, run_cappvs,
                       run_ebesses, run_tcs, stats)

    print(f"\nCSV  → {csv_path}")
    print(f"Mean Fitness = {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")
    print(f"Min = {stats['min_fitness']:.4f}  Max = {stats['max_fitness']:.4f}")
    print(f"Mean CAPPV   = {stats['mean_cappv']:.4f}")
    print(f"Mean EBESS   = {stats['mean_ebess']:.4f}")
    print(f"Mean TC      = {stats['mean_tc']:.4f}")

    return mean_history, std_history, stats


def run_pso_multiple(filename="pso_aci_multirun"):
    from pso_main import N_PARTICLES, MAX_ITER

    print(f"PSO MULTI-RUN  [{filename}]")
    print(f"n_particles={N_PARTICLES}  max_iter={MAX_ITER}  N_RUNS={N_RUNS}")
    print(f"{'='*60}")

    run_fitnesses, run_cappvs, run_ebesses, run_tcs, run_histories = [], [], [], [], []

    for run_idx in range(1, N_RUNS + 1):
        print(f"  run {run_idx}/{N_RUNS} ...", end=" ", flush=True)
        bf, bc, be, bt, hist = _run_one_trial_pso()
        run_fitnesses.append(bf);  run_cappvs.append(bc)
        run_ebesses.append(be);    run_tcs.append(bt)
        run_histories.append(hist)
        print(f"fitness={bf:.4f}")

    mean_history, std_history, stats = _multirun_stats(
        run_fitnesses, run_cappvs, run_ebesses, run_tcs, run_histories)

    csv_path = os.path.join(DIR_PSO, f"pso_multirun_{filename}.csv")
    _save_multirun_csv(csv_path, run_fitnesses, run_cappvs,
                       run_ebesses, run_tcs, stats)

    graph_path = os.path.join(DIR_PSO, f"pso_multirun_{filename}.jpg")
    _plot_multirun_convergence(
        mean_history, std_history, stats["all_histories"],
        label="PSO", color="#2A9D8F", filename=filename, save_path=graph_path)

    print(f"\nCSV   → {csv_path}")
    print(f"Graph → {graph_path}")
    print(f"Mean Fitness = {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")
    print(f"Min = {stats['min_fitness']:.4f}  Max = {stats['max_fitness']:.4f}")
    print(f"Mean CAPPV   = {stats['mean_cappv']:.4f}")
    print(f"Mean EBESS   = {stats['mean_ebess']:.4f}")
    print(f"Mean TC      = {stats['mean_tc']:.4f}")

    return mean_history, std_history, stats


def _plot_multirun_convergence(mean_history, std_history, all_histories,
                                label, color, filename, save_path):
    mh = np.array(mean_history)
    sh = np.array(std_history)
    xs = list(range(1, len(mh) + 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f"{label} Convergence ({N_RUNS} runs) — {filename}",
        fontsize=12, fontweight='bold'
    )

    for h in all_histories:
        ax.plot(range(1, len(h) + 1), h,
                color=color, alpha=0.20, linewidth=0.8)

    ax.plot(xs, mh, color=color, linewidth=2.2, label="Mean")
    ax.fill_between(xs, mh - sh, mh + sh,
                    alpha=0.18, color=color, label="±1σ")
    ax.set_xlabel("Iteration / Generation")
    ax.set_ylabel("Best Fitness")
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.annotate(f"Final mean: {mh[-1]:.2f}",
                xy=(xs[-1], mh[-1]),
                xytext=(-80, 10), textcoords='offset points',
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ga_pso_comparison(ga_mean_hist, ga_std_hist, ga_stats,
                            pso_mean_hist, pso_std_hist, pso_stats,
                            save_path="comparison_ga_vs_pso_multirun.jpg"):
    ga_mh  = np.array(ga_mean_hist)
    ga_sh  = np.array(ga_std_hist)
    pso_mh = np.array(pso_mean_hist)
    pso_sh = np.array(pso_std_hist)
    ga_xs  = list(range(1, len(ga_mh)  + 1))
    pso_xs = list(range(1, len(pso_mh) + 1))

    # per-run final fitness values (last element of each padded history)
    ga_finals  = [h[-1] for h in ga_stats["all_histories"]]
    pso_finals = [h[-1] for h in pso_stats["all_histories"]]

    fig, (ax_conv, ax_box) = plt.subplots(
        1, 2, figsize=(18, 6),
        gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.suptitle(
        f"GA vs PSO Comparison  ({N_RUNS} runs each — mean ± 1σ)",
        fontsize=13, fontweight='bold'
    )

    # ── LEFT: convergence curves (zoomed y-axis so the gap is visible) ──
    ax_conv.plot(ga_xs, ga_mh, color="steelblue", linewidth=2.2,
                 label=f"GA  (mean={ga_mh[-1]:.4g})")
    ax_conv.fill_between(ga_xs, ga_mh - ga_sh, ga_mh + ga_sh,
                         alpha=0.18, color="steelblue")

    ax_conv.plot(pso_xs, pso_mh, color="tomato", linewidth=2.2,
                 linestyle='--', label=f"PSO (mean={pso_mh[-1]:.4g})")
    ax_conv.fill_between(pso_xs, pso_mh - pso_sh, pso_mh + pso_sh,
                         alpha=0.18, color="tomato")

    # zoom y-axis to the converged region (last 80% of iterations)
    cutoff = max(1, min(len(ga_mh), len(pso_mh)) // 5)
    all_late = list(ga_mh[cutoff:]) + list(pso_mh[cutoff:])
    y_lo = min(all_late) * 0.9995
    y_hi = max(all_late) * 1.0005
    ax_conv.set_ylim(y_lo, y_hi)

    ax_conv.set_xlabel("Generation / Iteration", fontsize=10)
    ax_conv.set_ylabel("Best Fitness (zoomed)", fontsize=10)
    ax_conv.legend(fontsize=9)
    ax_conv.grid(linestyle='--', alpha=0.4)
    ax_conv.set_title("Mean Convergence ± 1σ  (y-axis zoomed to converged region)")

    # ── RIGHT: box plot + individual run dots ──
    bp = ax_box.boxplot(
        [ga_finals, pso_finals],
        labels=["GA", "PSO"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("tomato")
    bp["boxes"][1].set_alpha(0.6)

    # jitter individual points
    rng = np.random.default_rng(0)
    for i, (finals, color) in enumerate(
            [(ga_finals, "steelblue"), (pso_finals, "tomato")], 1):
        jitter = rng.uniform(-0.12, 0.12, size=len(finals))
        ax_box.scatter(i + jitter, finals, color=color,
                       s=28, zorder=5, alpha=0.8, edgecolors="white", linewidths=0.5)

    # annotate means
    for i, (m, color) in enumerate(
            [(ga_stats['mean_fitness'], "steelblue"),
             (pso_stats['mean_fitness'], "tomato")], 1):
        ax_box.axhline(m, xmin=(i - 1) / 2, xmax=i / 2,
                       color=color, linestyle=':', linewidth=1.4)

    pct_diff = (ga_stats['mean_fitness'] - pso_stats['mean_fitness']) \
               / ga_stats['mean_fitness'] * 100
    ax_box.set_ylabel("Final Best Fitness", fontsize=10)
    ax_box.set_title(
        f"Per-run Final Fitness Distribution\n"
        f"PSO {'better' if pct_diff >= 0 else 'worse'} by {abs(pct_diff):.2f}%"
    )
    ax_box.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    pct_improvement = (ga_stats['mean_fitness'] - pso_stats['mean_fitness']) \
                      / ga_stats['mean_fitness'] * 100
    print("=" * 62)
    print(f"{'Metric':<24} {'GA':>17} {'PSO':>17}")
    print("=" * 62)
    print(f"{'Mean Fitness':<24} {ga_stats['mean_fitness']:>17.2f} {pso_stats['mean_fitness']:>17.2f}")
    print(f"{'Std Fitness':<24} {ga_stats['std_fitness']:>17.2f} {pso_stats['std_fitness']:>17.2f}")
    print(f"{'Min Fitness':<24} {ga_stats['min_fitness']:>17.2f} {pso_stats['min_fitness']:>17.2f}")
    print(f"{'Max Fitness':<24} {ga_stats['max_fitness']:>17.2f} {pso_stats['max_fitness']:>17.2f}")
    print(f"{'Mean CAPPV':<24} {ga_stats['mean_cappv']:>17.2f} {pso_stats['mean_cappv']:>17.2f}")
    print(f"{'Mean EBESS':<24} {ga_stats['mean_ebess']:>17.2f} {pso_stats['mean_ebess']:>17.2f}")
    print(f"{'Mean Total Cost':<24} {ga_stats['mean_tc']:>17.2f} {pso_stats['mean_tc']:>17.2f}")
    print("=" * 62)
    sign = "↓ better" if pct_improvement >= 0 else "↑ worse"
    print(f"PSO vs GA mean fitness: {pct_improvement:+.3f}%  ({sign})")
    print(f"Comparison graph → {save_path}")


def regenerate_comparison_chart(
        ga_multirun_csv  = "output_convergence/ga_multirun_ga_aci_final.csv",
        pso_multirun_csv = "output_pso/pso_multirun_pso_aci_final.csv",
        ga_conv_csv      = "output_convergence/convergence_aci_1year.csv",
        pso_conv_csv     = "pso_convergence_results_pso_aci_final.csv",
        save_path        = "comparison_ga_vs_pso_multirun.jpg"):
    """
    Rebuild the GA-vs-PSO comparison chart from existing CSV files only.

    Left panel : convergence curves from the single-run CSVs.
    Right panel: per-run final-fitness box plot from the multi-run CSVs.
    """

    def _read_multirun_csv(path):
        """Return list of per-run final fitness values + aggregate stats dict."""
        finals = []
        agg = {}   # keyed by tag: "MEAN"/"STD"/"MIN"/"MAX"
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tag = row["Run"]
                if tag.lstrip("-").isdigit():
                    finals.append(float(row["Best_Fitness"]))
                elif tag in ("MEAN", "STD", "MIN", "MAX"):
                    agg[tag] = {k: float(v) for k, v in row.items()
                                if k != "Run" and v != ""}
        stats = {
            "mean_fitness": agg.get("MEAN", {}).get("Best_Fitness", float("nan")),
            "std_fitness":  agg.get("STD",  {}).get("Best_Fitness", float("nan")),
            "min_fitness":  agg.get("MIN",  {}).get("Best_Fitness", float("nan")),
            "max_fitness":  agg.get("MAX",  {}).get("Best_Fitness", float("nan")),
            "mean_cappv":   agg.get("MEAN", {}).get("Best_CAPPV",   float("nan")),
            "mean_ebess":   agg.get("MEAN", {}).get("Best_EBESS",   float("nan")),
            "mean_tc":      agg.get("MEAN", {}).get("Best_TC",      float("nan")),
        }
        return finals, stats

    def _read_convergence_csv(path, fitness_col="Fitness"):
        """Return list of fitness values across iterations from a convergence CSV."""
        values = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                values.append(float(row[fitness_col]))
        return values

    ga_finals,  ga_stats  = _read_multirun_csv(ga_multirun_csv)
    pso_finals, pso_stats = _read_multirun_csv(pso_multirun_csv)
    ga_conv  = _read_convergence_csv(ga_conv_csv)
    pso_conv = _read_convergence_csv(pso_conv_csv)

    ga_xs  = list(range(1, len(ga_conv)  + 1))
    pso_xs = list(range(1, len(pso_conv) + 1))

    fig, (ax_conv, ax_box) = plt.subplots(
        1, 2, figsize=(18, 6),
        gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.suptitle(
        f"GA vs PSO Comparison  "
        f"(convergence: 1 run each | box plot: {len(ga_finals)} runs each)",
        fontsize=13, fontweight='bold'
    )

    # ── LEFT: convergence from single-run CSVs (zoomed) ──
    ax_conv.plot(ga_xs,  ga_conv,  color="steelblue", linewidth=2,
                 label=f"GA  (final={ga_conv[-1]:.4g})")
    ax_conv.plot(pso_xs, pso_conv, color="tomato",    linewidth=2,
                 linestyle='--', label=f"PSO (final={pso_conv[-1]:.4g})")

    cutoff = max(1, min(len(ga_conv), len(pso_conv)) // 5)
    all_late = ga_conv[cutoff:] + pso_conv[cutoff:]
    y_lo = min(all_late) * 0.9995
    y_hi = max(all_late) * 1.0005
    ax_conv.set_ylim(y_lo, y_hi)
    ax_conv.set_xlabel("Generation / Iteration", fontsize=10)
    ax_conv.set_ylabel("Best Fitness (zoomed)", fontsize=10)
    ax_conv.legend(fontsize=9)
    ax_conv.grid(linestyle='--', alpha=0.4)
    ax_conv.set_title("Single-run Convergence  (y-axis zoomed to converged region)")

    # ── RIGHT: box plot from multi-run CSVs ──
    bp = ax_box.boxplot(
        [ga_finals, pso_finals],
        labels=["GA", "PSO"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )
    bp["boxes"][0].set_facecolor("steelblue"); bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("tomato");    bp["boxes"][1].set_alpha(0.6)

    rng = np.random.default_rng(0)
    for i, (finals, color) in enumerate(
            [(ga_finals, "steelblue"), (pso_finals, "tomato")], 1):
        jitter = rng.uniform(-0.12, 0.12, size=len(finals))
        ax_box.scatter(i + jitter, finals, color=color,
                       s=28, zorder=5, alpha=0.8,
                       edgecolors="white", linewidths=0.5)

    pct = (ga_stats["mean_fitness"] - pso_stats["mean_fitness"]) \
          / ga_stats["mean_fitness"] * 100
    sign = "better" if pct >= 0 else "worse"
    ax_box.set_ylabel("Final Best Fitness", fontsize=10)
    ax_box.set_title(
        f"Per-run Final Fitness  ({len(ga_finals)} runs each)\n"
        f"PSO {sign} than GA by {abs(pct):.2f}%"
    )
    ax_box.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Chart saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════
#  PSO — SINGLE-RUN CONVERGENCE  (mirrors run_main_convergence)
# ═══════════════════════════════════════════════════════════════════

PSO_BASELINE = {"W": 0.7, "C1": 1.5, "C2": 1.5}

PSO_SENSITIVITY_GRIDS = {
    "W":  [0.4, 0.7, 0.9],
    "C1": [1.0, 1.5, 2.0],
    "C2": [1.0, 1.5, 2.0],
}


def _run_one_trial_pso_params(W=0.7, C1=1.5, C2=1.5):
    from pso_main import pso, N_PARTICLES

    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    try:
        fitness_hist, cappv_hist, ebess_hist, tc_hist, _, _ = pso(
            N_PARTICLES, w=W, c1=C1, c2=C2)
    finally:
        sys.stdout = sys.__stdout__
        devnull.close()

    return fitness_hist[-1], cappv_hist[-1], ebess_hist[-1], tc_hist[-1], fitness_hist


def run_pso_convergence(filename="pso_aci_1year"):
    from pso_main import pso, N_PARTICLES, MAX_ITER

    print(f"PSO CONVERGENCE RUN  [{filename}]")
    print(f"W={PSO_BASELINE['W']}  C1={PSO_BASELINE['C1']}  C2={PSO_BASELINE['C2']}  "
          f"n_particles={N_PARTICLES}  max_iter={MAX_ITER}")

    log_path = os.path.join(DIR_CONVERGENCE, f"{filename}_log.txt")
    orig_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as f:
        sys.stdout = f
        try:
            fitness_all, cappv_all, ebess_all, total_cost_all, cpv_all, cbess_all = \
                pso(N_PARTICLES, w=PSO_BASELINE["W"],
                    c1=PSO_BASELINE["C1"], c2=PSO_BASELINE["C2"])
        finally:
            sys.stdout = orig_stdout

    def _rv(v):
        return float(v.real if hasattr(v, 'real') else v)

    csv_path = os.path.join(DIR_CONVERGENCE, f"convergence_{filename}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Iteration", "Fitness", "CAPPV", "CPV",
                    "EBESS", "CBESS", "Total_Cost"])
        for i in range(len(fitness_all)):
            w.writerow([i + 1,
                        round(_rv(fitness_all[i]),    4),
                        round(_rv(cappv_all[i]),      4),
                        round(_rv(cpv_all[i]),        4),
                        round(_rv(ebess_all[i]),      4),
                        round(_rv(cbess_all[i]),      4),
                        round(_rv(total_cost_all[i]), 4)])

    graph_path = os.path.join(DIR_CONVERGENCE, f"convergence_{filename}.jpg")
    _plot_convergence(fitness_all, cappv_all, ebess_all, total_cost_all,
                      title=f"PSO Convergence — {filename} "
                            f"(W={PSO_BASELINE['W']}, "
                            f"C1={PSO_BASELINE['C1']}, "
                            f"C2={PSO_BASELINE['C2']})",
                      save_path=graph_path)

    print(f"Log              → {log_path}")
    print(f"Convergence CSV  → {csv_path}")
    print(f"Convergence Graph→ {graph_path}")
    print(f"Best Fitness     = {round(_rv(fitness_all[-1]),    4)}")
    print(f"Best CAPPV       = {round(_rv(cappv_all[-1]),      4)}")
    print(f"Best EBESS       = {round(_rv(ebess_all[-1]),      4)}")
    print(f"Best Total Cost  = {round(_rv(total_cost_all[-1]), 4)}")

    return fitness_all, cappv_all, ebess_all, total_cost_all, cpv_all, cbess_all


# ═══════════════════════════════════════════════════════════════════
#  PSO — SENSITIVITY ANALYSIS  (sweeps W, C1, C2)
# ═══════════════════════════════════════════════════════════════════

def run_pso_sensitivity_analysis():
    from pso_main import N_PARTICLES, MAX_ITER

    print(f"PSO SENSITIVITY ANALYSIS")
    print(f"Fixed: n_particles={N_PARTICLES}, max_iter={MAX_ITER}")
    print(f"Swept: W, C1, C2")
    print(f"Runs per config: {N_RUNS}  (No Free Lunch)")
    print(f"{'='*60}")

    all_results  = {}
    summary_rows = []

    for param_name, grid in PSO_SENSITIVITY_GRIDS.items():
        print(f"\n  ▶ Sweeping: {param_name}  grid={grid}")
        param_results = []

        for val in grid:
            cfg = dict(PSO_BASELINE)
            cfg[param_name] = val

            run_fitnesses  = []
            run_cappvs     = []
            run_ebesses    = []
            run_tcs        = []
            run_histories  = []

            for run_idx in range(1, N_RUNS + 1):
                print(f"     {param_name}={val}  run {run_idx}/{N_RUNS} ...",
                      end=" ", flush=True)
                bf, bc, be, bt, hist = _run_one_trial_pso_params(**cfg)
                run_fitnesses.append(bf)
                run_cappvs.append(bc)
                run_ebesses.append(be)
                run_tcs.append(bt)
                run_histories.append(hist)
                print(f"fitness={bf:.4f}")

            mean_f = float(np.mean(run_fitnesses))
            std_f  = float(np.std(run_fitnesses, ddof=1))
            min_f  = float(np.min(run_fitnesses))
            max_f  = float(np.max(run_fitnesses))

            mean_cappv = float(np.mean(run_cappvs))
            mean_ebess = float(np.mean(run_ebesses))
            mean_tc    = float(np.mean(run_tcs))

            target_len   = int(np.median([len(h) for h in run_histories]))
            norm_hists   = [_pad_history(h, target_len) for h in run_histories]
            mean_history = list(np.mean(norm_hists, axis=0))
            std_history  = list(np.std(norm_hists,  axis=0, ddof=1))

            entry = {
                "param_name"     : param_name,
                "param_value"    : val,
                "mean_fitness"   : mean_f,
                "std_fitness"    : std_f,
                "min_fitness"    : min_f,
                "max_fitness"    : max_f,
                "mean_cappv"     : mean_cappv,
                "mean_ebess"     : mean_ebess,
                "mean_total_cost": mean_tc,
                "mean_history"   : mean_history,
                "std_history"    : std_history,
                "all_histories"  : norm_hists,
            }
            param_results.append(entry)
            summary_rows.append({k: v for k, v in entry.items()
                                  if k not in ("mean_history",
                                               "std_history",
                                               "all_histories")})

        all_results[param_name] = param_results

        csv_path = os.path.join(DIR_SENSITIVITY, f"sensitivity_pso_{param_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["param_name", "param_value",
                          "mean_fitness", "std_fitness",
                          "min_fitness",  "max_fitness",
                          "mean_cappv", "mean_ebess", "mean_total_cost"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in param_results:
                w.writerow({k: round(v, 6) if isinstance(v, float) else v
                             for k, v in r.items() if k in fieldnames})
        print(f"\nCSV saved → {csv_path}")

        graph_path = os.path.join(DIR_SENSITIVITY, f"sensitivity_pso_{param_name}.jpg")
        _plot_pso_sensitivity_param(param_name, grid, param_results, graph_path)
        print(f"Graph saved → {graph_path}")

    master_csv = os.path.join(DIR_SENSITIVITY, "sensitivity_pso_summary.csv")
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["param_name", "param_value",
                      "mean_fitness", "std_fitness",
                      "min_fitness",  "max_fitness",
                      "mean_cappv", "mean_ebess", "mean_total_cost"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v
                         for k, v in r.items() if k in fieldnames})
    print(f"\nMaster summary CSV → {master_csv}")

    overview_path = os.path.join(DIR_SENSITIVITY, "sensitivity_pso_overview.jpg")
    _plot_pso_sensitivity_overview(all_results, overview_path)
    print(f"Overview graph    → {overview_path}")


def _plot_pso_sensitivity_param(param_name, grid, results, save_path):
    means  = [r["mean_fitness"] for r in results]
    stds   = [r["std_fitness"]  for r in results]
    mins   = [r["min_fitness"]  for r in results]
    maxs   = [r["max_fitness"]  for r in results]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(grid)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"PSO Sensitivity Analysis — {param_name}\n"
        f"(n_particles=fixed, {N_RUNS} runs per config — No Free Lunch)",
        fontsize=12, fontweight='bold'
    )

    x_pos = np.arange(len(grid))
    bars  = ax1.bar(x_pos, means, yerr=stds, color=colors,
                    edgecolor='black', linewidth=0.6,
                    error_kw=dict(ecolor='black', capsize=5, linewidth=1.2))
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(v) for v in grid])
    ax1.set_title(f"Mean ± Std Best Fitness\n({N_RUNS} runs each)")
    ax1.set_xlabel(param_name)
    ax1.set_ylabel("Best Fitness")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + (max(means) * 0.005),
                 f"{m:.2f}", ha='center', va='bottom', fontsize=7.5)

    best_idx = int(np.argmin(means))
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2.2)
    ax1.annotate("Best\nmean",
                 xy=(x_pos[best_idx], means[best_idx]),
                 xytext=(0, 28), textcoords='offset points',
                 ha='center', fontsize=8, color='red',
                 arrowprops=dict(arrowstyle='->', color='red'))

    if PSO_BASELINE[param_name] in grid:
        bl_idx = grid.index(PSO_BASELINE[param_name])
        ax1.axvline(bl_idx, color='gray', linestyle=':', linewidth=1.5)
        ax1.text(bl_idx + 0.05, ax1.get_ylim()[0], "baseline",
                 fontsize=7.5, color='gray', va='bottom')

    for r, val, col in zip(results, grid, colors):
        mh     = r["mean_history"]
        sh     = r["std_history"]
        mh_arr = np.array(mh)
        sh_arr = np.array(sh)
        xs     = list(range(1, len(mh) + 1))
        ax2.plot(xs, mh_arr, label=f"{param_name}={val}",
                 color=col, linewidth=1.6)
        ax2.fill_between(xs, mh_arr - sh_arr, mh_arr + sh_arr,
                         alpha=0.15, color=col)

    ax2.set_title(f"Mean Convergence ± Std\n({N_RUNS} runs, shaded = ±1σ)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Fitness")
    ax2.legend(fontsize=7.5, loc='upper right')
    ax2.grid(linestyle='--', alpha=0.4)

    for val, col, mn, mx, me in zip(grid, colors, mins, maxs, means):
        ax3.plot([val, val], [mn, mx], color=col, linewidth=3, solid_capstyle='round')
        ax3.scatter([val], [me], color=col, s=60, zorder=5)
        ax3.scatter([val], [mn], marker='^', color=col, s=40, zorder=5)
        ax3.scatter([val], [mx], marker='v', color=col, s=40, zorder=5)

    ax3.set_title(f"Min / Mean / Max Fitness\n(robustness across {N_RUNS} runs)")
    ax3.set_xlabel(param_name)
    ax3.set_ylabel("Best Fitness")
    ax3.grid(linestyle='--', alpha=0.4)

    if PSO_BASELINE[param_name] in grid:
        ax3.axvline(PSO_BASELINE[param_name], color='gray',
                    linestyle=':', linewidth=1.5, label='baseline')
        ax3.legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_pso_sensitivity_overview(all_results, save_path):
    params  = list(PSO_SENSITIVITY_GRIDS.keys())
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, len(params)))

    fig, axs = plt.subplots(1, len(params), figsize=(7 * len(params), 5))
    if len(params) == 1:
        axs = [axs]
    fig.suptitle(
        "PSO Sensitivity Overview — W, C1, C2\n"
        f"({N_RUNS} runs/config — No Free Lunch)",
        fontsize=12, fontweight='bold'
    )

    for ax, param, color in zip(axs, params, palette):
        results = all_results[param]
        vals    = [r["param_value"]  for r in results]
        means   = [r["mean_fitness"] for r in results]
        stds    = [r["std_fitness"]  for r in results]
        means_a = np.array(means)
        stds_a  = np.array(stds)

        ax.plot(vals, means_a, marker='o', color=color, linewidth=2,
                markersize=7, markerfacecolor='white', markeredgewidth=2,
                label='mean fitness')
        ax.fill_between(vals, means_a - stds_a, means_a + stds_a,
                        alpha=0.18, color=color, label='±1σ')

        ax.set_title(f"Effect of {param}", fontsize=11)
        ax.set_xlabel(param, fontsize=9)
        ax.set_ylabel("Best Fitness (mean ± std)", fontsize=9)
        ax.grid(linestyle='--', alpha=0.4)
        ax.legend(fontsize=8)

        baseline_val = PSO_BASELINE[param]
        if baseline_val in vals:
            ax.axvline(baseline_val, color='gray', linestyle=':', linewidth=1.4)
            bl_y = means[vals.index(baseline_val)]
            ax.annotate("baseline",
                        xy=(baseline_val, bl_y),
                        xytext=(8, -14), textcoords='offset points',
                        fontsize=7.5, color='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)