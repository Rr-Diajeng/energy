import os
import sys
import csv
import functools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ga_main import genetic_algorithm, POPULATION_SIZE

FIXED_POPULATION_SIZE = 100
FIXED_GENERATIONS     = 100
N_RUNS                = 3

BASELINE = {
    "mutation_rate"  : 0.10,
    "crossover_rate" : 0.80,
}

SENSITIVITY_GRIDS = {
    "mutation_rate"  : [0.10, 0.20, 0.30],
    "crossover_rate" : [0.70, 0.80, 0.90],
}

DIR_CONVERGENCE = "output_convergence"
DIR_SENSITIVITY = "output_sensitivity"

for _d in [DIR_CONVERGENCE, DIR_SENSITIVITY]:
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