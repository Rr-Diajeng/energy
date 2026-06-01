from get_fitness import fitness
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

N_PARTICLES = 70
MAX_ITER = 100
W = 0.7 
C1 = 1.5 
C2 = 1.5 
V_MAX = 1000  
CAPPV_MIN, CAPPV_MAX = 100, 20000
EBESS_MIN, EBESS_MAX = 100, 20000


def _to_real(val):
    return val.real if hasattr(val, 'real') else val


def init_swarm(n_particles):
    particles = []
    for _ in range(n_particles):
        cappv = float(random.randint(CAPPV_MIN, CAPPV_MAX))
        ebess = float(random.randint(EBESS_MIN, EBESS_MAX))
        particles.append({
            'cappv': cappv,
            'ebess': ebess,
            'v_cappv': random.uniform(-V_MAX, V_MAX),
            'v_ebess': random.uniform(-V_MAX, V_MAX),
            'pbest_cappv': cappv,
            'pbest_ebess': ebess,
            'pbest_fitness': None,
        })
    return particles


def _particles_to_population(particles):
    population = []
    for p in particles:
        cappv = max(CAPPV_MIN, min(CAPPV_MAX, int(round(p['cappv']))))
        ebess = max(EBESS_MIN, min(EBESS_MAX, int(round(p['ebess']))))
        cappv_bits = [int(b) for b in bin(cappv)[2:].zfill(15)]
        ebess_bits = [int(b) for b in bin(ebess)[2:].zfill(15)]
        population.append({
            'cappv_bits': cappv_bits,
            'cappv_desimal': cappv,
            'ebess_bits': ebess_bits,
            'ebess_desimal': ebess,
        })
    return population


def pso(n_particles=N_PARTICLES, w=W, c1=C1, c2=C2):
    particles = init_swarm(n_particles)

    evaluated = fitness(_particles_to_population(particles))

    gbest = None
    for particle, result in zip(particles, evaluated):
        f = _to_real(result['fitness'])
        if result['erase']:
            particle['cappv'] = float(random.randint(CAPPV_MIN, CAPPV_MAX))
            particle['ebess'] = float(random.randint(EBESS_MIN, EBESS_MAX))
            particle['v_cappv'] = random.uniform(-V_MAX, V_MAX)
            particle['v_ebess'] = random.uniform(-V_MAX, V_MAX)
            particle['pbest_cappv'] = particle['cappv']
            particle['pbest_ebess'] = particle['ebess']
            particle['pbest_fitness'] = float('inf')
            continue
        particle['pbest_fitness'] = f
        if gbest is None or f < gbest['fitness']:
            gbest = {
                'fitness': f,
                'cappv': particle['cappv'],
                'ebess': particle['ebess'],
                'cpv': _to_real(result['cpv']),
                'cbess': _to_real(result['cbess']),
                'TC': _to_real(result['TC']),
            }

    fitness_all = []
    cappv_all = []
    ebess_all = []
    total_cost_all = []
    cpv_all = []
    cbess_all = []

    for iteration in range(MAX_ITER):
        print(f"\nPSO Iteration {iteration + 1}:")

        # update velocities and positions
        for particle in particles:
            r1 = random.random()
            r2 = random.random()

            particle['v_cappv'] = (
                w * particle['v_cappv']
                + c1 * r1 * (particle['pbest_cappv'] - particle['cappv'])
                + c2 * r2 * (gbest['cappv'] - particle['cappv'])
            )
            particle['v_ebess'] = (
                w * particle['v_ebess']
                + c1 * r1 * (particle['pbest_ebess'] - particle['ebess'])
                + c2 * r2 * (gbest['ebess'] - particle['ebess'])
            )

            # clamp velocity
            particle['v_cappv'] = max(-V_MAX, min(V_MAX, particle['v_cappv']))
            particle['v_ebess'] = max(-V_MAX, min(V_MAX, particle['v_ebess']))

            # update position and clamp to bounds
            particle['cappv'] = max(CAPPV_MIN, min(CAPPV_MAX, particle['cappv'] + particle['v_cappv']))
            particle['ebess'] = max(EBESS_MIN, min(EBESS_MAX, particle['ebess'] + particle['v_ebess']))

        # evaluate new positions
        evaluated = fitness(_particles_to_population(particles))

        for particle, result in zip(particles, evaluated):
            f = _to_real(result['fitness'])

            if result['erase']:
                particle['cappv'] = float(random.randint(CAPPV_MIN, CAPPV_MAX))
                particle['ebess'] = float(random.randint(EBESS_MIN, EBESS_MAX))
                particle['v_cappv'] = random.uniform(-V_MAX, V_MAX)
                particle['v_ebess'] = random.uniform(-V_MAX, V_MAX)
                particle['pbest_cappv'] = particle['cappv']
                particle['pbest_ebess'] = particle['ebess']
                particle['pbest_fitness'] = float('inf')
                continue

            if f < particle['pbest_fitness']:
                particle['pbest_fitness'] = f
                particle['pbest_cappv'] = particle['cappv']
                particle['pbest_ebess'] = particle['ebess']

            if f < gbest['fitness']:
                gbest = {
                    'fitness': f,
                    'cappv': particle['cappv'],
                    'ebess': particle['ebess'],
                    'cpv': _to_real(result['cpv']),
                    'cbess': _to_real(result['cbess']),
                    'TC': _to_real(result['TC']),
                }

        # convergence tracking
        if len(fitness_all) == 0:
            fitness_all.append(gbest['fitness'])
            cappv_all.append(gbest['cappv'])
            ebess_all.append(gbest['ebess'])
            total_cost_all.append(gbest['TC'])
            cpv_all.append(gbest['cpv'])
            cbess_all.append(gbest['cbess'])
        else:
            prev = fitness_all[-1]
            curr = gbest['fitness']
            if curr <= prev:
                fitness_all.append(curr)
                cappv_all.append(gbest['cappv'])
                ebess_all.append(gbest['ebess'])
                total_cost_all.append(gbest['TC'])
                cpv_all.append(gbest['cpv'])
                cbess_all.append(gbest['cbess'])
                print(f"Iteration {iteration + 1}: NEW BEST stored. Fitness = {curr:.4f}")
            else:
                fitness_all.append(fitness_all[-1])
                cappv_all.append(cappv_all[-1])
                ebess_all.append(ebess_all[-1])
                total_cost_all.append(total_cost_all[-1])
                cpv_all.append(cpv_all[-1])
                cbess_all.append(cbess_all[-1])
                print(f"Iteration {iteration + 1}: No improvement. Best = {prev:.4f}")

    return fitness_all, cappv_all, ebess_all, total_cost_all, cpv_all, cbess_all


def do_convergence_pso(
    fitness_all,
    cappv_all,
    ebess_all,
    total_cost_all,
    cpv_all,
    cbess_all,
    filename,
):
    csv_filename = f'pso_convergence_results_{filename}.csv'
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Fitness', 'CAPPV', 'CPV', 'EBESS', 'CBESS', 'Total_Cost'])
        for i in range(len(fitness_all)):
            writer.writerow([
                i + 1,
                round(fitness_all[i], 4),
                round(cappv_all[i], 4),
                round(cpv_all[i], 4),
                round(ebess_all[i], 4),
                round(cbess_all[i], 4),
                round(total_cost_all[i], 4),
            ])

    return fitness_all, cappv_all, ebess_all, total_cost_all


def make_graph_pso(fitness_all, cappv_all, ebess_all, total_cost_all, count):
    fig, axs = plt.subplots(4, 1, figsize=(15, 12))

    axs[0].plot(fitness_all, marker='o', linestyle='-', markersize=4)
    axs[0].set_title('PSO - Fitness Over Iterations')
    axs[0].set_ylabel('Fitness')
    axs[0].set_xlabel('Iteration')

    axs[1].plot(cappv_all, marker='o', linestyle='-', markersize=4)
    axs[1].set_title('PSO - CAPPV Over Iterations')
    axs[1].set_ylabel('CAPPV')
    axs[1].set_xlabel('Iteration')

    axs[2].plot(ebess_all, marker='o', linestyle='-', markersize=4)
    axs[2].set_title('PSO - EBESS Over Iterations')
    axs[2].set_ylabel('EBESS')
    axs[2].set_xlabel('Iteration')

    axs[3].plot(total_cost_all, marker='o', linestyle='-', markersize=4)
    axs[3].set_title('PSO - Total Cost Over Iterations')
    axs[3].set_ylabel('Total Cost')
    axs[3].set_xlabel('Iteration')

    plt.tight_layout()
    plt.savefig(f'pso_graph_output_{count}.jpg', format='jpg')
    plt.close(fig)
