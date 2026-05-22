import random
import csv

#parameters for PVs and BESSs
SYSTEM_LOSS = 0.14

#BESS parameters
BETA_0 = 3832 # fitting curve coefficient, depends on batteries type
BETA_1 = 0.68 # fitting curve coefficient, depends on batteries type
BETA_2 = 1.64 # fitting curve coefficient, depends on batteries type
BESS_CH_EFF = 0.95
BESS_DCH_EFF = 0.9
COST_CAP_BESS = 50000 # cost BESSs per kW
ALPHA_I = 0.03
EFF = 0.95

#BESS Cost parameters
POWER_RATING = 320 # $ per kW
ENERGY_RATING = 360 # $ per kW
BESS_INSTALLATION_COST = 15 # $ per kW
BESS_OPERATION_MAINTENANCE = 5 # $ per kW
CCAP = 50000 # battery capital cost

# Solar PV Cost parameters
PV_INSTALLATION_COST = 1200 # $ per kW
PV_OPERATION_MAINTENANCE = 0.04 # $ per kW

def desimal_to_binary(des):
    binary = bin(des)[2:]
    return binary.zfill(15)

def binary_to_desimal(bits):
    # Gabungkan setiap bit menjadi satu string, bukan keseluruhan list
    binary_str = ''.join(str(bit) for bit in bits)
    return int(binary_str, 2)

def init_population(population_size):
    population = []
    for _ in range(population_size):
        cappv = random.randint(100, 20000)
        ebess = random.randint(100, 20000)

        cappv_bin = desimal_to_binary(cappv)
        ebess_bin = desimal_to_binary(ebess)

        cappv_bits = [int(bit) for bit in cappv_bin]
        ebess_bits = [int(bit) for bit in ebess_bin]

        population.append(dict(cappv_bits=cappv_bits, cappv_desimal=cappv, ebess_bits=ebess_bits, ebess_desimal=ebess))
    return population


FILE_PATH = "./data_with_datetime.csv"

dataset = []

with open(FILE_PATH, 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter=';')
    next(reader)  # Skip header row

    for row in reader:
        load = row[0].replace(',', '.')
        sg = row[1].replace(',', '.')
        dataset.append((float(load), float(sg)))

def fitness(population):
    population_iterated = []
    for chromosome in population:
        cappv_desimal = chromosome['cappv_desimal']
        ebess_desimal = chromosome['ebess_desimal']

        cappv_binary = chromosome['cappv_bits']
        ebess_binary = chromosome['ebess_bits']

        # print("Cappv: ", cappv_desimal)
        # print("Ebess: ", ebess_desimal)

        sg_total = round(sum(sg for _, sg in dataset), 2)

        #math model start here
        pbess_list = []
        ppv_out_list = [] #kW

        for load, sg in dataset:
            ppv_out = sg * cappv_desimal
            ppv_out_list.append(round(ppv_out, 2))

            pbess = ppv_out - load
            pbess_list.append(round(pbess, 2))
        
        # print("PBESS: ", pbess_list)
        # print("Ppv_out: ", ppv_out_list)

        pdischarge_list = []
        pcharge_list = []

        for pbess in pbess_list:
            pdischarge = 0
            if pbess < 0:
                pdischarge = -pbess / 0.9
            pdischarge_list.append(round(pdischarge, 2))

            pcharge = 0
            if pbess > 0:
                pcharge = pbess * 0.95
            pcharge_list.append(round(pcharge, 2))
        
        # print("Pdischarge: ", pdischarge_list)
        # print("Pcharge: ", pcharge_list)

        state_of_charge_list = []
        depth_of_discharge_list = []
        ncycle_list = [] #constant for one-cycle degradation cost
        ce_soc_list = [] #one cycle degradation cost function
        cbt_list = [] #cost per 1 cycle
        excess_soc_list = [] # when state_of_charge_list for bess is fully charged (1), append the value in here. otherwise, 0
        surge_soc_list = [] # when state_of_charge_list cannot provide for demand, append here (penalty)

        # main di pdischarge karena panjang keduanya sama
        for i, (pdischarge, pcharge) in enumerate(zip(pdischarge_list, pcharge_list)):
            state_of_charge = 0.8  # Mulai dengan SoC 80%

            if i > 0:
                state_of_charge = state_of_charge_list[i - 1]

            result_soc = (state_of_charge * (1 - ALPHA_I)) + ((pcharge * EFF) / ebess_desimal) - (pdischarge / (ebess_desimal * EFF))

            if result_soc < 0.2:
                surge_soc_list.append(round(result_soc, 2))
            else:
                surge_soc_list.append(0)

            if result_soc > 0.8:
                excess_soc_list.append(round(result_soc, 2))
            else:
                excess_soc_list.append(0)

            if result_soc > 1:
                result_soc = 1
            elif result_soc < 0:
                result_soc = 0
            state_of_charge_list.append(round(result_soc, 2))

            depth_of_discharge = 0
            if pdischarge == 0:
                #jika tidak ada listrik yang keluar dari baterai
                depth_of_discharge = 0
            elif result_soc < state_of_charge_list[i - 1]:
                #ada listrik yang keluar dari baterai karena presentase daya sekarang lebih kecil daripada daya sebelumnya
                depth_of_discharge = 1 - result_soc
            depth_of_discharge_list.append(depth_of_discharge)

            ncycle = 0
            if depth_of_discharge != 0:
                #ncycle: representase baterai bisa dipakai berapa kali. merepresentasikan waktu hidup dengan DoD
                power_of_ncyle = BETA_2 * (1 - depth_of_discharge)
                ncycle = BETA_0 * (depth_of_discharge ** (-BETA_1)) * (2.71828 ** power_of_ncyle)
            ncycle_list.append(ncycle)

            #one cycle degradation cost
            ce_soc = 0
            if ncycle != 0:
                ce_soc = CCAP / ncycle
            ce_soc_list.append(ce_soc)

            #degradation cost function
            cbt = 0
            if pdischarge != 0:
                cbt = ce_soc - ce_soc_list[i - 1]
            cbt_list.append(cbt)

        # print(len(depth_of_discharge_list), "DoD:", depth_of_discharge_list)
        # print(len(ncycle_list), "Ncyc: ", ncycle_list)
        # print(len(ce_soc_list), "CESoC: ", ce_soc_list)
        # print(len(cbt_list), "CBt: ", cbt_list)
        # print(len(state_of_charge_list), "Hourly Battery Percentage/SoC_BESS:", state_of_charge_list)

        max_pcharge = max(pcharge_list)
        max_pdischarge = max(pdischarge_list)
        pbess_max = round(max(max_pcharge, max_pdischarge), 2)
        total_pbess_list = sum(pbess_list)

        # print("PBESS_max/Daya: ", pbess_max)
        # print("PBESS total: ", round(total_pbess_list, 2))
        # print(len(excess_soc_list), "excess SoC: ", excess_soc_list)
        # print("excess SoC:", excess_soc_list[-1])
        # print(len(surge_soc_list), "surge SoC: ", surge_soc_list)
        # print("surge SoC: ", surge_soc_list[-1])

        #fitness equation (smaller better)
        cost_pv = (sg_total * cappv_desimal * PV_OPERATION_MAINTENANCE) + (PV_INSTALLATION_COST * cappv_desimal)
        installation_cost = (pbess_max * (POWER_RATING + BESS_OPERATION_MAINTENANCE)) + (ebess_desimal * (ENERGY_RATING + BESS_INSTALLATION_COST))
        degradation_cost = sum(cbt_list)
        cost_bess = installation_cost + degradation_cost
        total_cost = cost_pv + cost_bess

        #ada ketika soc bess dibawah 0, negative2 di sum lalu dikali 100 karena bentuk presentase
        #dikali ebess karena surge soc list itu presentase ebess
        penalty_srg = (abs(sum(surge_soc_list) * 100) * ebess_desimal * 3)

        #dikali ebess karena excess soc list itu presentase ebess
        penalty_exc = (abs(sum(excess_soc_list) * 100) * ebess_desimal * 0.11)

        #total cost inalizednya menyesuaikan penalty karena fitness function duit maka penalty dijadikan duit
        # tc pinalized dia akan bertambah menyesuaikan penalty exc dan penalty srg
        total_cost_pinalized = (total_cost + penalty_exc + penalty_srg)

        # print("Cpv: $", round(cost_pv, 2))
        # print("Installation Cost/IC: $", installation_cost)
        # print("Degradation Cost/DC: $", degradation_cost)
        # print("TC= $", total_cost)
        # print("TC_penalized= $", total_cost_pinalized)

        bool_srg = False
        for value in surge_soc_list:
            if value < 0:
                bool_srg = True
                break
            else: continue

        print(bool_srg)

        population_iterated.append({"fitness": total_cost_pinalized, "cappv_desimal": cappv_desimal, "cappv_binary": cappv_binary,  "ebess_desimal": ebess_desimal, "ebess_binary": ebess_binary, "erase": bool_srg, "cpv": cost_pv, "ic": installation_cost, "dc": degradation_cost, "cbess": cost_bess, "TC": total_cost})

    return population_iterated
