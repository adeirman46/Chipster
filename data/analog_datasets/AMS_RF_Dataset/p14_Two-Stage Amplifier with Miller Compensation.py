from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
circuit = Circuit('Two-Stage Amplifier with Miller Compensation')
# Define power supply
circuit.V('dd', 'Vdd', circuit.gnd, 5@u_V)
# Define bias voltage for active load
circuit.V('bias', 'Vbias', circuit.gnd, 2.5@u_V)
# Transistor models
circuit.model('nmos', 'nmos', level=1, vto=0.5, kp=100e-6)
circuit.model('pmos', 'pmos', level=1, vto=-0.5, kp=50e-6)
# First Stage: NMOS common-source with PMOS active load
# M1: NMOS input transistor
circuit.MOSFET('M1', 'Vmid', 'Vin', 'gnd', 'gnd', model='nmos', w=10e-6, l=1e-6)
# M2: PMOS active load
circuit.MOSFET('M2', 'Vmid', 'Vbias', 'Vdd', 'Vdd', model='pmos', w=20e-6, l=1e-6)
# Second Stage: NMOS common-source
circuit.MOSFET('M3', 'Vout', 'Vmid', 'gnd', 'gnd', model='nmos', w=10e-6, l=1e-6)
# Load resistor for second stage
circuit.R('load', 'Vout', 'Vdd', 10@u_kΩ)
# Miller Compensation Capacitor
circuit.C('miller', 'Vmid', 'Vout', 1@u_pF)
# Input source
circuit.V('in', 'Vin', circuit.gnd, "dc 1@u_V ac 1n")
# Connect all components properly
# (Connections are made via node names above)
simulator = circuit.simulator()

try:
    analysis = simulator.operating_point()
    fopen = open("p14_op.txt", "w")
    for node in analysis.nodes.values(): 
        fopen.write(f"{str(node)}\t{float(analysis[str(node)][0]):.6f}\n")
    fopen.close()
except Exception as e:
    print("Analysis failed due to an error:")
    print(str(e))

simulator_id = circuit.simulator()
mosfet_names = []
import PySpice.Spice.BasicElement
for element in circuit.elements:
    if isinstance(element, PySpice.Spice.BasicElement.Mosfet):
        mosfet_names.append(element.name)

mosfet_name_ids = []
for mosfet_name in mosfet_names:
    mosfet_name_ids.append(f"@{mosfet_name}[id]")

simulator_id.save_internal_parameters(*mosfet_name_ids)
analysis_id = simulator_id.operating_point()

id_correct = 1
for mosfet_name in mosfet_names:
    mosfet_id = float(analysis_id[f"@{mosfet_name}[id]"][0])
    if mosfet_id < 1e-5:
        id_correct = 0
        print("The circuit does not function correctly. "
          "the current I_D for {} is 0. ".format(mosfet_name)
          .format(mosfet_name))

if id_correct == 0:
    print("Please fix the wrong operating point.\n")
    sys.exit(2)


frequency = 100@u_Hz
analysis = simulator.ac(start_frequency=frequency, stop_frequency=frequency*10, 
    number_of_points=2, variation='dec')

import numpy as np

node = 'vout'

# find whether vout in the circuit

has_node = False
# find any node with "vout"
for element in circuit.elements:
    # get pins
    for pin in element.pins:
        if "vout" == str(pin.node).lower():
            node = str(pin.node)
            has_node = True
            break

if has_node == False:
    for element in circuit.elements:
        for pin in element.pins:
            if "vout" in str(pin.node).lower():
                node = str(pin.node)
                break

output_voltage = analysis[node].as_ndarray()[0]
gain = np.abs(output_voltage / (1e-9))

print(f"Voltage Gain (Av) at 100 Hz: {gain}")

required_gain = 1e-5
import sys
if gain > required_gain:
    print("The circuit functions correctly at 100 Hz.\n")
    sys.exit(0)
else:
    print("The circuit does not function correctly.\n"
          "the gain is less than 1e-5.\n"
          "Please fix the wrong operating point.\n")
    sys.exit(2)